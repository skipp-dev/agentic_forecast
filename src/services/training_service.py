"""
GPU Training Service

Encapsulates model training logic with GPU acceleration and model registry integration.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, List
import torch
import os
import time

from src.gpu_services import GPUServices
from src.services.model_registry_service import ModelRegistryService
from models.model_zoo import DataSpec
from sklearn.linear_model import LinearRegression
from src.agents.graph_model_agent import GraphModelAgent, GraphTrainingData

# Conditional imports for NeuralForecast
try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NLinear, NHITS, NBEATS, TFT
    from neuralforecast.auto import AutoDLinear, AutoNHITS, AutoTFT, AutoNBEATS
    from neuralforecast.losses.pytorch import MAE
    _HAS_NEURALFORECAST = True
except ImportError:
    _HAS_NEURALFORECAST = False

logger = logging.getLogger(__name__)

class GPUTrainingService:
    """
    Service for training models using GPU resources.
    """
    
    def __init__(self, gpu_services: Optional[GPUServices] = None, model_registry: Optional[ModelRegistryService] = None):
        self.gpu_services = gpu_services or GPUServices()
        self.model_registry = model_registry or ModelRegistryService()
        
    def _prepare_df(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Ensures DataFrame has required columns for NeuralForecast."""
        df = df.copy()
        if 'ds' not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                # Assuming index became first col, rename it
                df = df.rename(columns={df.columns[0]: 'ds'}) 
            else:
                 # Try to find date column
                 date_cols = [c for c in df.columns if 'date' in c.lower()]
                 if date_cols:
                     df = df.rename(columns={date_cols[0]: 'ds'})
                 else:
                     # If no date column, create one if it's for sklearn
                     pass
        
        if 'unique_id' not in df.columns:
            df['unique_id'] = symbol
            
        if 'y' not in df.columns:
            if 'close' in df.columns:
                df['y'] = df['close']
            else:
                # For sklearn, we might not need 'y' if we pass X and y separately, but here we assume df has target
                pass
        return df

    def train_model(self, 
                    symbol: str, 
                    model_type: str, 
                    data: Union[pd.DataFrame, DataSpec], 
                    hyperparams: Optional[Dict[str, Any]] = None,
                    job_id: str = "manual_train") -> Dict[str, Any]:
        """
        Train a model for a specific symbol.
        
        Args:
            symbol: The symbol to train on.
            model_type: The type of model (e.g., 'NLinear', 'LSTM', 'BaselineLinear').
            data: Training data (DataFrame or DataSpec).
            hyperparams: Hyperparameters for the model.
            job_id: Identifier for the training job.
            
        Returns:
            Dict containing training results and model ID.
        """
        start_time = time.time()
        logger.info(f"Starting training for {symbol} using {model_type} on device {self.gpu_services.device}")
        
        # Prepare Data
        train_df = None
        val_df = None
        
        if isinstance(data, DataSpec):
            train_df = self._prepare_df(data.train_df, symbol)
            if data.val_df is not None and not data.val_df.empty:
                val_df = self._prepare_df(data.val_df, symbol)
        elif isinstance(data, pd.DataFrame):
            train_df = self._prepare_df(data, symbol)
        else:
            raise ValueError("Data must be a DataFrame or DataSpec")

        # Handle Graph STGCNN
        if model_type == "graph_stgcnn":
            try:
                # Check if we have graph data in DataSpec
                if not isinstance(data, DataSpec) or not hasattr(data, 'edge_index') or data.edge_index is None:
                    return {'status': 'skipped', 'error': 'Graph data (edge_index) not provided'}
                
                # Prepare GraphTrainingData
                # We need to construct adjacency matrix and features
                # This assumes data.node_features and data.edge_index are available
                
                # Construct adjacency
                # Assuming edge_index is [2, num_edges]
                # We need symbol_to_idx to know size
                if not hasattr(data, 'symbol_to_idx') or data.symbol_to_idx is None:
                     return {'status': 'skipped', 'error': 'symbol_to_idx not provided'}
                
                num_nodes = len(data.symbol_to_idx)
                adj = np.zeros((num_nodes, num_nodes))
                
                edge_index = data.edge_index
                if hasattr(edge_index, 'cpu'):
                    edge_index = edge_index.cpu().numpy()
                
                for i in range(edge_index.shape[1]):
                    src, dst = edge_index[0, i], edge_index[1, i]
                    adj[src, dst] = 1
                    
                # Prepare features
                # data.node_features is likely [num_nodes, horizon] or similar
                features = data.node_features
                if hasattr(features, 'cpu'):
                    features = features.cpu().numpy()
                
                # Reshape to [time, nodes, features] if needed
                # STGCNN expects [time, nodes, features]
                if len(features.shape) == 2:
                    features = features[np.newaxis, :, :]
                
                training_data = GraphTrainingData(
                    symbols=list(data.symbol_to_idx.keys()),
                    times=[], # Dummy
                    features=features,
                    adjacency=adj
                )
                
                agent = GraphModelAgent()
                agent.fit(training_data)
                
                # Save model
                # We save it via registry but GraphModelAgent has its own save
                # We can save the agent or the internal model
                # Let's save the internal model via registry
                
                execution_time = time.time() - start_time
                
                metadata = {
                    'metrics': {}, # TODO: Extract metrics
                    'hyperparameters': hyperparams or {},
                    'training_config': {'job_id': job_id},
                    'execution_time': execution_time
                }
                
                # We need to wrap the model or save the agent's model
                # GraphModelAgent.model is the internal model
                
                model_id = self.model_registry.save_model(
                    model=agent.model,
                    symbol=symbol, # Note: Graph model is usually global, but we save under the requested symbol or 'global'
                    model_type=model_type,
                    metadata=metadata,
                    framework="pytorch"
                )
                
                return {
                    'status': 'success',
                    'model_id': model_id,
                    'execution_time': execution_time,
                    'metrics': {}
                }
                
            except Exception as e:
                logger.error(f"Training failed for {symbol} (graph_stgcnn): {e}")
                return {'status': 'failed', 'error': str(e)}

        # Handle BaselineLinear (Sklearn)
        if model_type == "BaselineLinear":
            try:
                # Prepare X and y
                # Simple autoregressive or time-based
                # For simplicity, let's use time index as feature like in execution_nodes fallback
                X_train = np.arange(len(train_df)).reshape(-1, 1)
                y_train = train_df['y'].values
                
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                metrics = {}
                if val_df is not None:
                    X_val = np.arange(len(train_df), len(train_df) + len(val_df)).reshape(-1, 1)
                    y_pred = model.predict(X_val)
                    y_true = val_df['y'].values
                    
                    metrics['mae'] = float(np.mean(np.abs(y_true - y_pred)))
                    metrics['mse'] = float(np.mean((y_true - y_pred)**2))
                
                execution_time = time.time() - start_time
                
                metadata = {
                    'metrics': metrics,
                    'hyperparameters': hyperparams or {},
                    'training_config': {'job_id': job_id},
                    'execution_time': execution_time
                }
                
                model_id = self.model_registry.save_model(
                    model=model,
                    symbol=symbol,
                    model_type=model_type,
                    metadata=metadata,
                    framework="sklearn"
                )
                
                return {
                    'status': 'success',
                    'model_id': model_id,
                    'execution_time': execution_time,
                    'metrics': metrics
                }
                
            except Exception as e:
                logger.error(f"Training failed for {symbol} (BaselineLinear): {e}")
                return {'status': 'failed', 'error': str(e)}

        if not _HAS_NEURALFORECAST:
            return {'status': 'failed', 'error': 'NeuralForecast not installed'}

        # Optimize GPU for training
        self.gpu_services.optimize_for_training()
        
        try:
            # Configure Hyperparams
            params = hyperparams or {}
            
            # Default GPU settings
            if self.gpu_services.device.type == 'cuda':
                params['accelerator'] = 'gpu'
                params['devices'] = 1 
            
            # Instantiate Model
            horizon = params.pop('horizon', 5) # Default horizon
            input_size = params.pop('input_size', 2 * horizon)
            
            model_cls = None
            if model_type == "NLinear":
                model_cls = NLinear
            elif model_type == "NHITS":
                model_cls = NHITS
            elif model_type == "NBEATS":
                model_cls = NBEATS
            elif model_type == "TFT":
                model_cls = TFT
            elif model_type == "AutoDLinear":
                model_cls = AutoDLinear
            elif model_type == "AutoNHITS":
                model_cls = AutoNHITS
            elif model_type == "AutoTFT":
                model_cls = AutoTFT
            elif model_type == "AutoNBEATS":
                model_cls = AutoNBEATS
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
            # Auto models might have different init signatures (config vs h/input_size)
            # NeuralForecast Auto models usually take 'h' and 'config' or 'search_alg'
            # Standard models take 'h', 'input_size'
            
            if model_type.startswith("Auto"):
                # Auto models
                # They typically accept 'h', 'loss', 'config', 'search_alg', 'num_samples'
                # We map hyperparams to config if needed
                
                # Extract standard params
                h = horizon
                
                # Prepare config for Ray Tune if passed
                # For now, we pass kwargs directly, assuming they match AutoModel signature
                # or we construct a minimal set
                
                # Common Auto params
                auto_kwargs = {
                    'h': h,
                    'loss': MAE(), # Default loss
                }
                
                # Add other params from hyperparams
                # e.g. num_samples, cpus, gpus
                if 'num_samples' in params:
                    auto_kwargs['num_samples'] = params['num_samples']
                
                # Pass remaining params as config or direct args
                # Auto models are flexible
                # We'll pass the rest of params into the constructor
                # But we need to be careful about 'input_size' which Auto models might determine or accept
                
                # Let's just pass **params combined with defaults, but filter out known conflicts
                # For simplicity, we instantiate with minimal required and let Auto handle defaults
                
                model = model_cls(**auto_kwargs)
            else:
                # Standard models
                model = model_cls(h=horizon, input_size=input_size, **params)
            
            # Train
            nf = NeuralForecast(models=[model], freq='D')
            nf.fit(df=train_df)
            
            # Evaluate if validation data is available
            metrics = {}
            if val_df is not None:
                # Predict on validation set
                # NeuralForecast predict uses the history in the model to predict future
                # We need to make sure the model has seen the training data (it has, via fit)
                # And we want to predict the horizon for the validation period.
                # However, predict() usually generates forecasts for the next 'h' steps after the end of input df.
                # If we want to evaluate on val_df, we might need to do a rolling forecast or just predict the immediate future if val_df is the immediate future.
                
                # For simplicity, we'll assume val_df follows train_df immediately.
                # We'll predict 'h' steps ahead.
                forecasts = nf.predict()
                
                # We need to compare forecasts with val_df.
                # forecasts has 'ds', 'unique_id', 'model_name'.
                # val_df has 'ds', 'unique_id', 'y'.
                
                # Merge
                merged = pd.merge(val_df, forecasts, on=['ds', 'unique_id'], how='inner')
                
                if not merged.empty:
                    y_true = merged['y'].values
                    y_pred = merged[model_type].values
                    
                    # Calculate MAE
                    mae = np.mean(np.abs(y_true - y_pred))
                    metrics['mae'] = float(mae)
                    
                    # Calculate MSE
                    mse = np.mean((y_true - y_pred)**2)
                    metrics['mse'] = float(mse)
                    
                    # Calculate MAPE
                    # Avoid division by zero
                    mask = y_true != 0
                    if np.any(mask):
                        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
                        metrics['mape'] = float(mape)
                    else:
                        metrics['mape'] = None
                    
                    logger.info(f"Evaluation metrics for {symbol}: {metrics}")
                else:
                    logger.warning(f"No overlap between forecasts and validation data for {symbol}")

            execution_time = time.time() - start_time
            
            metadata = {
                'metrics': metrics,
                'hyperparameters': params,
                'training_config': {
                    'job_id': job_id, 
                    'horizon': horizon,
                    'input_size': input_size
                },
                'execution_time': execution_time
            }
            
            model_id = self.model_registry.save_model(
                model=nf,
                symbol=symbol,
                model_type=model_type,
                metadata=metadata,
                framework="neuralforecast"
            )
            
            logger.info(f"Training completed for {symbol}. Model ID: {model_id}")
            
            return {
                'status': 'success',
                'model_id': model_id,
                'execution_time': execution_time,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Training failed for {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'status': 'failed', 'error': str(e)}
        finally:
            # Cleanup
            torch.cuda.empty_cache()
