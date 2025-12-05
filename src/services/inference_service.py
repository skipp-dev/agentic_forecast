"""
Inference Service

High-performance inference with GPU batching and model registry integration.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, List
import torch
import time
import json
from pathlib import Path

from src.gpu_services import GPUServices
from src.services.model_registry_service import ModelRegistryService
from models.model_zoo import DataSpec

# Conditional imports
try:
    from neuralforecast import NeuralForecast
    _HAS_NEURALFORECAST = True
except ImportError:
    _HAS_NEURALFORECAST = False

logger = logging.getLogger(__name__)

class InferenceService:
    """
    Service for running inference using trained models.
    Supports GPU acceleration and batch processing.
    """
    
    def __init__(self, gpu_services: Optional[GPUServices] = None, model_registry: Optional[ModelRegistryService] = None):
        self.gpu_services = gpu_services or GPUServices()
        self.model_registry = model_registry or ModelRegistryService()
        self._model_cache = {}
        
    def _load_model(self, model_id: str, symbol: str, model_type: str):
        """Load model from registry with caching."""
        cache_key = f"{symbol}_{model_type}_{model_id}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
            
        model_dir = self.model_registry.storage_path / symbol / model_type / model_id
        metadata_path = model_dir / "metadata.json"
        
        if not metadata_path.exists():
            raise ValueError(f"Model metadata not found for {model_id} at {metadata_path}")
            
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            raise
            
        artifact_path = Path(metadata.get('artifact_path', ''))
        framework = metadata.get('framework', 'pytorch')
        
        # Handle relative paths if needed (though we saved absolute)
        if not artifact_path.exists():
            # Try relative to model_dir
            artifact_path = model_dir / artifact_path.name
            if not artifact_path.exists():
                 # Try 'checkpoints' for NF
                 if framework == 'neuralforecast':
                     artifact_path = model_dir / "checkpoints"
                 
                 if not artifact_path.exists():
                    raise ValueError(f"Artifact not found at {artifact_path}")

        model = None
        try:
            if framework == 'neuralforecast':
                if _HAS_NEURALFORECAST:
                    # NeuralForecast.load expects a directory
                    model = NeuralForecast.load(str(artifact_path))
                else:
                    raise ImportError("NeuralForecast not installed")
            elif framework == 'sklearn':
                import joblib
                model = joblib.load(artifact_path)
            else:
                # Default to PyTorch
                model = torch.load(artifact_path)
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        self._model_cache[cache_key] = model
        return model

    def predict(self, 
                symbol: str, 
                model_id: str, 
                model_type: str, 
                data: pd.DataFrame, 
                horizon: int = 5) -> Dict[str, Any]:
        """
        Generate predictions.
        
        Args:
            symbol: Stock symbol.
            model_id: ID of the model to use.
            model_type: Type of model.
            data: Input DataFrame (history).
            horizon: Forecast horizon.
            
        Returns:
            Dict with predictions and metadata.
        """
        start_time = time.time()
        
        try:
            model = self._load_model(model_id, symbol, model_type)
            
            # Optimize GPU
            self.gpu_services.optimize_for_inference()
            
            predictions = None
            
            if _HAS_NEURALFORECAST and isinstance(model, NeuralForecast):
                # NeuralForecast predict
                # Ensure data has required cols
                df = data.copy()
                if 'ds' not in df.columns:
                    if isinstance(df.index, pd.DatetimeIndex):
                        df = df.reset_index()
                        df = df.rename(columns={df.columns[0]: 'ds'})
                if 'unique_id' not in df.columns:
                    df['unique_id'] = symbol
                if 'y' not in df.columns and 'close' in df.columns:
                    df['y'] = df['close']
                    
                # Predict
                # NeuralForecast.predict() generates forecasts for the next horizon steps
                # based on the input df as history.
                # Note: The model object loaded might have been trained on old data.
                # predict(df) uses the new df as history.
                forecasts = model.predict(df=df)
                predictions = forecasts
                
            elif isinstance(model, (torch.nn.Module, torch.ScriptModule)):
                # PyTorch model inference
                # This requires specific knowledge of input shape and preprocessing
                # For now, we assume it's not implemented generically
                logger.warning(f"Generic PyTorch inference not implemented for {type(model)}")
                return {'status': 'failed', 'error': 'Generic PyTorch inference not implemented'}
                
            else:
                # Sklearn or other models
                # Assume model has predict method
                if hasattr(model, 'predict'):
                    # Prepare X
                    # For BaselineLinear, we used time index
                    # We need to know how the model was trained.
                    # Ideally, metadata should contain feature info.
                    # For now, we assume simple time-based regression as in training service
                    
                    # If data has 'y', we might be predicting for validation
                    # If we want future, we need to extend the index
                    
                    # Let's assume we want to predict for the input data rows
                    X = np.arange(len(data)).reshape(-1, 1)
                    
                    # If we want to predict future (horizon), we need to know horizon
                    # But predict() usually takes input features.
                    # If the model expects future features, we need to generate them.
                    
                    # For BaselineLinear trained on time index, we can predict for the input range
                    preds = model.predict(X)
                    
                    # If we want future, we need to extend X
                    # But this method signature takes 'data' as input.
                    # Usually 'data' is history.
                    # If we want to forecast horizon steps ahead:
                    X_future = np.arange(len(data), len(data) + horizon).reshape(-1, 1)
                    preds_future = model.predict(X_future)
                    
                    # Construct result DataFrame
                    last_date = data['ds'].max() if 'ds' in data.columns else data.index.max()
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq='D')
                    
                    predictions = pd.DataFrame({
                        'ds': future_dates,
                        'unique_id': symbol,
                        model_type: preds_future
                    })
                else:
                    logger.warning(f"Model {type(model)} does not have predict method")
                    return {'status': 'failed', 'error': 'Model does not have predict method'}
                
            execution_time = time.time() - start_time
            
            return {
                'status': 'success',
                'symbol': symbol,
                'model_id': model_id,
                'predictions': predictions.to_dict(orient='records') if isinstance(predictions, pd.DataFrame) else predictions,
                'execution_time': execution_time
            }
            
        except Exception as e:
            logger.error(f"Inference failed for {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'status': 'failed', 'error': str(e)}
