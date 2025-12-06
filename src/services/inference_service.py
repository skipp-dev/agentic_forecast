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
from src.services.feature_store_service import FeatureStoreService
from src.data.types import DataSpec
from dataclasses import dataclass

# Conditional imports
try:
    from neuralforecast import NeuralForecast
    _HAS_NEURALFORECAST = True
except ImportError:
    _HAS_NEURALFORECAST = False

logger = logging.getLogger(__name__)

@dataclass
class InferenceRequest:
    symbol: str
    model_type: Optional[str] = None
    data: Optional[pd.DataFrame] = None
    model_id: Optional[str] = None

@dataclass
class InferenceResult:
    symbol: str
    prediction: float
    confidence: float
    model_id: str
    inference_time_ms: float

class InferenceService:
    """
    Service for running inference using trained models.
    Supports GPU acceleration and batch processing.
    """
    
    def __init__(self, gpu_services: Optional[GPUServices] = None, model_registry: Optional[ModelRegistryService] = None, feature_store_service: Optional[FeatureStoreService] = None):
        self.gpu_services = gpu_services or GPUServices()
        self.model_registry = model_registry or ModelRegistryService()
        self.feature_store_service = feature_store_service or FeatureStoreService()
        self._model_cache = {}
        
    def _generate_fallback_prediction(self, data: pd.DataFrame, horizon: int, symbol: str) -> Dict[str, Any]:
        """Generate a naive fallback prediction (last value carried forward)."""
        logger.warning(f"Generating fallback prediction for {symbol}")
        
        try:
            last_val = 0.0
            last_date = pd.Timestamp.now()
            
            if data is not None and not data.empty:
                # Try to find target column
                target_col = 'y' if 'y' in data.columns else 'close'
                if target_col in data.columns:
                    last_val = float(data[target_col].iloc[-1])
                
                # Try to find date column
                if 'ds' in data.columns:
                    last_date = pd.to_datetime(data['ds'].iloc[-1])
                elif isinstance(data.index, pd.DatetimeIndex):
                    last_date = data.index[-1]
            
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq='D')
            
            # Create predictions with generic column name 'y' or specific model type if needed
            # Using 'y_pred' or just 'prediction' might be safer, but let's stick to 'y' or 'prediction'
            # The caller expects a list of dicts.
            
            predictions = pd.DataFrame({
                'ds': future_dates,
                'unique_id': symbol,
                'prediction': [last_val] * horizon # Naive forecast
            })
            
            return {
                'status': 'success_fallback',
                'symbol': symbol,
                'model_id': 'fallback_naive',
                'predictions': predictions.to_dict(orient='records'),
                'execution_time': 0.0,
                'note': 'Fallback: Naive forecast used due to model failure'
            }
        except Exception as e:
            logger.error(f"Fallback generation failed: {e}")
            return {'status': 'failed', 'error': f"Fallback failed: {e}"}

    def _load_model(self, model_id: str, symbol: str, model_type: str):
        """Load model from registry with caching."""
        cache_key = f"{symbol}_{model_type}_{model_id}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
            
        try:
            # Use the ModelRegistryService to load the model (MLflow integration)
            model = self.model_registry.load_model(model_id)
            self._model_cache[cache_key] = model
            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise e

    def predict(self, 
                symbol: str, 
                model_id: Optional[str], 
                model_type: Optional[str], 
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
            # Resolve model_id if not provided
            if not model_id:
                # We need model_type to find the latest model
                if not model_type:
                     # Try to infer or list all models for symbol
                     models = self.model_registry.list_models(symbol=symbol)
                     if not models:
                         raise ValueError(f"No models found for {symbol}")
                     # Pick the most recent one
                     latest_model = models[0] # list_models sorts by start_time DESC
                     model_id = latest_model['model_id']
                     model_type = latest_model['model_type']
                else:
                    # Get latest for specific type
                    models = self.model_registry.list_models(symbol=symbol, model_type=model_type)
                    if not models:
                         raise ValueError(f"No models found for {symbol} and type {model_type}")
                    model_id = models[0]['model_id']

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
                
            elif hasattr(model, 'predict'):
                # Sklearn or other models (including MockModel which is also torch.nn.Module)
                # Assume model has predict method
                print(f"DEBUG: Checking predict method for model type: {type(model)}")
                print(f"DEBUG: Has predict: {hasattr(model, 'predict')}")
                
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

            elif isinstance(model, torch.nn.Module):
                # PyTorch model inference
                try:
                    # Prepare input tensor
                    if isinstance(data, pd.DataFrame):
                        # Drop non-feature columns if present
                        cols_to_drop = ['ds', 'unique_id', 'y', 'target']
                        feature_df = data.drop(columns=[c for c in cols_to_drop if c in data.columns])
                        # Ensure numeric
                        feature_df = feature_df.select_dtypes(include=[np.number])
                        input_data = feature_df.values.astype(np.float32)
                    elif isinstance(data, dict):
                        # Assume data is a dict of features or 'data' key
                        if 'data' in data:
                            input_data = np.array(data['data'], dtype=np.float32)
                        else:
                            # Try to convert dict to df then array
                            input_data = pd.DataFrame([data]).values.astype(np.float32)
                    else:
                        input_data = np.array(data, dtype=np.float32)

                    # Convert to tensor
                    input_tensor = torch.tensor(input_data)
                    
                    # Add batch dimension if needed
                    # If input is (seq, features), make it (1, seq, features)
                    if input_tensor.dim() == 2:
                        input_tensor = input_tensor.unsqueeze(0)
                    elif input_tensor.dim() == 1:
                        # (features,) -> (1, 1, features)
                        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
                    elif input_tensor.dim() == 0:
                        # Scalar -> (1, 1, 1)
                        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                    
                    # Move to device if needed (assuming CPU for inference service for now)
                    model.eval()
                    try:
                        with torch.no_grad():
                            output = model(input_tensor)
                    except RuntimeError as e:
                        if "input.size(-1) must be equal to input_size" in str(e):
                            # Try to fetch features
                            print(f"DEBUG: Shape mismatch for {symbol}. Attempting to fetch features from store.")
                            # We need symbol and date range
                            # Assuming data has 'ds'
                            if isinstance(data, pd.DataFrame) and 'ds' in data.columns:
                                start_date = pd.to_datetime(data['ds'].min())
                                end_date = pd.to_datetime(data['ds'].max())
                                
                                # Fetch features - try 'engineered' then 'technical'
                                features_df = self.feature_store_service.get_features(symbol, "engineered")
                                if features_df is None:
                                    print(f"DEBUG: 'engineered' features not found for {symbol}. Trying 'technical'.")
                                    features_df = self.feature_store_service.get_features(symbol, "technical")
                                    
                                if features_df is not None and not features_df.empty:
                                    print(f"DEBUG: Found features for {symbol}. Shape: {features_df.shape}")
                                    # Filter by date range
                                    # features_df index is DatetimeIndex
                                    mask = (features_df.index >= start_date) & (features_df.index <= end_date)
                                    filtered_features = features_df.loc[mask]
                                    
                                    if not filtered_features.empty:
                                        print(f"DEBUG: Filtered features shape: {filtered_features.shape}")
                                        # Use these features
                                        # Ensure numeric
                                        filtered_features = filtered_features.select_dtypes(include=[np.number])
                                        input_data = filtered_features.values.astype(np.float32)
                                        input_tensor = torch.tensor(input_data)
                                        
                                        # Add batch dimension
                                        if input_tensor.dim() == 2:
                                            input_tensor = input_tensor.unsqueeze(0)
                                        elif input_tensor.dim() == 1:
                                            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
                                        
                                        with torch.no_grad():
                                            output = model(input_tensor)
                                    else:
                                        print(f"DEBUG: No features found for {symbol} in range {start_date} - {end_date}")
                                        raise e
                                else:
                                    print(f"DEBUG: No features found for {symbol} in store")
                                    raise e
                            else:
                                print(f"DEBUG: Data is not DataFrame or missing 'ds'")
                                raise e
                        else:
                            raise e
                        
                    # Convert output to numpy
                    preds = output.numpy().flatten()
                    
                    # Construct result DataFrame
                    # We assume single step forecast for now as per LSTMModel architecture
                    last_date = pd.Timestamp.now()
                    if isinstance(data, pd.DataFrame) and 'ds' in data.columns:
                        last_date = pd.to_datetime(data['ds'].iloc[-1])
                    
                    future_date = last_date + pd.Timedelta(days=1)
                    
                    predictions = pd.DataFrame({
                        'ds': [future_date],
                        'unique_id': [symbol],
                        model_type: [preds[0]]
                    })
                    
                except Exception as e:
                    logger.error(f"PyTorch inference failed: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    raise e # Re-raise to trigger fallback
                
            else:
                logger.warning(f"Model {type(model)} does not have predict method")
                raise ValueError(f"Model {type(model)} does not have predict method")
                
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
            
            # Attempt fallback
            logger.info(f"Attempting fallback for {symbol}")
            return self._generate_fallback_prediction(data, horizon, symbol)

    async def predict_async(self, request: InferenceRequest) -> InferenceResult:
        """
        Async wrapper for predict method.
        """
        import asyncio
        # Run synchronous predict in a thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.predict, request.symbol, request.model_id, request.model_type, request.data)
        
        # Convert dict result to InferenceResult
        if result['status'] in ['success', 'success_fallback']:
            preds = result['predictions']
            # Extract a single prediction value for simplicity, or handle full dataframe
            prediction_val = 0.0
            if isinstance(preds, list) and len(preds) > 0:
                # Assuming list of dicts
                first_pred = preds[0]
                # Find the prediction column (not 'ds' or 'unique_id')
                for k, v in first_pred.items():
                    if k not in ['ds', 'unique_id']:
                        prediction_val = float(v)
                        break
            
            # Lower confidence for fallback
            confidence = 0.95 if result['status'] == 'success' else 0.5
            
            return InferenceResult(
                symbol=request.symbol,
                prediction=prediction_val,
                confidence=confidence,
                model_id=result.get('model_id', 'unknown'),
                inference_time_ms=result.get('execution_time', 0.0) * 1000
            )
        else:
            raise RuntimeError(f"Inference failed: {result.get('error')}")
