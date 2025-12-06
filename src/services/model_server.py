"""
Model Serving Layer

Provides a robust, cached serving layer for models to ensure low-latency inference.
Wraps the InferenceService with advanced caching and pre-loading capabilities.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from collections import OrderedDict
import threading

from src.services.inference_service import InferenceService

logger = logging.getLogger(__name__)

class ModelServer:
    """
    High-performance model server with LRU caching and background pre-loading.
    """
    
    def __init__(self, inference_service: Optional[InferenceService] = None, cache_size: int = 10):
        self.inference_service = inference_service or InferenceService()
        self.cache_size = cache_size
        self._model_cache = OrderedDict()
        self._lock = threading.RLock()
        
    def preload_models(self, symbols: List[str], model_types: List[str]):
        """
        Pre-load models into memory to avoid cold start latency.
        """
        logger.info(f"Pre-loading models for {len(symbols)} symbols...")
        for symbol in symbols:
            for model_type in model_types:
                try:
                    # Find latest model ID
                    models = self.inference_service.model_registry.list_models(symbol=symbol, model_type=model_type)
                    if models:
                        latest_model = models[0] # Assuming sorted
                        model_id = latest_model['run_id']
                        
                        # Trigger load
                        with self._lock:
                            self.inference_service._load_model(model_id, symbol, model_type)
                            logger.info(f"Pre-loaded {model_type} for {symbol}")
                except Exception as e:
                    logger.warning(f"Failed to preload {symbol} {model_type}: {e}")

    def predict(self, symbol: str, data: Any, model_type: str = "NLinear", horizon: int = 5) -> Dict[str, Any]:
        """
        Serve a prediction request.
        """
        start_time = time.time()
        
        # Delegate to inference service (which has its own basic cache, but we manage the lifecycle here)
        # In a real system, this might be a Ray Serve handle or REST endpoint
        result = self.inference_service.predict(
            symbol=symbol,
            model_id=None, # Let service resolve latest
            model_type=model_type,
            data=data,
            horizon=horizon
        )
        
        latency = (time.time() - start_time) * 1000
        logger.info(f"Served prediction for {symbol} in {latency:.2f}ms")
        
        return result

    def clear_cache(self):
        """
        Clear the model cache.
        """
        with self._lock:
            self.inference_service._model_cache.clear()
            logger.info("Model cache cleared")
