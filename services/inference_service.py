"""
Inference Service

High-performance model inference service with GPU acceleration.
Provides real-time prediction capabilities for the IB Forecast system.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
import asyncio
import time
from dataclasses import dataclass
import json

# Add paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from services.model_registry_service import ModelRegistryService
from agents.feature_engineer_agent import FeatureEngineerAgent
from src.gpu_services import get_gpu_services

logger = logging.getLogger(__name__)

@dataclass
class InferenceRequest:
    """Inference request data."""
    symbol: str
    model_id: Optional[str] = None
    features: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None

@dataclass
@dataclass
class InferenceResult:
    """Inference result data."""
    symbol: str
    prediction: float
    confidence: float
    model_id: str
    model_version: str
    features_used: List[str]
    inference_time: float
    timestamp: datetime

class InferenceService:
    """
    High-performance inference service.

    Features:
    - GPU-accelerated inference
    - Batch processing capabilities
    - Model caching and warm-up
    - Real-time performance monitoring
    - A/B testing support
    - Async inference processing
    """

    def __init__(self, model_registry: Optional[ModelRegistryService] = None,
                 feature_agent: Optional[FeatureEngineerAgent] = None,
                 gpu_services=None):
        """
        Initialize inference service.

        Args:
            model_registry: Model registry service instance
            feature_agent: Feature engineering agent instance
            gpu_services: GPU services instance
        """
        self.model_registry = model_registry or ModelRegistryService()
        self.feature_agent = feature_agent or FeatureEngineerAgent()
        self.gpu_services = gpu_services or get_gpu_services()

        # Model cache
        self.model_cache = {}
        self.cache_timestamps = {}

        # Performance monitoring
        self.inference_stats = {
            'total_requests': 0,
            'total_inference_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0
        }

        # GPU device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Warm up models
        self._warm_up_models()

        logger.info("Inference Service initialized")

    async def predict_async(self, request: InferenceRequest) -> InferenceResult:
        """
        Async prediction for a single symbol.

        Args:
            request: Inference request

        Returns:
            Inference result
        """
        start_time = time.time()

        try:
            # Get or prepare features
            if request.features is None:
                features_df = await self._prepare_features_async(request.symbol)
            else:
                features_df = pd.DataFrame([request.features])

            # Get model
            model_id = request.model_id or self._select_best_model(request.symbol)
            if not model_id:
                raise ValueError(f"No model available for {request.symbol}")

            model, metadata = self._load_cached_model(model_id)

            # Prepare input
            feature_names = metadata.feature_names
            missing_features = set(feature_names) - set(features_df.columns)
            if missing_features:
                logger.warning(f"Missing features for {request.symbol}: {missing_features}")

            # Fill missing features with 0
            for feature in missing_features:
                features_df[feature] = 0.0

            input_features = features_df[feature_names].values

            # GPU inference
            if self.device == 'cuda' and hasattr(model, 'to'):
                model = model.to(self.device)
                input_tensor = torch.tensor(input_features, dtype=torch.float32).to(self.device)
            else:
                input_tensor = torch.tensor(input_features, dtype=torch.float32)

            # Perform inference
            model.eval()
            with torch.no_grad():
                if hasattr(model, 'lstm'):  # LSTM model
                    # Add sequence dimension for LSTM
                    input_tensor = input_tensor.unsqueeze(1)
                elif hasattr(model, 'transformer_encoder'):  # Transformer model
                    input_tensor = input_tensor.unsqueeze(1)

                prediction = model(input_tensor).item()

            # Calculate confidence (simplified)
            confidence = self._calculate_confidence(prediction, metadata)

            inference_time = time.time() - start_time

            # Update stats
            self.inference_stats['total_requests'] += 1
            self.inference_stats['total_inference_time'] += inference_time

            result = InferenceResult(
                symbol=request.symbol,
                prediction=prediction,
                confidence=confidence,
                model_id=model_id,
                model_version=metadata.version,
                features_used=feature_names,
                inference_time=inference_time,
                timestamp=datetime.now()
            )

            logger.info(f"Inference completed for {request.symbol}: {prediction:.4f}")

            return result

        except Exception as e:
            self.inference_stats['errors'] += 1
            logger.error(f"Inference failed for {request.symbol}: {e}")
            raise

    def predict_batch(self, requests: List[InferenceRequest]) -> List[InferenceResult]:
        """
        Batch prediction for multiple symbols.

        Args:
            requests: List of inference requests

        Returns:
            List of inference results
        """
        logger.info(f"Processing batch inference for {len(requests)} requests")

        # Group by model for efficient batching
        model_groups = {}
        for request in requests:
            model_id = request.model_id or self._select_best_model(request.symbol)
            if model_id not in model_groups:
                model_groups[model_id] = []
            model_groups[model_id].append(request)

        results = []

        for model_id, model_requests in model_groups.items():
            try:
                model_results = self._predict_batch_for_model(model_id, model_requests)
                results.extend(model_results)
            except Exception as e:
                logger.error(f"Batch prediction failed for model {model_id}: {e}")
                # Add error results
                for request in model_requests:
                    results.append(InferenceResult(
                        symbol=request.symbol,
                        prediction=0.0,
                        confidence=0.0,
                        model_id=model_id,
                        model_version='unknown',
                        features_used=[],
                        inference_time=0.0,
                        timestamp=datetime.now()
                    ))

        return results

    def _predict_batch_for_model(self, model_id: str,
                                requests: List[InferenceRequest]) -> List[InferenceResult]:
        """Batch prediction for a specific model."""
        model, metadata = self._load_cached_model(model_id)
        feature_names = metadata.feature_names

        # Prepare batch features
        batch_features = []
        valid_requests = []

        for request in requests:
            try:
                if request.features is None:
                    # Synchronous feature preparation for batch
                    features_df = self._prepare_features_sync(request.symbol)
                else:
                    features_df = pd.DataFrame([request.features])

                # Handle missing features
                missing_features = set(feature_names) - set(features_df.columns)
                for feature in missing_features:
                    features_df[feature] = 0.0

                input_features = features_df[feature_names].iloc[0].values
                batch_features.append(input_features)
                valid_requests.append(request)

            except Exception as e:
                logger.warning(f"Feature preparation failed for {request.symbol}: {e}")
                continue

        if not batch_features:
            return []

        # Convert to tensor
        batch_tensor = torch.tensor(np.array(batch_features), dtype=torch.float32)

        if self.device == 'cuda' and hasattr(model, 'to'):
            model = model.to(self.device)
            batch_tensor = batch_tensor.to(self.device)

        start_time = time.time()

        # Batch inference
        model.eval()
        with torch.no_grad():
            if hasattr(model, 'lstm'):  # LSTM model
                batch_tensor = batch_tensor.unsqueeze(1)
            elif hasattr(model, 'transformer_encoder'):  # Transformer model
                batch_tensor = batch_tensor.unsqueeze(1)

            predictions = model(batch_tensor).squeeze().cpu().numpy()

        inference_time = time.time() - start_time

        # Create results
        results = []
        for i, (request, prediction) in enumerate(zip(valid_requests, predictions)):
            if np.isscalar(predictions):
                prediction = predictions

            confidence = self._calculate_confidence(prediction, metadata)

            result = InferenceResult(
                symbol=request.symbol,
                prediction=float(prediction),
                confidence=confidence,
                model_id=model_id,
                model_version=metadata.version,
                features_used=feature_names,
                inference_time=inference_time / len(valid_requests),  # Average time
                timestamp=datetime.now()
            )
            results.append(result)

        return results

    async def _prepare_features_async(self, symbol: str) -> pd.DataFrame:
        """Async feature preparation."""
        # For now, use sync method (can be made async later)
        return self._prepare_features_sync(symbol)

    def _prepare_features_sync(self, symbol: str) -> pd.DataFrame:
        """Synchronous feature preparation."""
        try:
            # Get latest market data
            raw_data = self.feature_agent.data_pipeline.av_client.get_daily_data(symbol, outputsize='full')
            if raw_data.empty:
                raise ValueError(f"No data available for {symbol}")

            data_df = pd.DataFrame(raw_data)

            # Engineer features
            features_df = self.feature_agent.engineer_features(
                symbol, data_df, feature_sets=['basic', 'spectral']
            )

            # Return latest row
            return features_df.tail(1)

        except Exception as e:
            logger.error(f"Feature preparation failed for {symbol}: {e}")
            raise

    def _select_best_model(self, symbol: str) -> Optional[str]:
        """Select the best available model for a symbol."""
        # Get all active models for the symbol
        models = self.model_registry.list_models(symbol=symbol, status='active')

        if not models:
            # Try any model for the symbol
            models = self.model_registry.list_models(symbol=symbol)

        if not models:
            return None

        # Select model with best performance (lowest MAE)
        best_model = min(models, key=lambda m: m.performance_metrics.get('mae', float('inf')))

        return best_model.model_id

    def _load_cached_model(self, model_id: str) -> Tuple[Any, Any]:
        """Load model from cache or registry."""
        current_time = time.time()

        # Check cache
        if model_id in self.model_cache:
            self.inference_stats['cache_hits'] += 1
            self.cache_timestamps[model_id] = current_time
            return self.model_cache[model_id], self.model_cache[f'{model_id}_metadata']
        else:
            self.inference_stats['cache_misses'] += 1

        # Load from registry
        model = self.model_registry.load_model(model_id)
        metadata = self.model_registry.get_model_metadata(model_id)

        if not model or not metadata:
            raise ValueError(f"Model {model_id} not found in registry")

        # Cache model
        self.model_cache[model_id] = model
        self.model_cache[f'{model_id}_metadata'] = metadata
        self.cache_timestamps[model_id] = current_time

        # Limit cache size (keep 10 most recent)
        if len(self.model_cache) > 20:  # 10 models + 10 metadata
            oldest_id = min(self.cache_timestamps.items(), key=lambda x: x[1])[0]
            # Remove model and metadata
            model_base = oldest_id.replace('_metadata', '')
            if model_base in self.model_cache:
                del self.model_cache[model_base]
            if f'{model_base}_metadata' in self.model_cache:
                del self.model_cache[f'{model_base}_metadata']
            del self.cache_timestamps[oldest_id]

        return model, metadata

    def _calculate_confidence(self, prediction: float, metadata: Any) -> float:
        """Calculate prediction confidence."""
        # Simplified confidence based on model performance
        mae = metadata.performance_metrics.get('mae', 0.1)
        rmse = metadata.performance_metrics.get('rmse', 0.15)

        # Confidence decreases with model error
        base_confidence = max(0.1, 1.0 - (mae + rmse) / 2)

        # Adjust based on prediction magnitude (extreme predictions less confident)
        magnitude_penalty = min(1.0, abs(prediction) / 0.5)  # Penalize >50% predictions

        return base_confidence * (1.0 / magnitude_penalty)

    def _warm_up_models(self):
        """Warm up frequently used models."""
        try:
            # Load top 3 models by usage (simplified)
            all_models = self.model_registry.list_models(status='active')
            top_models = all_models[:3]  # Just load first 3

            for metadata in top_models:
                try:
                    self._load_cached_model(metadata.model_id)
                    logger.info(f"Warmed up model: {metadata.model_id}")
                except Exception as e:
                    logger.warning(f"Failed to warm up {metadata.model_id}: {e}")

        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get inference performance statistics."""
        stats = self.inference_stats.copy()

        if stats['total_requests'] > 0:
            stats['avg_inference_time'] = stats['total_inference_time'] / stats['total_requests']
            stats['cache_hit_rate'] = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
            stats['error_rate'] = stats['errors'] / stats['total_requests']

        return stats

    def clear_cache(self):
        """Clear model cache."""
        self.model_cache.clear()
        self.cache_timestamps.clear()
        logger.info("Model cache cleared")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            # Test basic inference
            test_request = InferenceRequest(symbol='AAPL')
            result = await self.predict_async(test_request)

            return {
                'status': 'healthy',
                'gpu_available': self.device == 'cuda',
                'cache_size': len(self.model_cache) // 2,  # Divide by 2 for model/metadata pairs
                'test_inference_time': result.inference_time
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }

# Convenience functions
def create_inference_service():
    """Create and configure inference service."""
    return InferenceService()

async def predict_symbol_async(symbol: str, model_id: Optional[str] = None) -> InferenceResult:
    """Async prediction for a symbol with default settings."""
    service = create_inference_service()
    request = InferenceRequest(symbol=symbol, model_id=model_id)
    return await service.predict_async(request)

def predict_symbols_batch(symbols: List[str]) -> List[InferenceResult]:
    """Batch prediction for multiple symbols."""
    service = create_inference_service()
    requests = [InferenceRequest(symbol=symbol) for symbol in symbols]
    return service.predict_batch(requests)