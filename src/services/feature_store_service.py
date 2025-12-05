"""
Feature Store Service

Efficient time-series feature storage and retrieval with optional Redis caching.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime
from pathlib import Path
import json

# Optional Redis import
try:
    import redis
    _HAS_REDIS = True
except ImportError:
    _HAS_REDIS = False

logger = logging.getLogger(__name__)

class FeatureStoreService:
    """
    Service for storing and retrieving time-series features.
    Uses Parquet for persistence and optional Redis for caching.
    """
    
    def __init__(self, storage_path: str = "data/feature_store", redis_url: Optional[str] = None):
        """
        Initialize the feature store service.
        
        Args:
            storage_path: Path to store Parquet files.
            redis_url: URL for Redis connection (e.g., "redis://localhost:6379/0").
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.redis_client = None
        if _HAS_REDIS and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                logger.info(f"Connected to Redis at {redis_url}")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. Caching disabled.")
                self.redis_client = None
        elif redis_url and not _HAS_REDIS:
            logger.warning("Redis URL provided but 'redis' package not installed.")
            
    def _get_file_path(self, symbol: str, feature_type: str) -> Path:
        """Get the file path for a symbol and feature type."""
        type_dir = self.storage_path / feature_type
        type_dir.mkdir(exist_ok=True)
        return type_dir / f"{symbol}.parquet"
        
    def _get_cache_key(self, symbol: str, feature_type: str) -> str:
        return f"features:{symbol}:{feature_type}"

    def store_features(self, symbol: str, features: pd.DataFrame, feature_type: str = "technical") -> bool:
        """
        Store features for a symbol.
        
        Args:
            symbol: Stock symbol.
            features: DataFrame containing features.
            feature_type: Type of features (e.g., 'technical', 'spectral').
            
        Returns:
            bool: Success status.
        """
        try:
            # Ensure DatetimeIndex
            if not isinstance(features.index, pd.DatetimeIndex):
                if 'date' in features.columns:
                    features = features.set_index('date')
                    features.index = pd.to_datetime(features.index)
                elif 'ds' in features.columns:
                    features = features.set_index('ds')
                    features.index = pd.to_datetime(features.index)
            
            # 1. Save to Parquet (Persistence)
            file_path = self._get_file_path(symbol, feature_type)
            features.to_parquet(file_path)
            
            # 2. Cache in Redis (Fast Access)
            if self.redis_client:
                key = self._get_cache_key(symbol, feature_type)
                # Store as JSON or msgpack. JSON is easier for now.
                # Serialize DataFrame to JSON
                json_data = features.to_json(date_format='iso', orient='split')
                self.redis_client.set(key, json_data, ex=3600) # Expire in 1 hour
                
            logger.info(f"Stored {feature_type} features for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store features for {symbol}: {e}")
            return False

    def get_features(self, symbol: str, feature_type: str = "technical") -> Optional[pd.DataFrame]:
        """
        Retrieve features for a symbol.
        
        Args:
            symbol: Stock symbol.
            feature_type: Type of features.
            
        Returns:
            DataFrame or None if not found.
        """
        # 1. Try Redis Cache
        if self.redis_client:
            try:
                key = self._get_cache_key(symbol, feature_type)
                cached_data = self.redis_client.get(key)
                if cached_data:
                    df = pd.read_json(cached_data, orient='split')
                    if isinstance(df.index, pd.DatetimeIndex):
                        df.index.name = 'date'
                    return df
            except Exception as e:
                logger.warning(f"Redis cache miss/error for {symbol}: {e}")
        
        # 2. Try Parquet File
        file_path = self._get_file_path(symbol, feature_type)
        if file_path.exists():
            try:
                df = pd.read_parquet(file_path)
                
                # Update Cache
                if self.redis_client:
                    try:
                        json_data = df.to_json(date_format='iso', orient='split')
                        key = self._get_cache_key(symbol, feature_type)
                        self.redis_client.set(key, json_data, ex=3600)
                    except Exception as e:
                        logger.warning(f"Failed to update cache for {symbol}: {e}")
                        
                return df
            except Exception as e:
                logger.error(f"Failed to read parquet for {symbol}: {e}")
                return None
        
        return None
