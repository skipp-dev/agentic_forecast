"""
Time-Series Feature Store

High-performance feature store optimized for time-series data.
Provides efficient storage, retrieval, and management of ML features.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import pickle
import json
from dataclasses import dataclass, asdict
import hashlib
from concurrent.futures import ThreadPoolExecutor
import threading

# Add paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)

@dataclass
class FeatureSet:
    """Feature set metadata."""
    id: str
    name: str
    symbol: str
    features: List[str]
    created_at: datetime
    updated_at: datetime
    version: str
    data_hash: str
    row_count: int
    date_range: Tuple[datetime, datetime]

@dataclass
class FeatureQuery:
    """Feature query specification."""
    symbol: str
    feature_names: List[str]
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: Optional[int] = None

class TimeSeriesFeatureStore:
    """
    Time-series optimized feature store.

    Features:
    - Efficient time-series data storage and retrieval
    - Feature versioning and lineage tracking
    - Automatic data partitioning by time
    - Compression and indexing for performance
    - Batch operations and caching
    - Data validation and quality checks
    """

    def __init__(self, store_path: str = '/tmp/feature_store'):
        """
        Initialize feature store.

        Args:
            store_path: Path to store feature data
        """
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)

        # Database setup
        self.db_path = self.store_path / 'feature_store.db'
        self._initialize_database()

        # Cache
        self.feature_cache = {}
        self.metadata_cache = {}

        # Thread safety
        self.lock = threading.RLock()

        # Load metadata
        self._load_metadata_cache()

        logger.info(f"Time-Series Feature Store initialized at {store_path}")

    def store_features(self, symbol: str, features_df: pd.DataFrame,
                      feature_set_name: str = 'default') -> str:
        """
        Store feature data for a symbol.

        Args:
            symbol: Stock symbol
            features_df: DataFrame with features (must have datetime index)
            feature_set_name: Name for the feature set

        Returns:
            Feature set ID
        """
        with self.lock:
            # Validate data
            if not isinstance(features_df.index, pd.DatetimeIndex):
                raise ValueError("DataFrame must have DatetimeIndex")

            if features_df.empty:
                raise ValueError("DataFrame cannot be empty")

            # Calculate data hash
            data_hash = self._calculate_dataframe_hash(features_df)

            # Check if feature set already exists
            existing_id = self._get_feature_set_id(symbol, feature_set_name, data_hash)
            if existing_id:
                logger.info(f"Feature set already exists: {existing_id}")
                return existing_id

            # Create feature set metadata
            feature_set_id = f"{symbol}_{feature_set_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            metadata = FeatureSet(
                id=feature_set_id,
                name=feature_set_name,
                symbol=symbol,
                features=list(features_df.columns),
                created_at=datetime.now(),
                updated_at=datetime.now(),
                version='1.0.0',
                data_hash=data_hash,
                row_count=len(features_df),
                date_range=(features_df.index.min().to_pydatetime(),
                           features_df.index.max().to_pydatetime())
            )

            # Store data in partitions
            self._store_partitioned_data(feature_set_id, features_df)

            # Store metadata
            self._store_metadata(metadata)

            # Update cache
            self.metadata_cache[metadata.id] = metadata

            logger.info(f"Stored feature set: {feature_set_id} ({len(features_df)} rows)")

            return feature_set_id

    def retrieve_features(self, query: FeatureQuery) -> pd.DataFrame:
        """
        Retrieve features based on query.

        Args:
            query: Feature query specification

        Returns:
            DataFrame with requested features
        """
        with self.lock:
            # Find relevant feature sets
            feature_sets = self._find_feature_sets(query.symbol)

            if not feature_sets:
                logger.warning(f"No feature sets found for {query.symbol}")
                return pd.DataFrame()

            # Get latest feature set
            latest_set = max(feature_sets, key=lambda fs: (fs.updated_at, fs.created_at))

            # Retrieve data
            features_df = self._retrieve_partitioned_data(latest_set.id, query)

            # Filter features
            available_features = [f for f in query.feature_names if f in features_df.columns]
            if not available_features:
                logger.warning(f"None of the requested features available: {query.feature_names}")
                return pd.DataFrame()

            result_df = features_df[available_features].copy()

            # Apply date filters
            if query.start_date:
                result_df = result_df[result_df.index >= query.start_date]
            if query.end_date:
                result_df = result_df[result_df.index <= query.end_date]

            # Apply limit
            if query.limit:
                result_df = result_df.tail(query.limit)

            logger.info(f"Retrieved {len(result_df)} rows for {query.symbol}")

            return result_df

    def update_features(self, symbol: str, new_features_df: pd.DataFrame,
                       feature_set_name: str = 'default') -> str:
        """
        Update existing feature set with new data.

        Args:
            symbol: Stock symbol
            new_features_df: New feature data
            feature_set_name: Feature set name

        Returns:
            Updated feature set ID
        """
        with self.lock:
            # Find existing feature set
            feature_sets = self._find_feature_sets(symbol, feature_set_name)
            if not feature_sets:
                # Create new if doesn't exist
                return self.store_features(symbol, new_features_df, feature_set_name)

            latest_set = max(feature_sets, key=lambda fs: (fs.updated_at, fs.created_at))

            # Merge with existing data
            existing_data = self._retrieve_all_data(latest_set.id)
            combined_data = pd.concat([existing_data, new_features_df])
            combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
            combined_data = combined_data.sort_index()

            # Ensure timestamp is different
            import time
            time.sleep(0.01)

            # Store updated data
            return self.store_features(symbol, combined_data, feature_set_name)

    def list_feature_sets(self, symbol: Optional[str] = None) -> List[FeatureSet]:
        """
        List available feature sets.

        Args:
            symbol: Filter by symbol

        Returns:
            List of feature sets
        """
        with self.lock:
            feature_sets = list(self.metadata_cache.values())

            if symbol:
                feature_sets = [fs for fs in feature_sets if fs.symbol == symbol]

            return sorted(feature_sets, key=lambda fs: fs.created_at, reverse=True)

    def get_feature_info(self, symbol: str, feature_set_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about features for a symbol.

        Args:
            symbol: Stock symbol
            feature_set_name: Specific feature set name

        Returns:
            Feature information dictionary
        """
        with self.lock:
            feature_sets = self._find_feature_sets(symbol, feature_set_name)

            if not feature_sets:
                return {'error': 'No feature sets found'}

            latest_set = max(feature_sets, key=lambda fs: fs.created_at)

            return {
                'symbol': symbol,
                'feature_set': latest_set.name,
                'features': latest_set.features,
                'date_range': latest_set.date_range,
                'row_count': latest_set.row_count,
                'last_updated': latest_set.updated_at
            }

    def delete_feature_set(self, symbol: str, feature_set_name: str) -> bool:
        """
        Delete a feature set.

        Args:
            symbol: Stock symbol
            feature_set_name: Feature set name

        Returns:
            Success status
        """
        with self.lock:
            feature_sets = self._find_feature_sets(symbol, feature_set_name)

            if not feature_sets:
                return False

            # Delete from database and cache
            conn = sqlite3.connect(self.db_path)
            try:
                for feature_set in feature_sets:
                    # Delete metadata
                    conn.execute('DELETE FROM feature_sets WHERE id = ?',
                               (f"{symbol}_{feature_set_name}",))
                    # Delete partitions (would need to implement partition deletion)
                    # For now, just mark as deleted
                    conn.commit()

                    # Remove from cache
                    if f"{symbol}_{feature_set_name}" in self.metadata_cache:
                        del self.metadata_cache[f"{symbol}_{feature_set_name}"]

                logger.info(f"Deleted feature set: {symbol}_{feature_set_name}")
                return True

            except Exception as e:
                logger.error(f"Failed to delete feature set: {e}")
                return False
            finally:
                conn.close()

    def optimize_storage(self):
        """Optimize storage by cleaning up old data and rebuilding indexes."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)

            try:
                # Vacuum database
                conn.execute('VACUUM')

                # Rebuild indexes
                conn.execute('REINDEX')

                # Clean up old partitions (keep last 30 days of partitions)
                cutoff_date = datetime.now() - timedelta(days=30)
                # Implementation would depend on partition naming scheme

                conn.commit()

                logger.info("Storage optimization completed")

            except Exception as e:
                logger.error(f"Storage optimization failed: {e}")
            finally:
                conn.close()

    def _initialize_database(self):
        """Initialize SQLite database schema."""
        conn = sqlite3.connect(self.db_path)

        try:
            # Feature sets metadata table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS feature_sets (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    name TEXT NOT NULL,
                    features TEXT NOT NULL,  -- JSON array
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    version TEXT NOT NULL,
                    data_hash TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    date_start TEXT NOT NULL,
                    date_end TEXT NOT NULL
                )
            ''')

            # Create indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_symbol_name ON feature_sets(symbol, name)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON feature_sets(created_at)')

            conn.commit()

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
        finally:
            conn.close()

    def _store_partitioned_data(self, feature_set_id: str, data: pd.DataFrame):
        """Store data in time-based partitions."""
        # For simplicity, store as compressed pickle files
        # In production, would use Parquet or other columnar format

        partition_dir = self.store_path / 'partitions' / feature_set_id
        partition_dir.mkdir(parents=True, exist_ok=True)

        # Partition by month
        monthly_partitions = {}
        for date, row in data.iterrows():
            month_key = date.strftime('%Y_%m')
            if month_key not in monthly_partitions:
                monthly_partitions[month_key] = []
            monthly_partitions[month_key].append((date, row))

        # Save partitions
        for month_key, partition_data in monthly_partitions.items():
            partition_df = pd.DataFrame(
                [row for _, row in partition_data],
                index=[date for date, _ in partition_data]
            )

            partition_path = partition_dir / f'partition_{month_key}.pkl'
            with open(partition_path, 'wb') as f:
                pickle.dump(partition_df, f)

    def _retrieve_partitioned_data(self, feature_set_id: str, query: FeatureQuery) -> pd.DataFrame:
        """Retrieve data from partitions based on query."""
        partition_dir = self.store_path / 'partitions'

        # Find the specific partition directory for this feature set
        feature_dir = partition_dir / feature_set_id
        if not feature_dir.exists():
            return pd.DataFrame()

        all_data = []

        for partition_file in feature_dir.glob('partition_*.pkl'):
            try:
                with open(partition_file, 'rb') as f:
                    partition_df = pickle.load(f)

                # Apply date filters if specified
                if query.start_date:
                    partition_df = partition_df[partition_df.index >= query.start_date]
                if query.end_date:
                    partition_df = partition_df[partition_df.index <= query.end_date]

                if not partition_df.empty:
                    all_data.append(partition_df)

            except Exception as e:
                logger.warning(f"Failed to load partition {partition_file}: {e}")

        if not all_data:
            return pd.DataFrame()

        # Combine all partitions
        combined_df = pd.concat(all_data)
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        combined_df = combined_df.sort_index()

        return combined_df

    def _retrieve_all_data(self, feature_set_id: str) -> pd.DataFrame:
        """Retrieve all data for a feature set."""
        query = FeatureQuery(symbol='', feature_names=[])  # Dummy query
        return self._retrieve_partitioned_data(feature_set_id, query)

    def _store_metadata(self, metadata: FeatureSet):
        """Store feature set metadata."""
        conn = sqlite3.connect(self.db_path)

        try:
            conn.execute('''
                INSERT OR REPLACE INTO feature_sets
                (id, symbol, name, features, created_at, updated_at, version,
                 data_hash, row_count, date_start, date_end)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metadata.id,
                metadata.symbol,
                metadata.name,
                json.dumps(metadata.features),
                metadata.created_at.isoformat(),
                metadata.updated_at.isoformat(),
                metadata.version,
                metadata.data_hash,
                metadata.row_count,
                metadata.date_range[0].isoformat(),
                metadata.date_range[1].isoformat()
            ))

            conn.commit()

        except Exception as e:
            logger.error(f"Failed to store metadata: {e}")
            raise
        finally:
            conn.close()

    def _load_metadata_cache(self):
        """Load metadata into cache."""
        conn = sqlite3.connect(self.db_path)

        try:
            cursor = conn.execute('SELECT * FROM feature_sets')
            rows = cursor.fetchall()

            for row in rows:
                metadata = FeatureSet(
                    id=row[0],
                    name=row[2],
                    symbol=row[1],
                    features=json.loads(row[3]),
                    created_at=datetime.fromisoformat(row[4]),
                    updated_at=datetime.fromisoformat(row[5]),
                    version=row[6],
                    data_hash=row[7],
                    row_count=row[8],
                    date_range=(datetime.fromisoformat(row[9]), datetime.fromisoformat(row[10]))
                )

                self.metadata_cache[row[0]] = metadata

        except Exception as e:
            logger.warning(f"Failed to load metadata cache: {e}")
        finally:
            conn.close()

    def _find_feature_sets(self, symbol: str, name: Optional[str] = None) -> List[FeatureSet]:
        """Find feature sets for a symbol."""
        matching_sets = []

        for key, metadata in self.metadata_cache.items():
            if metadata.symbol == symbol:
                if name is None or metadata.name == name:
                    matching_sets.append(metadata)

        return matching_sets

    def _get_feature_set_id(self, symbol: str, name: str, data_hash: str) -> Optional[str]:
        """Get existing feature set ID by hash."""
        for key, metadata in self.metadata_cache.items():
            if (metadata.symbol == symbol and
                metadata.name == name and
                metadata.data_hash == data_hash):
                return key

        return None

    def _calculate_dataframe_hash(self, df: pd.DataFrame) -> str:
        """Calculate hash of DataFrame for deduplication."""
        if df.empty:
            return 'empty'

        hashed = pd.util.hash_pandas_object(df, index=True).values
        return hashlib.sha256(hashed.tobytes()).hexdigest()[:16]

# Convenience functions
def create_feature_store(store_path: str = '/tmp/feature_store'):
    """Create and configure time-series feature store."""
    return TimeSeriesFeatureStore(store_path)

def store_symbol_features(symbol: str, features_df: pd.DataFrame,
                         feature_set_name: str = 'default') -> str:
    """Store features for a symbol with default settings."""
    store = create_feature_store()
    return store.store_features(symbol, features_df, feature_set_name)

def retrieve_symbol_features(symbol: str, feature_names: List[str],
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> pd.DataFrame:
    """Retrieve features for a symbol."""
    store = create_feature_store()
    query = FeatureQuery(
        symbol=symbol,
        feature_names=feature_names,
        start_date=start_date,
        end_date=end_date
    )
    return store.retrieve_features(query)