"""
Data Pipeline Validation Tests

Validates the complete data pipeline including feature store and metrics database.
Tests data flow, integrity, and performance characteristics.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import logging
from datetime import datetime, timedelta
import tempfile
import shutil
import time
from unittest.mock import MagicMock

# Mock talib before importing agents
class MockTalib:
    def __getattr__(self, name):
        def wrapper(*args, **kwargs):
            # Return a numpy array of the same length as the first argument (usually close price)
            if len(args) > 0:
                import numpy as np
                import pandas as pd
                arg = args[0]
                if isinstance(arg, (pd.Series, np.ndarray)):
                    return np.random.random(len(arg))
                elif isinstance(arg, list):
                     return np.random.random(len(arg))
            return np.array([])
        return wrapper
    
    # Handle functions that return multiple values
    def STOCH(self, *args, **kwargs):
        import numpy as np
        l = len(args[0])
        return np.random.random(l), np.random.random(l)
        
    def MACD(self, *args, **kwargs):
        import numpy as np
        l = len(args[0])
        return np.random.random(l), np.random.random(l), np.random.random(l)
        
    def BBANDS(self, *args, **kwargs):
        import numpy as np
        l = len(args[0])
        return np.random.random(l), np.random.random(l), np.random.random(l)
        
    def PLUS_DI(self, *args, **kwargs):
        import numpy as np
        l = len(args[0])
        return np.random.random(l)

    def MINUS_DI(self, *args, **kwargs):
        import numpy as np
        l = len(args[0])
        return np.random.random(l)

sys.modules["talib"] = MockTalib()

# Add paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data.feature_store import TimeSeriesFeatureStore, FeatureQuery
from data.metrics_database import MetricsDatabase, MetricQuery
from src.agents.feature_engineer_agent import FeatureEngineerAgent

logger = logging.getLogger(__name__)

class TestDataPipelineValidation(unittest.TestCase):
    """Comprehensive data pipeline validation tests."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.feature_store_path = os.path.join(self.temp_dir, "feature_store")
        self.metrics_db_path = os.path.join(self.temp_dir, "metrics.db")

        self.feature_store = TimeSeriesFeatureStore(self.feature_store_path)
        self.metrics_db = MetricsDatabase(self.metrics_db_path)
        self.feature_engineer = FeatureEngineerAgent()

        # Create test data
        self.test_data = self._create_test_data()

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_data(self) -> Dict[str, pd.DataFrame]:
        """Create comprehensive test data."""
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
        np.random.seed(42)

        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]

        data = {}
        for symbol in symbols:
            # Generate realistic price data
            n_days = len(dates)
            returns = np.random.normal(0.0005, 0.025, n_days)
            prices = 100 * np.exp(np.cumsum(returns))

            # Create OHLCV data
            high_mult = 1 + np.abs(np.random.normal(0, 0.02, n_days))
            low_mult = 1 - np.abs(np.random.normal(0, 0.02, n_days))
            volume = np.random.randint(1000000, 50000000, n_days)

            df = pd.DataFrame({
                "open": prices * (1 + np.random.normal(0, 0.005, n_days)),
                "high": prices * high_mult,
                "low": prices * low_mult,
                "close": prices,
                "volume": volume
            }, index=dates)

            # Ensure OHLC integrity
            df["high"] = np.maximum(df[["close", "open"]].max(axis=1), df["high"])
            df["low"] = np.minimum(df[["close", "open"]].min(axis=1), df["low"])

            data[symbol] = df

        return data

    def test_feature_store_data_integrity(self):
        """Test data integrity in feature store operations."""
        symbol = "AAPL"
        data = self.test_data[symbol]

        # Store original data
        original_checksum = self._calculate_dataframe_checksum(data)

        # Store in feature store
        feature_set_id = self.feature_store.store_features(symbol, data)
        self.assertIsNotNone(feature_set_id)

        # Retrieve data
        query = FeatureQuery(
            symbol=symbol,
            feature_names=list(data.columns),
            start_date=data.index[0],
            end_date=data.index[-1]
        )
        retrieved_data = self.feature_store.retrieve_features(query)

        # Verify data integrity
        self.assertIsNotNone(retrieved_data)
        self.assertFalse(retrieved_data.empty)
        self.assertEqual(len(retrieved_data), len(data))

        # Check data values (allowing for small numerical differences)
        for col in data.columns:
            if col in retrieved_data.columns:
                np.testing.assert_allclose(
                    retrieved_data[col].values,
                    data[col].values,
                    rtol=1e-10,
                    atol=1e-10
                )

    def test_feature_store_partitioning(self):
        """Test time-based partitioning functionality."""
        symbol = "AAPL"
        data = self.test_data[symbol]

        # Store data
        feature_set_id = self.feature_store.store_features(symbol, data)

        # Test retrieval with different date ranges
        mid_date = data.index[len(data) // 2]

        # First half
        query1 = FeatureQuery(
            symbol=symbol,
            feature_names=["close"],
            start_date=data.index[0],
            end_date=mid_date
        )
        data1 = self.feature_store.retrieve_features(query1)

        # Second half
        query2 = FeatureQuery(
            symbol=symbol,
            feature_names=["close"],
            start_date=mid_date + timedelta(days=1),
            end_date=data.index[-1]
        )
        data2 = self.feature_store.retrieve_features(query2)

        # Verify partitioning
        self.assertEqual(len(data1) + len(data2), len(data))
        self.assertTrue(data1.index.max() < data2.index.min())

    def test_feature_store_updates(self):
        """Test feature store update functionality."""
        symbol = "AAPL"
        initial_data = self.test_data[symbol].iloc[:100]  # First 100 days

        # Store initial data
        initial_id = self.feature_store.store_features(symbol, initial_data)

        # Add more data
        additional_data = self.test_data[symbol].iloc[100:200]  # Next 100 days
        update_id = self.feature_store.update_features(symbol, additional_data)

        # Verify update
        query = FeatureQuery(
            symbol=symbol,
            feature_names=["close"],
            start_date=initial_data.index[0],
            end_date=additional_data.index[-1]
        )
        combined_data = self.feature_store.retrieve_features(query)

        self.assertEqual(len(combined_data), 200)
        self.assertTrue(combined_data.index.min() == initial_data.index.min())
        self.assertTrue(combined_data.index.max() == additional_data.index.max())

    def test_metrics_database_time_series(self):
        """Test metrics database time-series functionality."""
        metric_name = "test_performance"
        symbol = "AAPL"

        # Store time-series metrics
        base_time = datetime.now()
        for i in range(100):
            timestamp = base_time + timedelta(minutes=i)
            value = 0.5 + 0.1 * np.sin(i * 0.1) + np.random.normal(0, 0.05)
            self.metrics_db.store_metric(
                metric_name, value,
                {"symbol": symbol, "type": "performance"},
                timestamp=timestamp
            )

        # Query metrics
        query = MetricQuery(
            metric_names=[metric_name],
            start_time=base_time,
            end_time=base_time + timedelta(minutes=99)
        )
        results = self.metrics_db.query_metrics(query)

        # Verify time-series integrity
        self.assertIsNotNone(results)
        self.assertFalse(results.empty)
        self.assertTrue(len(results) >= 90)  # Allow some tolerance

        # Check temporal ordering
        self.assertTrue(results["timestamp"].is_monotonic_increasing)

    def test_metrics_database_aggregation(self):
        """Test metrics database aggregation functionality."""
        metric_name = "test_aggregated"
        symbol = "AAPL"

        # Store hourly metrics for a day
        base_time = datetime(2023, 1, 1, 0, 0, 0)
        for hour in range(24):
            for minute in range(0, 60, 5):  # Every 5 minutes
                timestamp = base_time + timedelta(hours=hour, minutes=minute)
                value = 10 + hour + np.random.normal(0, 1)
                self.metrics_db.store_metric(
                    metric_name, value,
                    {"symbol": symbol, "type": "aggregated_test"},
                    timestamp=timestamp
                )

        # Test different aggregations
        query = MetricQuery(
            metric_names=[metric_name],
            start_time=base_time,
            end_time=base_time + timedelta(days=1),
            aggregation="mean",
            interval="1h"
        )
        hourly_means = self.metrics_db.query_metrics(query)

        self.assertIsNotNone(hourly_means)
        self.assertFalse(hourly_means.empty)
        self.assertTrue(len(hourly_means) <= 24)  # Should have ~24 hourly aggregations

    def test_data_pipeline_end_to_end(self):
        """Test complete data pipeline from feature engineering to storage."""
        symbol = "AAPL"
        raw_data = self.test_data[symbol]

        # Step 1: Feature engineering
        start_time = time.time()
        engineered_features = self.feature_engineer.engineer_features(symbol, raw_data)
        feature_engineering_time = time.time() - start_time

        self.assertIsNotNone(engineered_features)
        self.assertFalse(engineered_features.empty)
        self.assertTrue(len(engineered_features.columns) > len(raw_data.columns))

        # Step 2: Store features
        start_time = time.time()
        feature_set_id = self.feature_store.store_features(symbol, engineered_features)
        storage_time = time.time() - start_time

        self.assertIsNotNone(feature_set_id)

        # Step 3: Store pipeline metrics
        self.metrics_db.store_metric(
            "pipeline.feature_engineering_time", feature_engineering_time,
            {"symbol": symbol, "pipeline": "feature_engineering"}
        )
        self.metrics_db.store_metric(
            "pipeline.storage_time", storage_time,
            {"symbol": symbol, "pipeline": "storage"}
        )
        self.metrics_db.store_metric(
            "pipeline.features_count", len(engineered_features.columns),
            {"symbol": symbol, "pipeline": "feature_count"}
        )

        # Step 4: Retrieve and validate
        query = FeatureQuery(
            symbol=symbol,
            feature_names=["close", "volume", "rsi_14"],
            start_date=engineered_features.index[0],
            end_date=engineered_features.index[-1]
        )
        retrieved_features = self.feature_store.retrieve_features(query)

        self.assertIsNotNone(retrieved_features)
        self.assertFalse(retrieved_features.empty)

        # Step 5: Validate pipeline metrics
        metrics_query = MetricQuery(
            metric_names=["pipeline.feature_engineering_time", "pipeline.storage_time"],
            start_time=datetime.now() - timedelta(hours=1),
            end_time=datetime.now()
        )
        pipeline_metrics = self.metrics_db.query_metrics(metrics_query)

        self.assertIsNotNone(pipeline_metrics)
        self.assertFalse(pipeline_metrics.empty)

        logger.info(f"Data pipeline test completed: {len(engineered_features.columns)} features, "
                   ".3f")

    def test_multi_symbol_pipeline(self):
        """Test data pipeline with multiple symbols."""
        symbols = ["AAPL", "GOOGL", "MSFT"]

        # Process multiple symbols
        for symbol in symbols:
            data = self.test_data[symbol]

            # Feature engineering
            features = self.feature_engineer.engineer_features(symbol, data)

            # Storage
            feature_set_id = self.feature_store.store_features(symbol, features)

            # Metrics
            self.metrics_db.store_metric(
                "pipeline.symbol_processed", 1,
                {"symbol": symbol, "pipeline": "multi_symbol"}
            )

            self.assertIsNotNone(feature_set_id)

        # Verify all symbols processed
        metrics_query = MetricQuery(
            metric_names=["pipeline.symbol_processed"],
            start_time=datetime.now() - timedelta(hours=1),
            end_time=datetime.now()
        )
        results = self.metrics_db.query_metrics(metrics_query)

        self.assertEqual(len(results), len(symbols))

        # Verify feature sets
        stored_symbols = self.feature_store.list_symbols()
        self.assertTrue(set(symbols).issubset(set(stored_symbols)))

    def test_data_quality_validation(self):
        """Test data quality validation in the pipeline."""
        symbol = "AAPL"
        data = self.test_data[symbol].copy()

        # Process through pipeline
        features = self.feature_engineer.engineer_features(symbol, data)
        feature_set_id = self.feature_store.store_features(symbol, features)

        # Verify data quality handling
        self.assertIsNotNone(features)
        self.assertFalse(features.isnull().all().all())  # Should handle NaN values

        # Check that OHLC relationships are maintained
        retrieved = self.feature_store.retrieve_features(
            FeatureQuery(symbol=symbol, feature_names=["open", "high", "low", "close"])
        )

        # Verify OHLC relationships are valid
        valid_rows = retrieved.dropna()
        if len(valid_rows) > 0:
            # Basic OHLC relationships
            self.assertTrue(all(valid_rows["high"] >= valid_rows["low"]))
            self.assertTrue(all(valid_rows["low"] <= valid_rows["close"]))
            self.assertTrue(all(valid_rows["high"] >= valid_rows["close"]))
            self.assertTrue(all(valid_rows["low"] <= valid_rows["open"]))
            self.assertTrue(all(valid_rows["high"] >= valid_rows["open"]))

    def test_performance_characteristics(self):
        """Test performance characteristics of the data pipeline."""
        symbol = "AAPL"
        data = self.test_data[symbol]

        # Test feature engineering performance
        start_time = time.time()
        features = self.feature_engineer.engineer_features(symbol, data)
        feature_time = time.time() - start_time

        # Test storage performance
        start_time = time.time()
        feature_set_id = self.feature_store.store_features(symbol, features)
        storage_time = time.time() - start_time

        # Test retrieval performance
        start_time = time.time()
        query = FeatureQuery(
            symbol=symbol,
            feature_names=list(features.columns)[:10],  # First 10 features
            start_date=features.index[0],
            end_date=features.index[-1]
        )
        retrieved = self.feature_store.retrieve_features(query)
        retrieval_time = time.time() - start_time

        # Store performance metrics
        self.metrics_db.store_metric("performance.feature_engineering", feature_time,
                                   {"test": "performance", "operation": "feature_engineering"})
        self.metrics_db.store_metric("performance.storage", storage_time,
                                   {"test": "performance", "operation": "storage"})
        self.metrics_db.store_metric("performance.retrieval", retrieval_time,
                                   {"test": "performance", "operation": "retrieval"})

        # Performance assertions (reasonable bounds)
        self.assertLess(feature_time, 30.0)  # Should complete within 30 seconds
        self.assertLess(storage_time, 10.0)  # Should store within 10 seconds
        self.assertLess(retrieval_time, 5.0)  # Should retrieve within 5 seconds

        logger.info(f"Performance test: FE={feature_time:.2f}s, ST={storage_time:.2f}s, RT={retrieval_time:.2f}s")

    def _calculate_dataframe_checksum(self, df: pd.DataFrame) -> str:
        """Calculate checksum for dataframe validation."""
        import hashlib

        # Convert to string representation
        data_str = str(df.values.tobytes()) + str(df.index.tolist()) + str(df.columns.tolist())
        return hashlib.sha256(data_str.encode()).hexdigest()

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Run tests
    unittest.main(verbosity=2)

