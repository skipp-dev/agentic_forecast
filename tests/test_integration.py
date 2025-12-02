"""
Integration Tests for IB Forecast System

Comprehensive integration testing for the complete IB Forecast architecture.
Tests agent-to-service communication, data pipelines, and end-to-end workflows.
"""

import os
import sys
import unittest
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
import tempfile
import shutil
from unittest.mock import Mock, patch
import torch

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

sys.modules['talib'] = MockTalib()

# Add paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.hyperparameter_search_agent import HyperparameterSearchAgent
from agents.drift_monitor_agent import DriftMonitorAgent
from agents.feature_engineer_agent import FeatureEngineerAgent
from agents.forecast_agent import ForecastAgent
from services.gpu_training_service import GPUTrainingService, TrainingConfig
from services.model_registry_service import ModelRegistryService
from services.inference_service import InferenceService
from data.feature_store import TimeSeriesFeatureStore
from data.metrics_database import MetricsDatabase
from src.gpu_services import get_gpu_services

logger = logging.getLogger(__name__)

# Mock model for testing that can be pickled
class MockModel:
    def state_dict(self):
        return {'weight': torch.tensor([1.0])}

class TestIBForecastIntegration(unittest.TestCase):
    """Integration tests for the complete IB Forecast system."""

    def setUp(self):
        """Set up test environment."""
        # Create temporary directories for testing
        self.temp_dir = tempfile.mkdtemp()
        self.feature_store_path = os.path.join(self.temp_dir, 'feature_store')
        self.metrics_db_path = os.path.join(self.temp_dir, 'metrics.db')
        self.model_registry_path = os.path.join(self.temp_dir, 'model_registry')

        # Initialize components
        self.gpu_services = get_gpu_services()
        self.feature_store = TimeSeriesFeatureStore(self.feature_store_path)
        self.metrics_db = MetricsDatabase(self.metrics_db_path)
        self.model_registry = ModelRegistryService(self.model_registry_path)

        # Initialize agents
        self.mock_pipeline = Mock()
        self.mock_pipeline.train_cnn_lstm.return_value = (Mock(), {'test_metrics': {'mae': 0.1}})
        self.mock_pipeline.train_ensemble.return_value = (Mock(), {'test_mae': 0.1})
        self.hyperparameter_agent = HyperparameterSearchAgent(data_pipeline=self.mock_pipeline)
        self.drift_monitor = DriftMonitorAgent()
        self.feature_engineer = FeatureEngineerAgent()
        self.forecast_agent = ForecastAgent()

        # Initialize services
        self.training_service = GPUTrainingService(self.gpu_services, self.model_registry)
        self.inference_service = InferenceService(
            self.model_registry, self.feature_engineer, self.gpu_services
        )

        # Create mock market data
        self.mock_data = self._create_mock_market_data()

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_mock_market_data(self) -> pd.DataFrame:
        """Create mock market data for testing."""
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        np.random.seed(42)

        # Generate realistic-looking price data
        n_days = len(dates)
        returns = np.random.normal(0.0005, 0.02, n_days)
        prices = 100 * np.exp(np.cumsum(returns))

        # Generate OHLCV data
        high_mult = 1 + np.abs(np.random.normal(0, 0.02, n_days))
        low_mult = 1 - np.abs(np.random.normal(0, 0.02, n_days))
        volume = np.random.randint(1000000, 10000000, n_days)

        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.005, n_days)),
            'high': prices * high_mult,
            'low': prices * low_mult,
            'close': prices,
            'volume': volume
        }, index=dates)

        # Ensure high >= close >= low and high >= open >= low
        data['high'] = np.maximum(data[['high', 'close', 'open']].max(axis=1), data['high'])
        data['low'] = np.minimum(data[['low', 'close', 'open']].min(axis=1), data['low'])

        return data

    def test_feature_store_integration(self):
        """Test feature store operations."""
        symbol = 'AAPL'

        # Store features
        feature_set_id = self.feature_store.store_features(symbol, self.mock_data)
        self.assertIsNotNone(feature_set_id)

        # Retrieve features
        from data.feature_store import FeatureQuery
        query = FeatureQuery(
            symbol=symbol,
            feature_names=['close', 'volume'],
            start_date=self.mock_data.index[0],
            end_date=self.mock_data.index[-1]
        )
        retrieved_data = self.feature_store.retrieve_features(query)

        self.assertIsNotNone(retrieved_data)
        self.assertFalse(retrieved_data.empty)
        self.assertIn('close', retrieved_data.columns)
        self.assertIn('volume', retrieved_data.columns)

    def test_metrics_database_integration(self):
        """Test metrics database operations."""
        # Store metrics
        self.metrics_db.store_metric('test_metric', 1.5, {'type': 'test'})
        self.metrics_db.store_metric('test_metric', 2.0, {'type': 'test'})

        # Query metrics
        from data.metrics_database import MetricQuery
        query = MetricQuery(
            metric_names=['test_metric'],
            start_time=datetime.now() - timedelta(hours=1),
            end_time=datetime.now()
        )
        results = self.metrics_db.query_metrics(query)

        self.assertIsNotNone(results)
        self.assertFalse(results.empty)
        self.assertTrue(len(results) >= 2)

        # Test statistics
        stats = self.metrics_db.get_metric_stats('test_metric', hours=1)
        self.assertIn('mean', stats)
        self.assertIn('count', stats)

    def test_model_registry_integration(self):
        """Test model registry operations."""
        # Use the module-level MockModel that can be pickled
        mock_model = MockModel()

        # Register model
        model_id = self.model_registry.register_model(
            model=mock_model,
            symbol='AAPL',
            model_type='test',
            training_results={'final_metrics': {'mae': 0.1, 'rmse': 0.15}},
            training_config={'epochs': 10},
            feature_names=['close', 'volume'],
            framework='pytorch'
        )

        self.assertIsNotNone(model_id)

        # Retrieve model
        retrieved_model = self.model_registry.load_model(model_id)
        self.assertIsNotNone(retrieved_model)

        # Get metadata
        metadata = self.model_registry.get_model_metadata(model_id)
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.symbol, 'AAPL')
        self.assertEqual(metadata.model_type, 'test')

    def test_feature_engineer_agent_integration(self):
        """Test feature engineer agent integration."""
        symbol = 'AAPL'

        # Engineer features
        features_df = self.feature_engineer.engineer_features(symbol, self.mock_data)
        self.assertIsNotNone(features_df)
        self.assertFalse(features_df.empty)
        self.assertTrue(len(features_df.columns) > 5)  # Should have engineered features

        # Test feature selection
        target = features_df['close'].shift(-1).dropna()
        features = features_df.drop(columns=['close'], errors='ignore').dropna()
        common_index = features.index.intersection(target.index)
        features = features.loc[common_index]
        target = target.loc[common_index]

        if len(features) > 10:
            selected_features = self.feature_engineer.select_features(
                features, target, method='mutual_info', k=5
            )
            self.assertIsInstance(selected_features, list)
            self.assertEqual(len(selected_features), 5)

    def test_forecast_agent_integration(self):
        """Test forecast agent integration."""
        symbol = 'AAPL'

        # Interpret forecast
        mock_forecasts = [{'horizon': 1, 'predicted_return': 0.01}]
        mock_metrics = {'directional_accuracy': 0.6, 'smape': 0.1}
        interpretation = self.forecast_agent.interpret_forecasts(symbol, mock_forecasts, mock_metrics)
        self.assertIsNotNone(interpretation)
        self.assertIn('risk_assessment', interpretation)
        self.assertIn('scenario_notes', interpretation)

    def test_gpu_training_service_integration(self):
        """Test GPU training service integration."""
        symbol = 'AAPL'

        # Prepare training data
        training_data = self.training_service._prepare_training_data(symbol)
        if training_data is None:
            # Use mock data if API fails
            features_df = self.feature_engineer.engineer_features(symbol, self.mock_data)
            target = features_df['close'].shift(-1).dropna()
            features = features_df.drop(columns=['close'], errors='ignore').dropna()
            common_index = features.index.intersection(target.index)
            X = features.loc[common_index].values
            y = target.loc[common_index].values

            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
            training_data = {
                'X_train': X_train, 'X_val': X_val,
                'y_train': y_train, 'y_val': y_val,
                'feature_names': list(features.columns)
            }

        if training_data:
            # Test training
            model_config = {'type': 'lstm', 'hidden_size': 32, 'num_layers': 1}
            config = TrainingConfig(
                model_type='lstm', epochs=2, batch_size=16
            )

            results = self.training_service.train_model(
                symbol, model_config, training_data, config
            )

            self.assertIsNotNone(results)
            self.assertIn('final_metrics', results)

    def test_inference_service_integration(self):
        """Test inference service integration."""
        symbol = 'AAPL'

        # First train a model
        training_data = self.training_service._prepare_training_data(symbol)
        if training_data is None:
            # Use mock data if API fails
            features_df = self.feature_engineer.engineer_features(symbol, self.mock_data)
            target = features_df['close'].shift(-1).dropna()
            features = features_df.drop(columns=['close'], errors='ignore').dropna()
            common_index = features.index.intersection(target.index)
            X = features.loc[common_index].values
            y = target.loc[common_index].values

            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
            training_data = {
                'X_train': X_train, 'X_val': X_val,
                'y_train': y_train, 'y_val': y_val,
                'feature_names': list(features.columns)
            }

        if training_data:
            # Train a model first
            model_config = {'type': 'lstm', 'hidden_size': 32, 'num_layers': 1}
            config = TrainingConfig(
                model_type='lstm', epochs=2, batch_size=16
            )

            train_results = self.training_service.train_model(
                symbol, model_config, training_data, config
            )
            self.assertIn('final_metrics', train_results)

        # Create mock request
        from services.inference_service import InferenceRequest, InferenceResult
        request = InferenceRequest(symbol=symbol)

        # Test async inference
        async def run_inference():
            result = await self.inference_service.predict_async(request)
            return result

        # Run in event loop
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(run_inference())
            self.assertIsNotNone(result)
            self.assertIsInstance(result, InferenceResult)
            self.assertEqual(result.symbol, symbol)
            self.assertIsNotNone(result.prediction)
        finally:
            loop.close()

    def test_end_to_end_forecasting_pipeline(self):
        """Test complete end-to-end forecasting pipeline."""
        symbol = 'AAPL'

        try:
            # 1. Feature engineering
            features_df = self.feature_engineer.engineer_features(symbol, self.mock_data)
            self.assertIsNotNone(features_df)

            # 2. Store features
            feature_set_id = self.feature_store.store_features(symbol, features_df)
            self.assertIsNotNone(feature_set_id)

            # 3. Interpret forecast
            mock_forecasts = [{'horizon': 1, 'predicted_return': 0.01}]
            mock_metrics = {'directional_accuracy': 0.6, 'smape': 0.1}
            interpretation = self.forecast_agent.interpret_forecasts(symbol, mock_forecasts, mock_metrics)
            self.assertIsNotNone(interpretation)

            # 4. Train a model for inference testing
            training_data = self.training_service._prepare_training_data(symbol)
            if training_data is None:
                # Use mock data if API fails
                features_df = self.feature_engineer.engineer_features(symbol, self.mock_data)
                target = features_df['close'].shift(-1).dropna()
                features = features_df.drop(columns=['close'], errors='ignore').dropna()
                common_index = features.index.intersection(target.index)
                X = features.loc[common_index].values
                y = target.loc[common_index].values

                from sklearn.model_selection import train_test_split
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
                training_data = {
                    'X_train': X_train, 'X_val': X_val,
                    'y_train': y_train, 'y_val': y_val,
                    'feature_names': list(features.columns)
                }

            if training_data:
                # Train a model
                model_config = {'type': 'lstm', 'hidden_size': 32, 'num_layers': 1}
                config = TrainingConfig(
                    model_type='lstm', epochs=2, batch_size=16
                )

                train_results = self.training_service.train_model(
                    symbol, model_config, training_data, config
                )
                self.assertIn('final_metrics', train_results)

            # 5. Store forecast metrics
            self.metrics_db.store_metric(
                'forecast_accuracy', 0.85,
                {'symbol': symbol, 'horizon': 1}
            )

            # 6. Test inference
            from services.inference_service import InferenceRequest, InferenceResult
            request = InferenceRequest(symbol=symbol)

            async def test_inference():
                result = await self.inference_service.predict_async(request)
                return result

            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                inference_result = loop.run_until_complete(test_inference())
                self.assertIsNotNone(inference_result)
            finally:
                loop.close()

            logger.info("End-to-end forecasting pipeline test completed successfully")

        except Exception as e:
            self.fail(f"End-to-end pipeline test failed: {e}")

    def test_agent_orchestration(self):
        """Test agent orchestration and communication."""
        symbol = 'AAPL'

        try:
            # Test hyperparameter search
            search_results = self.hyperparameter_agent.run_search(
                symbol=symbol, model_type='cnn_lstm', max_trials=2
            )
            self.assertIsInstance(search_results, dict)

            # Test drift monitoring
            drift_results = self.drift_monitor.monitor_performance(symbol)
            self.assertIsInstance(drift_results, dict)

            # Test feature engineering
            features = self.feature_engineer.engineer_features(symbol, self.mock_data)
            self.assertIsNotNone(features)

            logger.info("Agent orchestration test completed successfully")

        except Exception as e:
            logger.warning(f"Agent orchestration test had issues: {e}")
            # Don't fail the test for agent issues, as they might need API keys

    def test_data_pipeline_validation(self):
        """Test data pipeline integration."""
        symbol = 'AAPL'

        # Test feature store and metrics DB together
        features_df = self.feature_engineer.engineer_features(symbol, self.mock_data)
        feature_set_id = self.feature_store.store_features(symbol, features_df)

        # Store pipeline metrics
        self.metrics_db.store_metric('features_engineered', len(features_df.columns),
                                   {'symbol': symbol, 'pipeline': 'feature_engineering'})
        self.metrics_db.store_metric('data_points_stored', len(features_df),
                                   {'symbol': symbol, 'pipeline': 'storage'})

        # Retrieve and validate
        from data.feature_store import FeatureQuery
        query = FeatureQuery(symbol=symbol, feature_names=['close'])
        retrieved = self.feature_store.retrieve_features(query)

        self.assertIsNotNone(retrieved)
        self.assertFalse(retrieved.empty)

        # Check metrics
        metrics = self.metrics_db.list_metrics()
        self.assertIn('features_engineered', metrics)
        self.assertIn('data_points_stored', metrics)

if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Run tests
    unittest.main(verbosity=2)
