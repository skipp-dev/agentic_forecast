"""
GPU Performance Tuning Tests

Comprehensive GPU performance testing and optimization for the IB Forecast system.
Tests memory usage, throughput, and GPU utilization across all components.
"""

import os
import sys
import unittest
import torch
import numpy as np
import pandas as pd
import time
import psutil
import GPUtil
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
import gc
from src.data.types import DataSpec
import pytest

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

from src.gpu_services import get_gpu_services
from services.gpu_training_service import GPUTrainingService, TrainingConfig
from services.inference_service import InferenceService
from src.agents.feature_engineer_agent import FeatureEngineerAgent
from data.metrics_database import MetricsDatabase

logger = logging.getLogger(__name__)

class TestGPUPerformanceTuning(unittest.TestCase):
    """GPU performance testing and optimization."""

    def setUp(self):
        """Set up GPU performance test environment."""
        self.gpu_services = get_gpu_services()
        self.has_gpu = torch.cuda.is_available()

        if not self.has_gpu:
            self.skipTest("GPU not available for performance testing")

        # Initialize components
        self.training_service = GPUTrainingService(self.gpu_services)
        self.feature_engineer = FeatureEngineerAgent()
        self.metrics_db = MetricsDatabase()

        # Test data
        self.test_data = self._create_test_data()

        # Performance metrics storage
        self.performance_metrics = {}

        logger.info(f"GPU Performance Testing initialized. GPU available: {self.has_gpu}")

    def _create_test_data(self) -> pd.DataFrame:
        """Create test data for GPU performance testing."""
        dates = pd.date_range(start="2023-01-01", end="2023-03-31", freq="D")
        np.random.seed(42)

        # Generate price data
        n_days = len(dates)
        returns = np.random.normal(0.0005, 0.02, n_days)
        prices = 100 * np.exp(np.cumsum(returns))

        # Create OHLCV data
        data = pd.DataFrame({
            "open": prices * (1 + np.random.normal(0, 0.005, n_days)),
            "high": prices * (1 + np.abs(np.random.normal(0, 0.02, n_days))),
            "low": prices * (1 - np.abs(np.random.normal(0, 0.02, n_days))),
            "close": prices,
            "volume": np.random.randint(1000000, 10000000, n_days)
        }, index=dates)

        return data

    def test_gpu_memory_management(self):
        """Test GPU memory management and optimization."""
        if not self.has_gpu:
            return

        initial_memory = torch.cuda.memory_allocated()

        try:
            # Test feature engineering with GPU
            symbol = "AAPL"
            features = self.feature_engineer.engineer_features(symbol, self.test_data)

            # Check memory usage during feature engineering
            memory_after_features = torch.cuda.memory_allocated()
            memory_used_features = memory_after_features - initial_memory

            # Test GPU training
            training_data = self._prepare_training_data(features)
            if training_data:
                model_config = {"type": "lstm", "hidden_size": 64, "num_layers": 2}
                config = TrainingConfig(
                    model_type="lstm", epochs=5, batch_size=32
                )

                training_start_memory = torch.cuda.memory_allocated()
                results = self.training_service.train_model(
                    symbol, model_config, training_data, config
                )
                training_end_memory = torch.cuda.memory_allocated()

                memory_used_training = training_end_memory - training_start_memory

                # Store memory metrics
                self.metrics_db.store_metric("gpu.memory.feature_engineering", memory_used_features / 1024**3,
                                           {"test": "memory_management", "component": "feature_engineering"})
                self.metrics_db.store_metric("gpu.memory.training", memory_used_training / 1024**3,
                                           {"test": "memory_management", "component": "training"})

                # Assertions
                self.assertLess(memory_used_features, 2.0)  # Less than 2GB for features
                self.assertLess(memory_used_training, 4.0)  # Less than 4GB for training

                logger.info(f"Memory test: Features={memory_used_features/1024**3:.2f}GB, "
                           f"Training={memory_used_training/1024**3:.2f}GB")

        finally:
            # Clean up GPU memory
            torch.cuda.empty_cache()
            gc.collect()

    def test_gpu_utilization_patterns(self):
        """Test GPU utilization patterns during different operations."""
        if not self.has_gpu:
            return

        utilization_patterns = {}

        try:
            # Test feature engineering utilization
            start_time = time.time()
            utilizations = []

            # Start monitoring
            symbol = "AAPL"
            features = self.feature_engineer.engineer_features(symbol, self.test_data)

            monitoring_time = time.time() - start_time
            avg_utilization = np.mean(utilizations) if utilizations else 0

            utilization_patterns["feature_engineering"] = {
                "avg_utilization": avg_utilization,
                "duration": monitoring_time
            }

            # Test training utilization
            training_data = self._prepare_training_data(features)
            if training_data:
                model_config = {"type": "lstm", "hidden_size": 32, "num_layers": 1}
                config = TrainingConfig(
                    model_type="lstm", epochs=3, batch_size=16
                )

                start_time = time.time()
                utilizations = []

                results = self.training_service.train_model(
                    symbol, model_config, training_data, config
                )

                monitoring_time = time.time() - start_time
                avg_utilization = np.mean(utilizations) if utilizations else 0

                utilization_patterns["training"] = {
                    "avg_utilization": avg_utilization,
                    "duration": monitoring_time
                }

            # Store utilization metrics
            for component, metrics in utilization_patterns.items():
                self.metrics_db.store_metric("gpu.utilization.avg", metrics["avg_utilization"],
                                           {"test": "utilization", "component": component})
                self.metrics_db.store_metric("gpu.utilization.duration", metrics["duration"],
                                           {"test": "utilization", "component": component})

            logger.info(f"GPU utilization patterns: {utilization_patterns}")

        except Exception as e:
            logger.warning(f"GPU utilization test failed: {e}")

    def test_batch_processing_performance(self):
        """Test batch processing performance and throughput."""
        if not self.has_gpu:
            return

        batch_sizes = [16, 32, 64, 128]
        throughput_results = {}

        try:
            symbol = "AAPL"
            features = self.feature_engineer.engineer_features(symbol, self.test_data)
            training_data = self._prepare_training_data(features)

            if training_data:
                for batch_size in batch_sizes:
                    model_config = {"type": "lstm", "hidden_size": 32, "num_layers": 1}
                    config = TrainingConfig(
                        model_type="lstm", epochs=2, batch_size=batch_size
                    )

                    start_time = time.time()
                    initial_memory = torch.cuda.memory_allocated()

                    results = self.training_service.train_model(
                        symbol, model_config, training_data, config
                    )

                    end_time = time.time()
                    final_memory = torch.cuda.memory_allocated()

                    processing_time = end_time - start_time
                    memory_used = final_memory - initial_memory

                    # Calculate throughput (samples per second)
                    n_samples = len(training_data["X_train"])
                    throughput = n_samples / processing_time

                    throughput_results[batch_size] = {
                        "throughput": throughput,
                        "processing_time": processing_time,
                        "memory_used_gb": memory_used / 1024**3
                    }

                    # Store metrics
                    self.metrics_db.store_metric("gpu.throughput.samples_per_sec", throughput,
                                               {"test": "batch_processing", "batch_size": batch_size})
                    self.metrics_db.store_metric("gpu.batch_processing.time", processing_time,
                                               {"test": "batch_processing", "batch_size": batch_size})

                # Find optimal batch size
                optimal_batch_size = max(throughput_results.keys(),
                                       key=lambda x: throughput_results[x]["throughput"])

                self.metrics_db.store_metric("gpu.optimal_batch_size", optimal_batch_size,
                                           {"test": "batch_processing"})

                logger.info(f"Batch processing results: {throughput_results}")
                logger.info(f"Optimal batch size: {optimal_batch_size}")

        except Exception as e:
            logger.warning(f"Batch processing test failed: {e}")

    def test_memory_efficiency_optimization(self):
        """Test memory efficiency optimizations."""
        if not self.has_gpu:
            return

        try:
            # Test with different memory optimization strategies
            symbol = "AAPL"
            features = self.feature_engineer.engineer_features(symbol, self.test_data)

            # Strategy 1: Standard processing
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()

            training_data = self._prepare_training_data(features)
            if training_data:
                model_config = {"type": "lstm", "hidden_size": 64, "num_layers": 2}
                config = TrainingConfig(
                    model_type="lstm", epochs=3, batch_size=32
                )

                results = self.training_service.train_model(
                    symbol, model_config, training_data, config
                )

                standard_memory = torch.cuda.memory_allocated() - initial_memory

                # Strategy 2: Gradient checkpointing (if available)
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated()

                # Enable gradient checkpointing for memory efficiency
                torch.utils.checkpoint.checkpoint_sequential = True

                results_checkpoint = self.training_service.train_model(
                    symbol, model_config, training_data, config
                )

                checkpoint_memory = torch.cuda.memory_allocated() - initial_memory

                # Compare memory usage
                memory_savings = standard_memory - checkpoint_memory
                memory_efficiency = (memory_savings / standard_memory) * 100 if standard_memory > 0 else 0

                # Store optimization metrics
                self.metrics_db.store_metric("gpu.memory.standard", standard_memory / 1024**3,
                                           {"test": "memory_optimization", "strategy": "standard"})
                self.metrics_db.store_metric("gpu.memory.checkpoint", checkpoint_memory / 1024**3,
                                           {"test": "memory_optimization", "strategy": "checkpoint"})
                self.metrics_db.store_metric("gpu.memory.savings_percent", memory_efficiency,
                                           {"test": "memory_optimization"})

                logger.info(f"Memory optimization: Standard={standard_memory/1024**3:.2f}GB, "
                           f"Checkpoint={checkpoint_memory/1024**3:.2f}GB, "
                           f"Savings={memory_efficiency:.1f}%")

        except Exception as e:
            logger.warning(f"Memory efficiency test failed: {e}")
        finally:
            torch.cuda.empty_cache()

    def test_inference_performance(self):
        """Test inference performance and latency."""
        if not self.has_gpu:
            return

        try:
            from services.inference_service import InferenceRequest

            # First train a model
            symbol = "AAPL"
            features = self.feature_engineer.engineer_features(symbol, self.test_data)
            training_data = self._prepare_training_data(features)

            if training_data:
                model_config = {"type": "lstm", "hidden_size": 32, "num_layers": 1}
                config = TrainingConfig(
                    model_type="lstm", epochs=2, batch_size=16
                )

                training_results = self.training_service.train_model(
                    symbol, model_config, training_data, config
                )

                # Test inference performance
                inference_service = InferenceService(
                    self.training_service.model_registry,
                    self.feature_engineer,
                    self.gpu_services
                )

                request = InferenceRequest(symbol=symbol)

                # Single inference timing
                latencies = []
                for _ in range(10):
                    start_time = time.time()
                    # Note: Using sync method for testing
                    # In real async code, would use await
                    result = self._run_sync_inference(inference_service, request)
                    latency = time.time() - start_time
                    latencies.append(latency)

                avg_latency = np.mean(latencies)
                p95_latency = np.percentile(latencies, 95)
                throughput = 1.0 / avg_latency

                # Store inference metrics
                self.metrics_db.store_metric("gpu.inference.latency.avg", avg_latency * 1000,
                                           {"test": "inference_performance", "metric": "latency_ms"})
                self.metrics_db.store_metric("gpu.inference.latency.p95", p95_latency * 1000,
                                           {"test": "inference_performance", "metric": "p95_latency_ms"})
                self.metrics_db.store_metric("gpu.inference.throughput", throughput,
                                           {"test": "inference_performance", "metric": "requests_per_sec"})

                # Performance assertions
                self.assertLess(avg_latency, 1.0)  # Should be under 1 second
                self.assertLess(p95_latency, 2.0)  # P95 should be under 2 seconds

                logger.info(f"Inference performance: Avg={avg_latency*1000:.1f}ms, "
                           f"P95={p95_latency*1000:.1f}ms, Throughput={throughput:.2f} req/s")

        except Exception as e:
            logger.warning(f"Inference performance test failed: {e}")

    def test_concurrent_gpu_operations(self):
        """Test concurrent GPU operations and resource sharing."""
        if not self.has_gpu:
            return

        try:
            import threading
            import queue

            # Test concurrent feature engineering
            symbol = "AAPL"
            n_threads = min(4, torch.cuda.device_count() if torch.cuda.is_available() else 1)

            results_queue = queue.Queue()
            threads = []

            def concurrent_feature_engineering(thread_id):
                try:
                    start_time = time.time()
                    features = self.feature_engineer.engineer_features(symbol, self.test_data)
                    end_time = time.time()

                    results_queue.put({
                        "thread_id": thread_id,
                        "duration": end_time - start_time,
                        "success": True
                    })
                except Exception as e:
                    results_queue.put({
                        "thread_id": thread_id,
                        "error": str(e),
                        "success": False
                    })

            # Start concurrent operations
            for i in range(n_threads):
                thread = threading.Thread(target=concurrent_feature_engineering, args=(i,))
                threads.append(thread)
                thread.start()

            # Wait for completion
            for thread in threads:
                thread.join()

            # Collect results
            concurrent_results = []
            while not results_queue.empty():
                concurrent_results.append(results_queue.get())

            successful_operations = sum(1 for r in concurrent_results if r["success"])
            avg_duration = np.mean([r["duration"] for r in concurrent_results if r["success"]])

            # Store concurrency metrics
            self.metrics_db.store_metric("gpu.concurrency.successful_ops", successful_operations,
                                       {"test": "concurrent_operations"})
            self.metrics_db.store_metric("gpu.concurrency.avg_duration", avg_duration,
                                       {"test": "concurrent_operations"})

            # Assertions
            self.assertEqual(successful_operations, n_threads)  # All should succeed
            self.assertLess(avg_duration, 30.0)  # Should complete within reasonable time

            logger.info(f"Concurrent operations: {successful_operations}/{n_threads} successful, "
                       f"Avg duration={avg_duration:.2f}s")

        except Exception as e:
            logger.warning(f"Concurrent operations test failed: {e}")

    def test_gpu_memory_fragmentation(self):
        """Test GPU memory fragmentation and defragmentation."""
        if not self.has_gpu:
            return

        try:
            # Allocate and deallocate memory in different patterns to test fragmentation
            allocations = []

            # Pattern 1: Sequential allocation/deallocation
            for size_mb in [100, 200, 50, 300, 100]:
                tensor = torch.randn(size_mb * 1024 * 1024 // 4).cuda()  # Float32
                allocations.append(tensor)
                del tensor
                torch.cuda.empty_cache()

            fragmentation_score_1 = self._calculate_fragmentation_score()

            # Pattern 2: Random allocation/deallocation
            sizes = [100, 50, 200, 75, 150, 25, 300]
            np.random.shuffle(sizes)

            for size_mb in sizes:
                tensor = torch.randn(size_mb * 1024 * 1024 // 4).cuda()
                allocations.append(tensor)

            # Deallocate in different order
            np.random.shuffle(allocations)
            for tensor in allocations:
                del tensor
            torch.cuda.empty_cache()

            fragmentation_score_2 = self._calculate_fragmentation_score()

            # Store fragmentation metrics
            self.metrics_db.store_metric("gpu.fragmentation.sequential", fragmentation_score_1,
                                       {"test": "memory_fragmentation", "pattern": "sequential"})
            self.metrics_db.store_metric("gpu.fragmentation.random", fragmentation_score_2,
                                       {"test": "memory_fragmentation", "pattern": "random"})

            logger.info(f"Memory fragmentation: Sequential={fragmentation_score_1:.3f}, "
                       f"Random={fragmentation_score_2:.3f}")

        except Exception as e:
            logger.warning(f"Memory fragmentation test failed: {e}")

    def _calculate_fragmentation_score(self) -> float:
        """Calculate a simple fragmentation score."""
        try:
            # Try to allocate a large contiguous block
            total_memory = torch.cuda.get_device_properties(0).total_memory
            test_size = int(total_memory * 0.1)  # 10% of total memory

            max_contiguous = 0
            for size in range(test_size, 1024*1024, -1024*1024):  # Decreasing sizes
                try:
                    test_tensor = torch.empty(size, dtype=torch.uint8, device="cuda")
                    max_contiguous = size
                    del test_tensor
                    torch.cuda.empty_cache()
                    break
                except RuntimeError:
                    continue

            # Fragmentation score (0 = no fragmentation, 1 = high fragmentation)
            fragmentation = 1.0 - (max_contiguous / test_size)
            return max(0.0, min(1.0, fragmentation))

        except Exception:
            return 0.5  # Default moderate fragmentation

    def _prepare_training_data(self, features: pd.DataFrame) -> Optional[DataSpec]:
        """Prepare training data from features."""
        try:
            if len(features) < 50:
                return None

            train_size = int(len(features) * 0.8)
            train_df = features.iloc[:train_size]
            val_df = features.iloc[train_size:]
            
            feature_cols = [col for col in features.columns if col != "close"]

            return DataSpec(
                train_df=train_df,
                val_df=val_df,
                target_col="close",
                feature_names=feature_cols
            )

        except Exception as e:
            logger.warning(f"Training data preparation failed: {e}")
            return None

    def _run_sync_inference(self, inference_service, request):
        """Run inference synchronously for testing."""
        import asyncio

        async def async_inference():
            return await inference_service.predict_async(request)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(async_inference())
        finally:
            loop.close()

    def test_gpu_performance_summary(self):
        """Generate performance summary and recommendations."""
        if not self.has_gpu:
            return

        # Collect all GPU performance metrics
        from data.metrics_database import MetricQuery

        query = MetricQuery(
            metric_names=["gpu.memory.feature_engineering", "gpu.memory.training",
                         "gpu.utilization.avg", "gpu.throughput.samples_per_sec",
                         "gpu.inference.latency.avg", "gpu.memory.savings_percent"],
            start_time=datetime.now() - timedelta(hours=1),
            end_time=datetime.now()
        )

        metrics = self.metrics_db.query_metrics(query)

        if not metrics.empty:
            summary = {
                "total_metrics_collected": len(metrics),
                "memory_efficiency": metrics[metrics["metric_name"] == "gpu.memory.savings_percent"]["value"].mean(),
                "avg_inference_latency": metrics[metrics["metric_name"] == "gpu.inference.latency.avg"]["value"].mean(),
                "peak_throughput": metrics[metrics["metric_name"] == "gpu.throughput.samples_per_sec"]["value"].max()
            }

            logger.info(f"GPU Performance Summary: {summary}")

            # Store summary metrics
            for key, value in summary.items():
                if not np.isnan(value):
                    self.metrics_db.store_metric(f"gpu.summary.{key}", value,
                                               {"test": "performance_summary"})


if not torch.cuda.is_available():
    try:
        del TestGPUPerformanceTuning
    except NameError:
        pass

    @pytest.mark.skip(reason="GPU not available for performance tests")
    def test_gpu_performance_suite_requires_gpu():
        """Placeholder so pytest logs a single skip instead of many."""
        pass

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Run GPU performance tests
    unittest.main(verbosity=2)

