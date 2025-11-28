"""
Monitoring Service

Production monitoring and metrics collection for IB Forecast system.
Integrates with Prometheus and provides comprehensive system metrics.
"""

import os
import sys
import time
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import threading
import asyncio
from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server
import GPUtil

# Add paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.metrics_database import MetricsDatabase

logger = logging.getLogger(__name__)

class MonitoringService:
    """
    Monitoring service for collecting and exposing system and application metrics.

    Provides Prometheus-compatible metrics for:
    - System resources (CPU, memory, disk, network)
    - GPU utilization and memory
    - Application performance metrics
    - Business metrics
    - Custom application metrics
    """

    def __init__(self, metrics_db: Optional[MetricsDatabase] = None, port: int = 8000):
        """
        Initialize monitoring service.

        Args:
            metrics_db: Metrics database instance
            port: Port to expose Prometheus metrics
        """
        self.metrics_db = metrics_db or MetricsDatabase()
        self.port = port

        # Prometheus metrics
        self._setup_prometheus_metrics()

        # Monitoring configuration
        self.monitoring_config = {
            'collection_interval': 15,  # seconds
            'gpu_monitoring_enabled': True,
            'system_monitoring_enabled': True,
            'custom_metrics_enabled': True
        }

        # Background monitoring thread
        self.monitoring_thread = None
        self.stop_monitoring = False

        logger.info(f"Monitoring Service initialized on port {port}")

    def _setup_prometheus_metrics(self):
        """Setup Prometheus metric collectors."""

        # System metrics
        self.cpu_usage = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
        self.memory_usage = Gauge('system_memory_usage_percent', 'Memory usage percentage')
        self.memory_used = Gauge('system_memory_used_bytes', 'Memory used in bytes')
        self.memory_total = Gauge('system_memory_total_bytes', 'Total memory in bytes')
        self.disk_usage = Gauge('system_disk_usage_percent', 'Disk usage percentage')
        self.disk_used = Gauge('system_disk_used_bytes', 'Disk used in bytes')
        self.disk_total = Gauge('system_disk_total_bytes', 'Total disk space in bytes')

        # Network metrics
        self.network_bytes_sent = Counter('system_network_bytes_sent', 'Network bytes sent')
        self.network_bytes_recv = Counter('system_network_bytes_recv', 'Network bytes received')
        self.network_packets_sent = Counter('system_network_packets_sent', 'Network packets sent')
        self.network_packets_recv = Counter('system_network_packets_recv', 'Network packets received')

        # GPU metrics
        self.gpu_count = Gauge('gpu_device_count', 'Number of GPU devices')
        self.gpu_memory_used = Gauge('gpu_memory_used_bytes', 'GPU memory used in bytes', ['gpu_id'])
        self.gpu_memory_total = Gauge('gpu_memory_total_bytes', 'GPU memory total in bytes', ['gpu_id'])
        self.gpu_memory_usage_percent = Gauge('gpu_memory_usage_percent', 'GPU memory usage percentage', ['gpu_id'])
        self.gpu_utilization = Gauge('gpu_utilization_percent', 'GPU utilization percentage', ['gpu_id'])
        self.gpu_temperature = Gauge('gpu_temperature_celsius', 'GPU temperature in Celsius', ['gpu_id'])

        # Application metrics
        self.api_requests_total = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
        self.api_request_duration = Histogram('api_request_duration_seconds', 'API request duration in seconds', ['method', 'endpoint'])
        self.api_active_connections = Gauge('api_active_connections', 'Number of active API connections')

        # Forecast metrics
        self.forecast_requests_total = Counter('forecast_requests_total', 'Total forecast requests')
        self.forecast_generation_time = Histogram('forecast_generation_time_seconds', 'Time to generate forecast')
        self.forecast_model_accuracy = Gauge('forecast_model_accuracy', 'Model accuracy score')
        self.forecast_cache_hits = Counter('forecast_cache_hits_total', 'Cache hits for forecasts')
        self.forecast_cache_misses = Counter('forecast_cache_misses_total', 'Cache misses for forecasts')

        # Business metrics
        self.business_forecast_value = Gauge('business_forecast_value', 'Total forecast value generated')
        self.business_active_users = Gauge('business_active_users', 'Number of active users')
        self.business_api_calls_per_minute = Gauge('business_api_calls_per_minute', 'API calls per minute')

        # Error metrics
        self.errors_total = Counter('errors_total', 'Total errors', ['type', 'component'])
        self.error_rate = Gauge('error_rate_per_minute', 'Error rate per minute', ['component'])

        # Custom metrics
        self.custom_metrics = {}

    def start_monitoring(self):
        """Start the monitoring service."""
        # Start Prometheus HTTP server
        start_http_server(self.port)
        logger.info(f"Prometheus metrics server started on port {self.port}")

        # Start background monitoring
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        logger.info("Background monitoring started")

    def stop_monitoring(self):
        """Stop the monitoring service."""
        self.stop_monitoring = True
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Monitoring service stopped")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        logger.info("Starting monitoring loop")

        while not self.stop_monitoring:
            try:
                # Collect system metrics
                if self.monitoring_config['system_monitoring_enabled']:
                    self._collect_system_metrics()

                # Collect GPU metrics
                if self.monitoring_config['gpu_monitoring_enabled']:
                    self._collect_gpu_metrics()

                # Collect custom metrics
                if self.monitoring_config['custom_metrics_enabled']:
                    self._collect_custom_metrics()

                # Sleep until next collection
                time.sleep(self.monitoring_config['collection_interval'])

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Brief pause before retrying

    def _collect_system_metrics(self):
        """Collect system resource metrics."""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=None)
            self.cpu_usage.set(cpu_percent)

            # Memory
            memory = psutil.virtual_memory()
            self.memory_usage.set(memory.percent)
            self.memory_used.set(memory.used)
            self.memory_total.set(memory.total)

            # Disk
            disk = psutil.disk_usage('/')
            self.disk_usage.set(disk.percent)
            self.disk_used.set(disk.used)
            self.disk_total.set(disk.total)

            # Network
            net_io = psutil.net_io_counters()
            self.network_bytes_sent._value.set(net_io.bytes_sent)
            self.network_bytes_recv._value.set(net_io.bytes_recv)
            self.network_packets_sent._value.set(net_io.packets_sent)
            self.network_packets_recv._value.set(net_io.packets_recv)

            # Store in metrics database
            timestamp = datetime.now()
            system_metrics = [
                {'metric_name': 'system.cpu_usage', 'value': cpu_percent, 'timestamp': timestamp},
                {'metric_name': 'system.memory_usage', 'value': memory.percent, 'timestamp': timestamp},
                {'metric_name': 'system.disk_usage', 'value': disk.percent, 'timestamp': timestamp}
            ]

            for metric in system_metrics:
                self.metrics_db.store_metric(**metric)

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

    def _collect_gpu_metrics(self):
        """Collect GPU metrics."""
        try:
            gpus = GPUtil.getGPUs()

            if not gpus:
                self.gpu_count.set(0)
                return

            self.gpu_count.set(len(gpus))

            for i, gpu in enumerate(gpus):
                gpu_id = str(i)

                # Memory metrics
                memory_used_bytes = gpu.memoryUsed * 1024 * 1024  # Convert MB to bytes
                memory_total_bytes = gpu.memoryTotal * 1024 * 1024
                memory_usage_percent = gpu.memoryUtil * 100

                self.gpu_memory_used.labels(gpu_id=gpu_id).set(memory_used_bytes)
                self.gpu_memory_total.labels(gpu_id=gpu_id).set(memory_total_bytes)
                self.gpu_memory_usage_percent.labels(gpu_id=gpu_id).set(memory_usage_percent)

                # Utilization and temperature
                self.gpu_utilization.labels(gpu_id=gpu_id).set(gpu.load * 100)
                self.gpu_temperature.labels(gpu_id=gpu_id).set(gpu.temperature)

                # Store in metrics database
                timestamp = datetime.now()
                gpu_metrics = [
                    {'metric_name': f'gpu.{gpu_id}.memory_usage', 'value': memory_usage_percent, 'timestamp': timestamp},
                    {'metric_name': f'gpu.{gpu_id}.utilization', 'value': gpu.load * 100, 'timestamp': timestamp},
                    {'metric_name': f'gpu.{gpu_id}.temperature', 'value': gpu.temperature, 'timestamp': timestamp}
                ]

                for metric in gpu_metrics:
                    self.metrics_db.store_metric(**metric)

        except Exception as e:
            logger.error(f"Error collecting GPU metrics: {e}")

    def _collect_custom_metrics(self):
        """Collect custom application metrics."""
        try:
            # Query recent metrics from database for derived metrics
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=5)

            # Calculate error rate
            error_metrics = self.metrics_db.query_metrics(
                metric_names=['errors.api', 'errors.forecast', 'errors.system'],
                start_time=start_time,
                end_time=end_time
            )

            if not error_metrics.empty:
                for component in ['api', 'forecast', 'system']:
                    component_errors = error_metrics[error_metrics['metric_name'] == f'errors.{component}']
                    if not component_errors.empty:
                        error_count = len(component_errors)
                        error_rate = error_count / 5  # errors per minute
                        self.error_rate.labels(component=component).set(error_rate)

        except Exception as e:
            logger.error(f"Error collecting custom metrics: {e}")

    # API methods for recording application metrics

    def record_api_request(self, method: str, endpoint: str, status: int, duration: float):
        """
        Record API request metrics.

        Args:
            method: HTTP method
            endpoint: API endpoint
            status: HTTP status code
            duration: Request duration in seconds
        """
        self.api_requests_total.labels(method=method, endpoint=endpoint, status=str(status)).inc()
        self.api_request_duration.labels(method=method, endpoint=endpoint).observe(duration)

    def record_forecast_request(self, duration: Optional[float] = None, cache_hit: bool = False):
        """
        Record forecast request metrics.

        Args:
            duration: Forecast generation time in seconds
            cache_hit: Whether this was a cache hit
        """
        self.forecast_requests_total.inc()

        if duration is not None:
            self.forecast_generation_time.observe(duration)

        if cache_hit:
            self.forecast_cache_hits.inc()
        else:
            self.forecast_cache_misses.inc()

    def update_model_accuracy(self, accuracy: float):
        """
        Update model accuracy metric.

        Args:
            accuracy: Model accuracy score
        """
        self.forecast_model_accuracy.set(accuracy)

    def record_error(self, error_type: str, component: str):
        """
        Record error metrics.

        Args:
            error_type: Type of error
            component: Component where error occurred
        """
        self.errors_total.labels(type=error_type, component=component).inc()

    def update_business_metrics(self, forecast_value: Optional[float] = None,
                              active_users: Optional[int] = None,
                              api_calls_per_minute: Optional[float] = None):
        """
        Update business metrics.

        Args:
            forecast_value: Total forecast value
            active_users: Number of active users
            api_calls_per_minute: API calls per minute
        """
        if forecast_value is not None:
            self.business_forecast_value.set(forecast_value)

        if active_users is not None:
            self.business_active_users.set(active_users)

        if api_calls_per_minute is not None:
            self.business_api_calls_per_minute.set(api_calls_per_minute)

    def add_custom_metric(self, name: str, value: float, metric_type: str = 'gauge',
                         labels: Optional[Dict[str, str]] = None):
        """
        Add a custom metric.

        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric ('gauge', 'counter', 'histogram', 'summary')
            labels: Metric labels
        """
        if name in self.custom_metrics:
            metric = self.custom_metrics[name]
        else:
            if metric_type == 'gauge':
                metric = Gauge(name, f'Custom metric: {name}', labelnames=list(labels.keys()) if labels else [])
            elif metric_type == 'counter':
                metric = Counter(name, f'Custom metric: {name}', labelnames=list(labels.keys()) if labels else [])
            elif metric_type == 'histogram':
                metric = Histogram(name, f'Custom metric: {name}', labelnames=list(labels.keys()) if labels else [])
            elif metric_type == 'summary':
                metric = Summary(name, f'Custom metric: {name}', labelnames=list(labels.keys()) if labels else [])
            else:
                raise ValueError(f"Unknown metric type: {metric_type}")

            self.custom_metrics[name] = metric

        # Update metric value
        if labels:
            if metric_type in ['counter']:
                metric.labels(**labels).inc(value)
            else:
                metric.labels(**labels).set(value)
        else:
            if metric_type in ['counter']:
                metric.inc(value)
            else:
                metric.set(value)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current metrics.

        Returns:
            Dictionary with metrics summary
        """
        return {
            'cpu_usage': self.cpu_usage._value,
            'memory_usage': self.memory_usage._value,
            'gpu_count': self.gpu_count._value,
            'api_requests_total': self.api_requests_total._value.get(None, {}).get(None, {}).get(None, 0),
            'forecast_requests_total': self.forecast_requests_total._value,
            'errors_total': self.errors_total._value.get(None, {}).get(None, 0),
            'timestamp': datetime.now().isoformat()
        }

# Global monitoring service instance
_monitoring_service = None

def get_monitoring_service() -> MonitoringService:
    """Get the global monitoring service instance."""
    global _monitoring_service
    if _monitoring_service is None:
        _monitoring_service = MonitoringService()
    return _monitoring_service

def start_monitoring_service(port: int = 8000):
    """Start the monitoring service."""
    global _monitoring_service
    _monitoring_service = MonitoringService(port=port)
    _monitoring_service.start_monitoring()
    return _monitoring_service