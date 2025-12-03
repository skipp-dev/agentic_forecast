#!/usr/bin/env python3
"""
Performance Monitoring and Optimization Script

Analyzes pipeline performance bottlenecks and provides optimization recommendations.
"""

import os
import sys
import yaml
import logging
import argparse
import time
import psutil
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def setup_logging(log_level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/performance_monitoring.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config():
    """Load configuration from config.yaml"""
    config_path = "config.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

class PerformanceMonitor:
    """Monitor and analyze pipeline performance."""

    def __init__(self, config=None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Performance metrics storage
        self.metrics = {
            'execution_times': {},
            'memory_usage': {},
            'cpu_usage': {},
            'bottlenecks': [],
            'recommendations': []
        }

    def monitor_execution_time(self, component_name, func, *args, **kwargs):
        """Monitor execution time of a component."""
        start_time = time.time()
        start_memory = psutil.virtual_memory().used
        start_cpu = psutil.cpu_percent(interval=None)

        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
            self.logger.error(f"Component {component_name} failed: {e}")

        end_time = time.time()
        end_memory = psutil.virtual_memory().used
        end_cpu = psutil.cpu_percent(interval=None)

        execution_time = end_time - start_time
        memory_delta = end_memory - start_memory
        avg_cpu = (start_cpu + end_cpu) / 2

        self.metrics['execution_times'][component_name] = execution_time
        self.metrics['memory_usage'][component_name] = memory_delta
        self.metrics['cpu_usage'][component_name] = avg_cpu

        self.logger.info(f"{component_name}: {execution_time:.2f}s, Memory: {memory_delta/1024/1024:.1f}MB, CPU: {avg_cpu:.1f}%")

        return result, success

    def analyze_bottlenecks(self):
        """Analyze performance bottlenecks."""
        execution_times = self.metrics['execution_times']

        if not execution_times:
            self.logger.warning("No execution time data available for bottleneck analysis")
            return

        # Find slowest components
        sorted_times = sorted(execution_times.items(), key=lambda x: x[1], reverse=True)
        total_time = sum(execution_times.values())

        self.logger.info("Performance Analysis:")
        self.logger.info(f"Total execution time: {total_time:.2f}s")

        bottlenecks = []
        for component, exec_time in sorted_times[:3]:  # Top 3 bottlenecks
            percentage = (exec_time / total_time) * 100
            self.logger.info(f"  {component}: {exec_time:.2f}s ({percentage:.1f}%)")

            if percentage > 50:
                bottlenecks.append({
                    'component': component,
                    'time': exec_time,
                    'percentage': percentage,
                    'severity': 'critical'
                })
            elif percentage > 25:
                bottlenecks.append({
                    'component': component,
                    'time': exec_time,
                    'percentage': percentage,
                    'severity': 'high'
                })

        self.metrics['bottlenecks'] = bottlenecks

    def generate_optimization_recommendations(self):
        """Generate optimization recommendations based on analysis."""
        recommendations = []

        execution_times = self.metrics['execution_times']
        memory_usage = self.metrics['memory_usage']
        cpu_usage = self.metrics['cpu_usage']

        # Data ingestion optimizations
        if 'data_ingestion' in execution_times and execution_times['data_ingestion'] > 300:  # 5 minutes
            recommendations.append({
                'component': 'data_ingestion',
                'issue': 'Slow data ingestion',
                'recommendation': 'Consider parallel data fetching, caching, or switching to faster data sources',
                'priority': 'high'
            })

        # Feature engineering optimizations
        if 'feature_engineering' in execution_times and execution_times['feature_engineering'] > 600:  # 10 minutes
            recommendations.append({
                'component': 'feature_engineering',
                'issue': 'Slow feature engineering',
                'recommendation': 'Implement parallel processing, use GPU acceleration, or optimize feature calculations',
                'priority': 'high'
            })

        # Model training optimizations
        if 'model_training' in execution_times and execution_times['model_training'] > 1800:  # 30 minutes
            recommendations.append({
                'component': 'model_training',
                'issue': 'Slow model training',
                'recommendation': 'Use GPU acceleration, reduce model complexity, or implement early stopping',
                'priority': 'critical'
            })

        # Memory optimizations
        for component, mem_usage in memory_usage.items():
            if mem_usage > 2 * 1024 * 1024 * 1024:  # 2GB
                recommendations.append({
                    'component': component,
                    'issue': 'High memory usage',
                    'recommendation': 'Implement data chunking, use memory-efficient data structures, or increase RAM',
                    'priority': 'medium'
                })

        # CPU optimizations
        for component, cpu_usage in cpu_usage.items():
            if cpu_usage > 80:
                recommendations.append({
                    'component': component,
                    'issue': 'High CPU usage',
                    'recommendation': 'Consider parallel processing or distributing workload across multiple cores',
                    'priority': 'medium'
                })

        # General recommendations
        recommendations.extend([
            {
                'component': 'general',
                'issue': 'No caching implemented',
                'recommendation': 'Implement result caching to avoid redundant computations',
                'priority': 'medium'
            },
            {
                'component': 'general',
                'issue': 'No performance monitoring',
                'recommendation': 'Set up continuous performance monitoring and alerting',
                'priority': 'low'
            }
        ])

        self.metrics['recommendations'] = recommendations
        return recommendations

    def create_performance_report(self, output_path=None):
        """Create a comprehensive performance report."""
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f'performance_report_{timestamp}.html'

        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Agentic Forecast Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .metric {{ background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .bottleneck {{ background: #ffebee; border-left: 5px solid #f44336; padding: 10px; margin: 10px 0; }}
                .recommendation {{ background: #e8f5e8; border-left: 5px solid #4caf50; padding: 10px; margin: 10px 0; }}
                .high {{ color: #f44336; }}
                .medium {{ color: #ff9800; }}
                .low {{ color: #4caf50; }}
            </style>
        </head>
        <body>
            <h1>Agentic Forecast Performance Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

            <h2>Execution Times</h2>
            {self._generate_execution_times_html()}

            <h2>Resource Usage</h2>
            {self._generate_resource_usage_html()}

            <h2>Performance Bottlenecks</h2>
            {self._generate_bottlenecks_html()}

            <h2>Optimization Recommendations</h2>
            {self._generate_recommendations_html()}
        </body>
        </html>
        """

        with open(output_path, 'w') as f:
            f.write(html_content)

        self.logger.info(f"Performance report saved to {output_path}")
        return output_path

    def _generate_execution_times_html(self):
        """Generate HTML for execution times."""
        html = ""
        for component, time_taken in self.metrics['execution_times'].items():
            html += f'<div class="metric">{component}: {time_taken:.2f} seconds</div>'
        return html

    def _generate_resource_usage_html(self):
        """Generate HTML for resource usage."""
        html = ""
        for component in self.metrics['execution_times'].keys():
            mem_usage = self.metrics['memory_usage'].get(component, 0)
            cpu_usage = self.metrics['cpu_usage'].get(component, 0)
            html += f'<div class="metric">{component}: Memory: {mem_usage/1024/1024:.1f} MB, CPU: {cpu_usage:.1f}%</div>'
        return html

    def _generate_bottlenecks_html(self):
        """Generate HTML for bottlenecks."""
        html = ""
        for bottleneck in self.metrics['bottlenecks']:
            severity_class = bottleneck['severity']
            html += f'<div class="bottleneck {severity_class}">'
            html += f"<strong>{bottleneck['component']}</strong>: {bottleneck['time']:.2f}s "
            html += f"({bottleneck['percentage']:.1f}% of total time)"
            html += '</div>'
        return html

    def _generate_recommendations_html(self):
        """Generate HTML for recommendations."""
        html = ""
        for rec in self.metrics['recommendations']:
            priority_class = rec['priority']
            html += f'<div class="recommendation {priority_class}">'
            html += f"<strong>{rec['component']} - {rec['issue']}</strong><br>"
            html += f"<em>Recommendation:</em> {rec['recommendation']}"
            html += '</div>'
        return html

def run_performance_analysis():
    """Run complete performance analysis."""
    logger = setup_logging()

    monitor = PerformanceMonitor()

    logger.info("Starting performance analysis...")

    # Import the component functions
    from scripts.run_data_ingestion import run_data_ingestion
    from scripts.run_feature_engineering import run_feature_engineering
    from scripts.run_model_training import run_model_training
    from scripts.run_monitoring import run_monitoring

    # Monitor each component
    logger.info("Analyzing data ingestion performance...")
    _, _ = monitor.monitor_execution_time('data_ingestion', run_data_ingestion)

    # For feature engineering, we need input data
    # This would normally use the output from data ingestion
    logger.info("Analyzing feature engineering performance...")
    # Skip if no input data available

    # For model training, we need features
    logger.info("Analyzing model training performance...")
    # Skip if no feature data available

    # Analyze bottlenecks
    monitor.analyze_bottlenecks()

    # Generate recommendations
    recommendations = monitor.generate_optimization_recommendations()

    # Create report
    report_path = monitor.create_performance_report()

    logger.info("Performance analysis completed")
    logger.info(f"Report generated: {report_path}")

    return monitor.metrics

def main():
    parser = argparse.ArgumentParser(description='Run performance monitoring and optimization')
    parser.add_argument('--output', help='Output path for performance report')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    parser.add_argument('--run-analysis', action='store_true',
                       help='Run complete performance analysis')

    args = parser.parse_args()

    # Setup logging
    log_level = getattr(logging, args.log_level)
    logger = setup_logging(log_level)

    try:
        if args.run_analysis:
            # Run complete analysis
            metrics = run_performance_analysis()

            print(f"\nPerformance Analysis Summary:")
            print(f"Components analyzed: {len(metrics['execution_times'])}")
            print(f"Bottlenecks identified: {len(metrics['bottlenecks'])}")
            print(f"Recommendations: {len(metrics['recommendations'])}")

            if metrics['bottlenecks']:
                print(f"\nTop Bottlenecks:")
                for bottleneck in metrics['bottlenecks'][:3]:
                    print(f"  - {bottleneck['component']}: {bottleneck['time']:.2f}s")

        else:
            # Just create a basic report template
            monitor = PerformanceMonitor()
            report_path = monitor.create_performance_report(args.output)
            print(f"Performance report template created: {report_path}")

    except Exception as e:
        logger.error(f"Performance analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()