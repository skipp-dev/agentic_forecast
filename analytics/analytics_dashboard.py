"""
Analytics Dashboard

Web-based dashboard for IB Forecast system analytics and reporting.
Provides real-time metrics visualization and performance monitoring.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import json

# Add paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.metrics_database import MetricsDatabase, MetricQuery

logger = logging.getLogger(__name__)

class AnalyticsDashboard:
    """
    Analytics dashboard for IB Forecast system.

    Provides real-time visualization of:
    - Model performance metrics
    - System health and throughput
    - Risk analytics
    - Market intelligence
    """

    def __init__(self, metrics_db: Optional[MetricsDatabase] = None):
        """
        Initialize analytics dashboard.

        Args:
            metrics_db: Metrics database instance
        """
        self.metrics_db = metrics_db or MetricsDatabase()

        # Dashboard configuration
        self.dashboard_config = {
            'refresh_interval': 60,  # seconds
            'metrics_history_days': 7,
            'alert_thresholds': {
                'model_accuracy': 0.8,
                'system_latency': 1000,  # ms
                'memory_usage': 0.9  # 90%
            }
        }

        logger.info("Analytics Dashboard initialized")

    def get_system_overview(self) -> Dict[str, Any]:
        """
        Get system overview metrics.

        Returns:
            Dictionary with system health metrics
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=self.dashboard_config['metrics_history_days'])

        # Query recent metrics
        query = MetricQuery(
            metric_names=['system.cpu_usage', 'system.memory_usage', 'system.disk_usage'],
            start_time=start_time,
            end_time=end_time,
            aggregation='mean',
            interval='1h'
        )

        metrics = self.metrics_db.query_metrics(query)

        if metrics.empty:
            return {'status': 'no_data'}

        # Calculate current values
        current_metrics = {}
        for metric_name in ['system.cpu_usage', 'system.memory_usage', 'system.disk_usage']:
            metric_data = metrics[metrics['metric_name'] == metric_name]
            if not metric_data.empty:
                current_metrics[metric_name.split('.')[-1]] = metric_data['value'].iloc[-1]

        return {
            'status': 'healthy',
            'timestamp': end_time.isoformat(),
            'current': current_metrics,
            'trends': self._calculate_trends(metrics)
        }

    def get_model_performance(self) -> Dict[str, Any]:
        """
        Get model performance metrics.

        Returns:
            Dictionary with model performance data
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=self.dashboard_config['metrics_history_days'])

        # Query model metrics
        query = MetricQuery(
            metric_names=['model.accuracy', 'model.precision', 'model.recall', 'model.f1_score'],
            start_time=start_time,
            end_time=end_time,
            aggregation='mean',
            interval='1d'
        )

        metrics = self.metrics_db.query_metrics(query)

        if metrics.empty:
            return {'status': 'no_data'}

        # Group by model type
        performance_by_model = {}
        for metric_name in metrics['metric_name'].unique():
            metric_data = metrics[metrics['metric_name'] == metric_name]
            performance_by_model[metric_name] = {
                'current': metric_data['value'].iloc[-1] if not metric_data.empty else None,
                'trend': self._calculate_trend(metric_data['value'].values) if len(metric_data) > 1 else 'stable'
            }

        return {
            'status': 'available',
            'timestamp': end_time.isoformat(),
            'models': performance_by_model,
            'alerts': self._check_performance_alerts(performance_by_model)
        }

    def get_risk_analytics(self) -> Dict[str, Any]:
        """
        Get risk analytics data.

        Returns:
            Dictionary with risk metrics
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=self.dashboard_config['metrics_history_days'])

        # Query risk metrics
        query = MetricQuery(
            metric_names=['risk.value_at_risk', 'risk.sharpe_ratio', 'risk.max_drawdown'],
            start_time=start_time,
            end_time=end_time,
            aggregation='mean',
            interval='1d'
        )

        metrics = self.metrics_db.query_metrics(query)

        if metrics.empty:
            return {'status': 'no_data'}

        risk_metrics = {}
        for metric_name in ['risk.value_at_risk', 'risk.sharpe_ratio', 'risk.max_drawdown']:
            metric_data = metrics[metrics['metric_name'] == metric_name]
            if not metric_data.empty:
                risk_metrics[metric_name.split('.')[-1]] = {
                    'current': metric_data['value'].iloc[-1],
                    'historical': metric_data['value'].tolist()
                }

        return {
            'status': 'available',
            'timestamp': end_time.isoformat(),
            'risk_metrics': risk_metrics,
            'risk_assessment': self._assess_portfolio_risk(risk_metrics)
        }

    def get_market_intelligence(self) -> Dict[str, Any]:
        """
        Get market intelligence insights.

        Returns:
            Dictionary with market analysis
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=self.dashboard_config['metrics_history_days'])

        # Query market metrics
        query = MetricQuery(
            metric_names=['market.volatility', 'market.trend_strength', 'market.sentiment'],
            start_time=start_time,
            end_time=end_time,
            aggregation='mean',
            interval='1d'
        )

        metrics = self.metrics_db.query_metrics(query)

        if metrics.empty:
            return {'status': 'no_data'}

        market_insights = {}
        for metric_name in ['market.volatility', 'market.trend_strength', 'market.sentiment']:
            metric_data = metrics[metrics['metric_name'] == metric_name]
            if not metric_data.empty:
                market_insights[metric_name.split('.')[-1]] = {
                    'current': metric_data['value'].iloc[-1],
                    'trend': self._calculate_trend(metric_data['value'].values)
                }

        return {
            'status': 'available',
            'timestamp': end_time.isoformat(),
            'market_insights': market_insights,
            'recommendations': self._generate_market_recommendations(market_insights)
        }

    def generate_performance_report(self, report_type: str = 'daily') -> Dict[str, Any]:
        """
        Generate performance report.

        Args:
            report_type: Type of report ('daily', 'weekly', 'monthly')

        Returns:
            Performance report data
        """
        if report_type == 'daily':
            days = 1
        elif report_type == 'weekly':
            days = 7
        elif report_type == 'monthly':
            days = 30
        else:
            days = 1

        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        # Collect all relevant metrics
        query = MetricQuery(
            metric_names=[
                'model.accuracy', 'model.precision', 'model.recall',
                'system.latency', 'system.throughput',
                'risk.value_at_risk', 'risk.sharpe_ratio'
            ],
            start_time=start_time,
            end_time=end_time,
            aggregation='mean',
            interval='1h'
        )

        metrics = self.metrics_db.query_metrics(query)

        if metrics.empty:
            return {'status': 'no_data'}

        # Generate report summary
        report = {
            'report_type': report_type,
            'period': f"{start_time.date()} to {end_time.date()}",
            'generated_at': end_time.isoformat(),
            'summary': self._generate_report_summary(metrics),
            'recommendations': self._generate_report_recommendations(metrics)
        }

        return report

    def _calculate_trends(self, metrics_df: pd.DataFrame) -> Dict[str, str]:
        """Calculate trends for metrics."""
        trends = {}
        for metric_name in metrics_df['metric_name'].unique():
            values = metrics_df[metrics_df['metric_name'] == metric_name]['value'].values
            if len(values) > 1:
                trends[metric_name] = self._calculate_trend(values)
            else:
                trends[metric_name] = 'stable'
        return trends

    def _calculate_trend(self, values: np.ndarray) -> str:
        """Calculate trend direction from values."""
        if len(values) < 2:
            return 'stable'

        recent_avg = np.mean(values[-3:]) if len(values) >= 3 else values[-1]
        earlier_avg = np.mean(values[:-3]) if len(values) > 3 else values[0]

        if recent_avg > earlier_avg * 1.05:
            return 'increasing'
        elif recent_avg < earlier_avg * 0.95:
            return 'decreasing'
        else:
            return 'stable'

    def _check_performance_alerts(self, performance_data: Dict[str, Any]) -> List[str]:
        """Check for performance alerts."""
        alerts = []

        thresholds = self.dashboard_config['alert_thresholds']

        for metric, data in performance_data.items():
            if metric == 'accuracy' and data.get('current', 0) < thresholds['model_accuracy']:
                alerts.append(f"Model accuracy below threshold: {data['current']:.3f}")
            elif metric == 'latency' and data.get('current', 0) > thresholds['system_latency']:
                alerts.append(f"System latency above threshold: {data['current']:.1f}ms")

        return alerts

    def _assess_portfolio_risk(self, risk_metrics: Dict[str, Any]) -> str:
        """Assess overall portfolio risk."""
        if not risk_metrics:
            return 'unknown'

        var = risk_metrics.get('value_at_risk', {}).get('current', 0)
        sharpe = risk_metrics.get('sharpe_ratio', {}).get('current', 0)

        if var > 0.05:  # 5% VaR threshold
            return 'high_risk'
        elif sharpe > 1.0:
            return 'low_risk'
        else:
            return 'moderate_risk'

    def _generate_market_recommendations(self, market_insights: Dict[str, Any]) -> List[str]:
        """Generate market recommendations based on insights."""
        recommendations = []

        volatility = market_insights.get('volatility', {}).get('current', 0)
        trend = market_insights.get('trend_strength', {}).get('trend', 'stable')
        sentiment = market_insights.get('sentiment', {}).get('current', 0)

        if volatility > 0.3:
            recommendations.append("High market volatility detected - consider risk management strategies")
        if trend == 'increasing' and sentiment > 0.6:
            recommendations.append("Strong bullish trend with positive sentiment")
        elif trend == 'decreasing' and sentiment < 0.4:
            recommendations.append("Bearish market conditions - exercise caution")

        return recommendations

    def _generate_report_summary(self, metrics_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate report summary statistics."""
        summary = {}

        for metric_name in metrics_df['metric_name'].unique():
            values = metrics_df[metrics_df['metric_name'] == metric_name]['value']
            if not values.empty:
                summary[metric_name] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max())
                }

        return summary

    def _generate_report_recommendations(self, metrics_df: pd.DataFrame) -> List[str]:
        """Generate report recommendations."""
        recommendations = []

        # Check model performance
        accuracy = metrics_df[metrics_df['metric_name'] == 'model.accuracy']['value']
        if not accuracy.empty and accuracy.mean() < 0.8:
            recommendations.append("Model accuracy needs improvement - consider retraining")

        # Check system performance
        latency = metrics_df[metrics_df['metric_name'] == 'system.latency']['value']
        if not latency.empty and latency.mean() > 1000:
            recommendations.append("System latency is high - optimize performance")

        return recommendations