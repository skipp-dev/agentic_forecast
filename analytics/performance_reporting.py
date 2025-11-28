"""
Performance Reporting

Automated reporting system for IB Forecast performance metrics.
Generates comprehensive reports on model performance, system health, and business metrics.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import json
from pathlib import Path

# Add paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.metrics_database import MetricsDatabase, MetricQuery

logger = logging.getLogger(__name__)

class PerformanceReporting:
    """
    Automated performance reporting system.

    Generates reports on:
    - Model performance metrics
    - System health and reliability
    - Business impact and ROI
    - Risk and compliance metrics
    """

    def __init__(self, metrics_db: Optional[MetricsDatabase] = None, reports_dir: str = 'reports'):
        """
        Initialize performance reporting.

        Args:
            metrics_db: Metrics database instance
            reports_dir: Directory to store generated reports
        """
        self.metrics_db = metrics_db or MetricsDatabase()
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)

        # Reporting configuration
        self.report_config = {
            'retention_days': 90,
            'schedule': {
                'daily': '08:00',
                'weekly': 'Monday 09:00',
                'monthly': '1st 10:00'
            },
            'formats': ['json', 'html', 'pdf'],
            'email_recipients': []  # Configure as needed
        }

        logger.info(f"Performance Reporting initialized with reports dir: {self.reports_dir}")

    def generate_daily_report(self) -> Dict[str, Any]:
        """
        Generate daily performance report.

        Returns:
            Daily report data
        """
        report_date = datetime.now().date()
        start_time = datetime.combine(report_date - timedelta(days=1), datetime.min.time())
        end_time = datetime.combine(report_date, datetime.min.time())

        report = {
            'report_type': 'daily',
            'date': report_date.isoformat(),
            'generated_at': datetime.now().isoformat(),
            'sections': {}
        }

        # Model Performance Section
        report['sections']['model_performance'] = self._generate_model_performance_section(start_time, end_time)

        # System Health Section
        report['sections']['system_health'] = self._generate_system_health_section(start_time, end_time)

        # Business Impact Section
        report['sections']['business_impact'] = self._generate_business_impact_section(start_time, end_time)

        # Risk Metrics Section
        report['sections']['risk_metrics'] = self._generate_risk_metrics_section(start_time, end_time)

        # Summary and Recommendations
        report['summary'] = self._generate_report_summary(report)
        report['recommendations'] = self._generate_recommendations(report)

        # Save report
        self._save_report(report, f"daily_report_{report_date}.json")

        return report

    def generate_weekly_report(self) -> Dict[str, Any]:
        """
        Generate weekly performance report.

        Returns:
            Weekly report data
        """
        report_date = datetime.now().date()
        start_time = datetime.combine(report_date - timedelta(days=7), datetime.min.time())
        end_time = datetime.combine(report_date, datetime.min.time())

        report = {
            'report_type': 'weekly',
            'period': f"{start_time.date()} to {end_time.date()}",
            'generated_at': datetime.now().isoformat(),
            'sections': {}
        }

        # Model Performance Trends
        report['sections']['model_performance_trends'] = self._generate_model_performance_trends(start_time, end_time)

        # System Reliability
        report['sections']['system_reliability'] = self._generate_system_reliability_section(start_time, end_time)

        # Business Metrics
        report['sections']['business_metrics'] = self._generate_business_metrics_section(start_time, end_time)

        # Risk Analysis
        report['sections']['risk_analysis'] = self._generate_risk_analysis_section(start_time, end_time)

        # Summary and Recommendations
        report['summary'] = self._generate_report_summary(report)
        report['recommendations'] = self._generate_recommendations(report)

        # Save report
        week_start = start_time.date()
        self._save_report(report, f"weekly_report_{week_start}.json")

        return report

    def generate_monthly_report(self) -> Dict[str, Any]:
        """
        Generate monthly performance report.

        Returns:
            Monthly report data
        """
        report_date = datetime.now().date()
        start_time = datetime.combine(report_date.replace(day=1) - timedelta(days=1), datetime.min.time())
        end_time = datetime.combine(report_date, datetime.min.time())

        report = {
            'report_type': 'monthly',
            'period': f"{start_time.date()} to {end_time.date()}",
            'generated_at': datetime.now().isoformat(),
            'sections': {}
        }

        # Comprehensive Model Analysis
        report['sections']['model_analysis'] = self._generate_model_analysis_section(start_time, end_time)

        # System Performance Review
        report['sections']['system_performance_review'] = self._generate_system_performance_review(start_time, end_time)

        # Business Value Assessment
        report['sections']['business_value'] = self._generate_business_value_section(start_time, end_time)

        # Compliance and Risk Review
        report['sections']['compliance_risk_review'] = self._generate_compliance_risk_review(start_time, end_time)

        # Summary and Recommendations
        report['summary'] = self._generate_report_summary(report)
        report['recommendations'] = self._generate_recommendations(report)

        # Save report
        month_start = start_time.date()
        self._save_report(report, f"monthly_report_{month_start}.json")

        return report

    def generate_custom_report(self, start_date: datetime, end_date: datetime,
                             metrics: List[str] = None) -> Dict[str, Any]:
        """
        Generate custom performance report for specified period.

        Args:
            start_date: Report start date
            end_date: Report end date
            metrics: List of specific metrics to include

        Returns:
            Custom report data
        """
        report = {
            'report_type': 'custom',
            'period': f"{start_date.date()} to {end_date.date()}",
            'generated_at': datetime.now().isoformat(),
            'sections': {}
        }

        # Query specified metrics or all metrics
        metric_names = metrics or [
            'model.accuracy', 'model.precision', 'model.recall', 'model.f1_score',
            'system.cpu_usage', 'system.memory_usage', 'system.latency', 'system.throughput',
            'risk.value_at_risk', 'risk.sharpe_ratio', 'risk.max_drawdown'
        ]

        query = MetricQuery(
            metric_names=metric_names,
            start_time=start_date,
            end_time=end_date,
            aggregation='mean',
            interval='1h'
        )

        metrics_data = self.metrics_db.query_metrics(query)

        if not metrics_data.empty:
            report['sections']['metrics_summary'] = self._analyze_metrics_data(metrics_data)
            report['sections']['trends_analysis'] = self._analyze_trends(metrics_data)
            report['sections']['anomaly_detection'] = self._detect_anomalies(metrics_data)

        # Summary and Recommendations
        report['summary'] = self._generate_report_summary(report)
        report['recommendations'] = self._generate_recommendations(report)

        # Save report
        report_id = f"custom_report_{start_date.date()}_{end_date.date()}"
        self._save_report(report, f"{report_id}.json")

        return report

    def _generate_model_performance_section(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate model performance section."""
        query = MetricQuery(
            metric_names=['model.accuracy', 'model.precision', 'model.recall', 'model.f1_score'],
            start_time=start_time,
            end_time=end_time,
            aggregation='mean',
            interval='1h'
        )

        metrics = self.metrics_db.query_metrics(query)

        if metrics.empty:
            return {'status': 'no_data'}

        performance = {}
        for metric_name in metrics['metric_name'].unique():
            values = metrics[metrics['metric_name'] == metric_name]['value']
            performance[metric_name.split('.')[-1]] = {
                'mean': float(values.mean()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
                'count': len(values)
            }

        return {
            'status': 'success',
            'performance_metrics': performance,
            'overall_score': self._calculate_overall_model_score(performance)
        }

    def _generate_system_health_section(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate system health section."""
        query = MetricQuery(
            metric_names=['system.cpu_usage', 'system.memory_usage', 'system.disk_usage', 'system.latency'],
            start_time=start_time,
            end_time=end_time,
            aggregation='mean',
            interval='1h'
        )

        metrics = self.metrics_db.query_metrics(query)

        if metrics.empty:
            return {'status': 'no_data'}

        health = {}
        for metric_name in metrics['metric_name'].unique():
            values = metrics[metrics['metric_name'] == metric_name]['value']
            health[metric_name.split('.')[-1]] = {
                'mean': float(values.mean()),
                'std': float(values.std()),
                'max': float(values.max()),
                'availability': self._calculate_availability(values)
            }

        return {
            'status': 'success',
            'health_metrics': health,
            'overall_health': self._assess_overall_health(health)
        }

    def _generate_business_impact_section(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate business impact section."""
        # This would integrate with business metrics - placeholder for now
        return {
            'status': 'placeholder',
            'forecast_accuracy_impact': 'To be implemented with business metrics integration',
            'trading_volume_impact': 'To be implemented with business metrics integration',
            'cost_savings': 'To be implemented with business metrics integration'
        }

    def _generate_risk_metrics_section(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate risk metrics section."""
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

        risk = {}
        for metric_name in metrics['metric_name'].unique():
            values = metrics[metrics['metric_name'] == metric_name]['value']
            risk[metric_name.split('.')[-1]] = {
                'mean': float(values.mean()),
                'std': float(values.std()),
                'worst_case': float(values.max()) if 'max_drawdown' in metric_name or 'value_at_risk' in metric_name else float(values.min())
            }

        return {
            'status': 'success',
            'risk_metrics': risk,
            'risk_assessment': self._assess_risk_level(risk)
        }

    def _calculate_overall_model_score(self, performance: Dict[str, Any]) -> float:
        """Calculate overall model performance score."""
        if not performance:
            return 0.0

        # Weighted average of key metrics
        weights = {'accuracy': 0.4, 'precision': 0.2, 'recall': 0.2, 'f1_score': 0.2}
        score = 0.0
        total_weight = 0.0

        for metric, weight in weights.items():
            if metric in performance:
                score += performance[metric]['mean'] * weight
                total_weight += weight

        return score / total_weight if total_weight > 0 else 0.0

    def _calculate_availability(self, values: pd.Series) -> float:
        """Calculate system availability percentage."""
        # Simple availability calculation - values below threshold considered available
        threshold = 0.95  # 95% for most metrics
        available_count = (values < threshold).sum()
        return float(available_count / len(values)) if len(values) > 0 else 0.0

    def _assess_overall_health(self, health: Dict[str, Any]) -> str:
        """Assess overall system health."""
        if not health:
            return 'unknown'

        # Simple health assessment based on availability
        avg_availability = np.mean([m.get('availability', 0) for m in health.values()])

        if avg_availability > 0.99:
            return 'excellent'
        elif avg_availability > 0.95:
            return 'good'
        elif avg_availability > 0.90:
            return 'fair'
        else:
            return 'poor'

    def _assess_risk_level(self, risk: Dict[str, Any]) -> str:
        """Assess overall risk level."""
        if not risk:
            return 'unknown'

        # Risk assessment based on VaR and max drawdown
        var = risk.get('value_at_risk', {}).get('mean', 0)
        max_dd = risk.get('max_drawdown', {}).get('worst_case', 0)

        if var > 0.05 or max_dd > 0.1:  # 5% VaR or 10% drawdown
            return 'high'
        elif var > 0.02 or max_dd > 0.05:  # 2% VaR or 5% drawdown
            return 'moderate'
        else:
            return 'low'

    def _generate_report_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall report summary."""
        summary = {
            'report_type': report.get('report_type'),
            'period': report.get('period') or report.get('date'),
            'overall_status': 'success'
        }

        # Check for any sections with no data
        no_data_sections = [section for section, data in report.get('sections', {}).items()
                          if data.get('status') == 'no_data']

        if no_data_sections:
            summary['warnings'] = f"No data available for sections: {', '.join(no_data_sections)}"

        return summary

    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on report data."""
        recommendations = []

        sections = report.get('sections', {})

        # Model performance recommendations
        model_perf = sections.get('model_performance', {})
        if model_perf.get('status') == 'success':
            score = model_perf.get('overall_score', 0)
            if score < 0.8:
                recommendations.append("Model performance needs improvement - consider retraining or feature engineering")

        # System health recommendations
        sys_health = sections.get('system_health', {})
        if sys_health.get('status') == 'success':
            health = sys_health.get('overall_health')
            if health in ['fair', 'poor']:
                recommendations.append("System health requires attention - review resource usage and optimize performance")

        # Risk recommendations
        risk = sections.get('risk_metrics', {})
        if risk.get('status') == 'success':
            risk_level = risk.get('risk_assessment')
            if risk_level == 'high':
                recommendations.append("High risk levels detected - implement additional risk controls")

        return recommendations

    def _save_report(self, report: Dict[str, Any], filename: str) -> None:
        """Save report to file."""
        filepath = self.reports_dir / filename
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Report saved to {filepath}")

    # Additional helper methods for weekly/monthly reports would go here
    # These are simplified versions for brevity

    def _generate_model_performance_trends(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate model performance trends for weekly report."""
        return self._generate_model_performance_section(start_time, end_time)

    def _generate_system_reliability_section(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate system reliability section for weekly report."""
        return self._generate_system_health_section(start_time, end_time)

    def _generate_business_metrics_section(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate business metrics section for weekly report."""
        return self._generate_business_impact_section(start_time, end_time)

    def _generate_risk_analysis_section(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate risk analysis section for weekly report."""
        return self._generate_risk_metrics_section(start_time, end_time)

    def _generate_model_analysis_section(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate comprehensive model analysis for monthly report."""
        return self._generate_model_performance_section(start_time, end_time)

    def _generate_system_performance_review(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate system performance review for monthly report."""
        return self._generate_system_health_section(start_time, end_time)

    def _generate_business_value_section(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate business value assessment for monthly report."""
        return self._generate_business_impact_section(start_time, end_time)

    def _generate_compliance_risk_review(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate compliance and risk review for monthly report."""
        return self._generate_risk_metrics_section(start_time, end_time)

    def _analyze_metrics_data(self, metrics_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze metrics data for custom reports."""
        analysis = {}

        for metric_name in metrics_df['metric_name'].unique():
            values = metrics_df[metrics_df['metric_name'] == metric_name]['value']
            analysis[metric_name] = {
                'mean': float(values.mean()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
                'trend': 'increasing' if values.iloc[-1] > values.iloc[0] else 'decreasing'
            }

        return analysis

    def _analyze_trends(self, metrics_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trends in metrics data."""
        trends = {}

        for metric_name in metrics_df['metric_name'].unique():
            values = metrics_df[metrics_df['metric_name'] == metric_name]['value'].values
            if len(values) > 1:
                # Simple linear trend
                x = np.arange(len(values))
                slope = np.polyfit(x, values, 1)[0]
                trends[metric_name] = 'increasing' if slope > 0 else 'decreasing'
            else:
                trends[metric_name] = 'stable'

        return trends

    def _detect_anomalies(self, metrics_df: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies in metrics data."""
        anomalies = {}

        for metric_name in metrics_df['metric_name'].unique():
            values = metrics_df[metrics_df['metric_name'] == metric_name]['value'].values
            if len(values) > 10:
                # Simple anomaly detection using z-score
                mean_val = np.mean(values)
                std_val = np.std(values)
                z_scores = np.abs((values - mean_val) / std_val)
                anomaly_indices = np.where(z_scores > 3)[0]  # 3 sigma threshold

                if len(anomaly_indices) > 0:
                    anomalies[metric_name] = {
                        'count': len(anomaly_indices),
                        'indices': anomaly_indices.tolist(),
                        'values': values[anomaly_indices].tolist()
                    }

        return anomalies