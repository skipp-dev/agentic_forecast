#!/usr/bin/env python3
"""
Guardrails System for Automated Performance Monitoring and Alerts

This module implements comprehensive guardrails to monitor system performance,
detect anomalies, and trigger automated responses to maintain system reliability.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import threading
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.config_loader import get_guardrail_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceThreshold:
    """Performance threshold configuration."""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    direction: str  # 'above' or 'below'
    rolling_window: int  # Number of periods to consider

@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    condition: str
    severity: str  # 'warning', 'critical', 'info'
    cooldown_minutes: int
    actions: List[str]

class GuardrailsSystem:
    """
    Comprehensive guardrails system for monitoring and maintaining system health.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the guardrails system.

        Args:
            config_path: Path to guardrails configuration file
        """
        self.config_path = config_path or project_root / "config" / "guardrails_config.json"
        self.thresholds = self._load_thresholds()
        self.alert_rules = self._load_alert_rules()
        self.performance_history = []
        self.alert_history = []
        self.monitoring_active = False
        self.monitor_thread = None

        # Alert callbacks
        self.alert_callbacks: Dict[str, Callable] = {}

        logger.info("Initialized Guardrails System")

    def _load_thresholds(self) -> List[PerformanceThreshold]:
        """Load performance thresholds from configuration."""
        try:
            config = get_guardrail_config()
            thresholds_data = config.get('thresholds', [])
            if not thresholds_data:
                # Default thresholds
                thresholds_data = [
                    {
                        "metric_name": "mae",
                        "warning_threshold": 200.0,
                        "critical_threshold": 300.0,
                        "direction": "above",
                        "rolling_window": 10
                    },
                    {
                        "metric_name": "directional_accuracy",
                        "warning_threshold": 0.45,
                        "critical_threshold": 0.35,
                        "direction": "below",
                        "rolling_window": 10
                    },
                    {
                        "metric_name": "prediction_latency",
                        "warning_threshold": 5.0,
                        "critical_threshold": 10.0,
                        "direction": "above",
                        "rolling_window": 5
                    }
                ]

            thresholds = []
            for t in thresholds_data:
                thresholds.append(PerformanceThreshold(**t))

            return thresholds

        except Exception as e:
            logger.error(f"Error loading thresholds: {e}")
            return []

    def _load_alert_rules(self) -> List[AlertRule]:
        """Load alert rules from configuration."""
        try:
            config = get_guardrail_config()
            rules_data = config.get('alert_rules', [])
            if not rules_data:
                # Default alert rules
                rules_data = [
                    {
                        "name": "high_error_rate",
                        "condition": "mae > 250",
                        "severity": "critical",
                        "cooldown_minutes": 30,
                        "actions": ["log", "email", "rollback"]
                    },
                    {
                        "name": "low_accuracy",
                        "condition": "directional_accuracy < 0.4",
                        "severity": "warning",
                        "cooldown_minutes": 60,
                        "actions": ["log", "retrain"]
                    },
                    {
                        "name": "system_degradation",
                        "condition": "performance_trend == 'degrading'",
                        "severity": "warning",
                        "cooldown_minutes": 120,
                        "actions": ["log", "investigate"]
                    }
                ]

            rules = []
            for r in rules_data:
                rules.append(AlertRule(**r))

            return rules

        except Exception as e:
            logger.error(f"Error loading alert rules: {e}")
            return []

    def start_monitoring(self, interval_seconds: int = 300):
        """
        Start continuous monitoring in background thread.

        Args:
            interval_seconds: Monitoring interval in seconds
        """
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"Started monitoring with {interval_seconds}s interval")

    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Stopped monitoring")

    def _monitoring_loop(self, interval_seconds: int):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self.check_system_health()
                time.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval_seconds)

    def check_system_health(self) -> Dict[str, Any]:
        """
        Perform comprehensive system health check.

        Returns:
            Health check results
        """
        current_time = datetime.now()

        # Gather current metrics
        current_metrics = self._gather_current_metrics()

        # Check thresholds
        threshold_violations = self._check_thresholds(current_metrics)

        # Evaluate alert rules
        alerts = self._evaluate_alert_rules(current_metrics)

        # Analyze trends
        trend_analysis = self._analyze_performance_trends()

        # Update history
        self.performance_history.append({
            'timestamp': current_time.isoformat(),
            'metrics': current_metrics,
            'violations': threshold_violations,
            'alerts': alerts
        })

        # Keep only recent history (last 100 entries)
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

        # Trigger alerts
        for alert in alerts:
            self._trigger_alert(alert)

        health_status = {
            'timestamp': current_time.isoformat(),
            'status': 'healthy' if not alerts else 'warning' if any(a['severity'] == 'warning' for a in alerts) else 'critical',
            'current_metrics': current_metrics,
            'threshold_violations': threshold_violations,
            'alerts': alerts,
            'trend_analysis': trend_analysis
        }

        logger.info(f"Health check completed: {health_status['status']}")
        return health_status

    def _gather_current_metrics(self) -> Dict[str, float]:
        """Gather current system performance metrics."""
        # This would integrate with actual system monitoring
        # For now, return mock data
        return {
            'mae': 175.5,
            'directional_accuracy': 0.52,
            'prediction_latency': 2.3,
            'memory_usage': 0.75,
            'cpu_usage': 0.45
        }

    def _check_thresholds(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check metrics against defined thresholds."""
        violations = []

        for threshold in self.thresholds:
            if threshold.metric_name not in metrics:
                continue

            value = metrics[threshold.metric_name]

            # Check if threshold is violated
            if threshold.direction == 'above' and value > threshold.critical_threshold:
                violations.append({
                    'metric': threshold.metric_name,
                    'value': value,
                    'threshold': threshold.critical_threshold,
                    'severity': 'critical',
                    'direction': threshold.direction
                })
            elif threshold.direction == 'above' and value > threshold.warning_threshold:
                violations.append({
                    'metric': threshold.metric_name,
                    'value': value,
                    'threshold': threshold.warning_threshold,
                    'severity': 'warning',
                    'direction': threshold.direction
                })
            elif threshold.direction == 'below' and value < threshold.critical_threshold:
                violations.append({
                    'metric': threshold.metric_name,
                    'value': value,
                    'threshold': threshold.critical_threshold,
                    'severity': 'critical',
                    'direction': threshold.direction
                })
            elif threshold.direction == 'below' and value < threshold.warning_threshold:
                violations.append({
                    'metric': threshold.metric_name,
                    'value': value,
                    'threshold': threshold.warning_threshold,
                    'severity': 'warning',
                    'direction': threshold.direction
                })

        return violations

    def _evaluate_alert_rules(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Evaluate alert rules against current metrics."""
        alerts = []

        for rule in self.alert_rules:
            # Check if alert is in cooldown
            if self._is_alert_in_cooldown(rule.name):
                continue

            # Evaluate condition (simplified - would need proper expression evaluation)
            if self._evaluate_condition(rule.condition, metrics):
                alerts.append({
                    'rule_name': rule.name,
                    'severity': rule.severity,
                    'condition': rule.condition,
                    'actions': rule.actions,
                    'timestamp': datetime.now().isoformat(),
                    'metrics': metrics
                })

        return alerts

    def _evaluate_condition(self, condition: str, metrics: Dict[str, float]) -> bool:
        """Evaluate alert condition (simplified implementation)."""
        # This is a simplified condition evaluator
        # In production, you'd want a proper expression parser

        try:
            if 'mae >' in condition:
                threshold = float(condition.split('>')[1].strip())
                return metrics.get('mae', 0) > threshold
            elif 'directional_accuracy <' in condition:
                threshold = float(condition.split('<')[1].strip())
                return metrics.get('directional_accuracy', 0) < threshold
            elif 'performance_trend ==' in condition:
                trend = condition.split("'")[1]
                return self._analyze_performance_trends().get('overall_trend') == trend
        except:
            pass

        return False

    def _is_alert_in_cooldown(self, rule_name: str) -> bool:
        """Check if alert rule is currently in cooldown period."""
        current_time = datetime.now()

        for alert in self.alert_history:
            if alert['rule_name'] == rule_name:
                alert_time = datetime.fromisoformat(alert['timestamp'])
                cooldown_end = alert_time + timedelta(minutes=alert.get('cooldown_minutes', 60))
                if current_time < cooldown_end:
                    return True

        return False

    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        if len(self.performance_history) < 5:
            return {'overall_trend': 'insufficient_data'}

        recent_metrics = self.performance_history[-10:]  # Last 10 checks

        # Calculate trends
        mae_values = [h['metrics'].get('mae', 0) for h in recent_metrics]
        acc_values = [h['metrics'].get('directional_accuracy', 0) for h in recent_metrics]

        mae_trend = 'improving' if mae_values[-1] < mae_values[0] else 'degrading'
        acc_trend = 'improving' if acc_values[-1] > acc_values[0] else 'degrading'

        overall_trend = 'stable'
        if mae_trend == 'degrading' or acc_trend == 'degrading':
            overall_trend = 'degrading'
        elif mae_trend == 'improving' and acc_trend == 'improving':
            overall_trend = 'improving'

        return {
            'overall_trend': overall_trend,
            'mae_trend': mae_trend,
            'accuracy_trend': acc_trend,
            'data_points': len(recent_metrics)
        }

    def _trigger_alert(self, alert: Dict[str, Any]):
        """Trigger alert actions."""
        # Add to alert history
        self.alert_history.append(alert)

        # Keep only recent alerts (last 50)
        if len(self.alert_history) > 50:
            self.alert_history = self.alert_history[-50:]

        # Execute alert actions
        for action in alert['actions']:
            self._execute_alert_action(action, alert)

        logger.warning(f"Alert triggered: {alert['rule_name']} ({alert['severity']})")

    def _execute_alert_action(self, action: str, alert: Dict[str, Any]):
        """Execute specific alert action."""
        if action == 'log':
            logger.warning(f"ALERT: {alert}")
        elif action == 'email':
            self._send_email_alert(alert)
        elif action == 'rollback':
            self._trigger_rollback(alert)
        elif action == 'retrain':
            self._trigger_retraining(alert)
        elif action == 'investigate':
            self._trigger_investigation(alert)

        # Call custom callbacks
        if action in self.alert_callbacks:
            try:
                self.alert_callbacks[action](alert)
            except Exception as e:
                logger.error(f"Error in alert callback {action}: {e}")

    def _send_email_alert(self, alert: Dict[str, Any]):
        """Send email alert (placeholder)."""
        logger.info(f"Would send email alert: {alert['rule_name']}")

    def _trigger_rollback(self, alert: Dict[str, Any]):
        """Trigger system rollback (placeholder)."""
        logger.info(f"Would trigger rollback due to: {alert['rule_name']}")

    def _trigger_retraining(self, alert: Dict[str, Any]):
        """Trigger model retraining (placeholder)."""
        logger.info(f"Would trigger retraining due to: {alert['rule_name']}")

    def _trigger_investigation(self, alert: Dict[str, Any]):
        """Trigger investigation process (placeholder)."""
        logger.info(f"Would trigger investigation due to: {alert['rule_name']}")

    def register_alert_callback(self, action: str, callback: Callable):
        """
        Register a callback function for specific alert actions.

        Args:
            action: Alert action name
            callback: Function to call when action is triggered
        """
        self.alert_callbacks[action] = callback
        logger.info(f"Registered callback for alert action: {action}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'monitoring_active': self.monitoring_active,
            'last_health_check': self.performance_history[-1] if self.performance_history else None,
            'active_alerts': len([a for a in self.alert_history if self._is_recent_alert(a)]),
            'total_alerts': len(self.alert_history),
            'performance_trend': self._analyze_performance_trends(),
            'thresholds': len(self.thresholds),
            'alert_rules': len(self.alert_rules)
        }

    def _is_recent_alert(self, alert: Dict[str, Any], hours: int = 24) -> bool:
        """Check if alert is within recent time window."""
        alert_time = datetime.fromisoformat(alert['timestamp'])
        return (datetime.now() - alert_time).total_seconds() < (hours * 3600)

    def export_monitoring_data(self, filepath: str):
        """Export monitoring data to file."""
        data = {
            'performance_history': self.performance_history,
            'alert_history': self.alert_history,
            'thresholds': [vars(t) for t in self.thresholds],
            'alert_rules': [vars(r) for r in self.alert_rules],
            'export_timestamp': datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported monitoring data to {filepath}")