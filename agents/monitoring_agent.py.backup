"""
Base Monitoring Agent

Provides basic monitoring functionality for the IB Forecast system.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class MonitoringAgent:
    """
    Base monitoring agent providing core monitoring functionality.

    This class serves as a foundation for specialized monitoring agents
    like drift detection, performance monitoring, etc.
    """

    def __init__(self, name: str = "MonitoringAgent"):
        """
        Initialize monitoring agent.

        Args:
            name: Name identifier for the agent
        """
        self.name = name
        self.start_time = datetime.now()
        self.metrics = {}
        self.alerts = []

        logger.info(f"Initialized {self.name}")

    def log_metric(self, key: str, value: Any, timestamp: Optional[datetime] = None):
        """
        Log a metric value.

        Args:
            key: Metric key
            value: Metric value
            timestamp: Optional timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()

        if key not in self.metrics:
            self.metrics[key] = []

        self.metrics[key].append({
            "value": value,
            "timestamp": timestamp
        })

        logger.debug(f"Logged metric {key}: {value}")

    def get_metric_history(self, key: str, limit: Optional[int] = None) -> list:
        """
        Get historical values for a metric.

        Args:
            key: Metric key
            limit: Maximum number of values to return

        Returns:
            List of metric values with timestamps
        """
        if key not in self.metrics:
            return []

        history = self.metrics[key]
        if limit:
            history = history[-limit:]

        return history

    def check_threshold(self, key: str, threshold: float, operator: str = "gt") -> bool:
        """
        Check if a metric exceeds a threshold.

        Args:
            key: Metric key
            threshold: Threshold value
            operator: Comparison operator ("gt", "lt", "eq")

        Returns:
            True if threshold condition is met
        """
        history = self.get_metric_history(key, limit=1)
        if not history:
            return False

        value = history[0]["value"]

        if operator == "gt":
            return value > threshold
        elif operator == "lt":
            return value < threshold
        elif operator == "eq":
            return value == threshold
        else:
            logger.warning(f"Unknown operator: {operator}")
            return False

    def add_alert(self, message: str, severity: str = "info"):
        """
        Add an alert message.

        Args:
            message: Alert message
            severity: Alert severity ("info", "warning", "error", "critical")
        """
        alert = {
            "message": message,
            "severity": severity,
            "timestamp": datetime.now()
        }

        self.alerts.append(alert)
        logger.log(getattr(logging, severity.upper(), logging.INFO), message)

    def get_alerts(self, severity: Optional[str] = None, limit: Optional[int] = None) -> list:
        """
        Get alerts, optionally filtered by severity.

        Args:
            severity: Filter by severity level
            limit: Maximum number of alerts to return

        Returns:
            List of alerts
        """
        alerts = self.alerts

        if severity:
            alerts = [a for a in alerts if a["severity"] == severity]

        if limit:
            alerts = alerts[-limit:]

        return alerts

    def get_status(self) -> Dict[str, Any]:
        """
        Get current agent status.

        Returns:
            Dictionary with agent status information
        """
        return {
            "name": self.name,
            "uptime": str(datetime.now() - self.start_time),
            "metrics_count": len(self.metrics),
            "alerts_count": len(self.alerts),
            "last_alert": self.alerts[-1] if self.alerts else None
        }

# Convenience functions
def create_monitoring_agent(name: str = "MonitoringAgent") -> MonitoringAgent:
    """Create and return a monitoring agent instance."""
    return MonitoringAgent(name)


import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from graphs.state import GraphState

class MonitoringDriftAgent:
    """
    Monitoring/Drift Agent that compares predictions vs. realized performance
    and triggers retrain loops when drift is detected.
    """

    def __init__(self, drift_threshold: float = 0.1, performance_window: int = 30):
        self.drift_threshold = drift_threshold  # Max acceptable prediction error
        self.performance_window = performance_window  # Days to evaluate performance
        self.performance_history = {}  # Store historical performance

    def monitor_performance(self, predictions: Dict[str, Any],
                          raw_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Monitor prediction performance and detect drift.
        """
        drift_metrics = {}
        retrain_trigger = False

        for symbol, symbol_predictions in predictions.items():
            if symbol not in raw_data:
                continue

            print(f"Monitoring performance for {symbol}...")

            # Get realized prices for evaluation period
            raw_df = raw_data[symbol]
            current_date = raw_df.index[-1] if hasattr(raw_df.index, '__getitem__') else datetime.now()

            # Evaluate each prediction timeframe
            symbol_metrics = {}
            symbol_drift_detected = False

            for timeframe, pred_data in symbol_predictions.get('ranked_predictions', {}).items():
                try:
                    metrics = self._evaluate_prediction(
                        pred_data, raw_df, timeframe, current_date
                    )
                    if metrics:
                        symbol_metrics[timeframe] = metrics

                        # Check for drift
                        if metrics['mae'] > self.drift_threshold:
                            symbol_drift_detected = True
                            print(f"⚠️ Drift detected for {symbol} {timeframe}: MAE={metrics['mae']:.4f}")

                except Exception as e:
                    print(f"Error evaluating {symbol} {timeframe}: {e}")

            # Overall symbol assessment
            if symbol_metrics:
                avg_mae = np.mean([m['mae'] for m in symbol_metrics.values()])
                drift_metrics[symbol] = {
                    'timeframe_metrics': symbol_metrics,
                    'overall_mae': avg_mae,
                    'drift_detected': symbol_drift_detected,
                    'evaluation_date': current_date
                }

                # Update performance history
                self._update_performance_history(symbol, avg_mae, symbol_drift_detected)

                # Check if retraining is needed
                if symbol_drift_detected or self._should_retrain(symbol):
                    retrain_trigger = True
                    print(f"🔄 Retrain trigger activated for {symbol}")

        return {
            'drift_metrics': drift_metrics,
            'retrain_trigger': retrain_trigger,
            'overall_drift_rate': self._calculate_overall_drift_rate(drift_metrics),
            'performance_summary': self._generate_performance_summary(drift_metrics)
        }

    def _evaluate_prediction(self, pred_data: Dict[str, Any],
                           raw_df: pd.DataFrame,
                           timeframe: str,
                           current_date: datetime) -> Optional[Dict[str, Any]]:
        """Evaluate a single prediction against realized prices"""

        # Parse timeframe (e.g., "5d" -> 5 days)
        days_ahead = int(timeframe.replace('d', ''))

        # Get prediction timestamp
        pred_timestamp = pred_data.get('timestamp', current_date)
        if isinstance(pred_timestamp, str):
            pred_timestamp = pd.to_datetime(pred_timestamp)

        # Calculate target date
        target_date = pred_timestamp + timedelta(days=days_ahead)

        # Find closest actual price to target date with validation
        realized_price = None
        
        if len(raw_df) == 0:
            print(f"⚠️  Warning: Empty raw_df for price lookup")
            return None
            
        if hasattr(raw_df.index, 'get_loc'):
            try:
                # If index is datetime
                target_idx = raw_df.index.get_loc(target_date, method='nearest')
                if 0 <= target_idx < len(raw_df):
                    realized_price = raw_df.iloc[target_idx]['close']
                else:
                    realized_price = raw_df['close'].iloc[-1]
            except Exception as e:
                # Fallback: use most recent price
                print(f"⚠️  Price lookup failed, using latest price: {e}")
                realized_price = raw_df['close'].iloc[-1]
        else:
            if len(raw_df) > 0:
                realized_price = raw_df['close'].iloc[-1]
        
        if realized_price is None or realized_price == 0:
            print(f"⚠️  Warning: Invalid realized price")
            return None

        predicted_price = pred_data['prediction']
        error = abs(predicted_price - realized_price)
        mae = error / realized_price if realized_price != 0 else 0  # Mean Absolute Percentage Error

        return {
            'predicted_price': predicted_price,
            'realized_price': realized_price,
            'error': error,
            'mae': mae,
            'direction_correct': ((predicted_price > pred_data.get('current_price', realized_price)) ==
                                (realized_price > pred_data.get('current_price', realized_price))),
            'target_date': target_date,
            'days_ahead': days_ahead
        }

    def _update_performance_history(self, symbol: str, mae: float, drift_detected: bool):
        """Update historical performance tracking"""
        if symbol not in self.performance_history:
            self.performance_history[symbol] = []

        self.performance_history[symbol].append({
            'date': datetime.now(),
            'mae': mae,
            'drift_detected': drift_detected
        })

        # Keep only recent history
        if len(self.performance_history[symbol]) > self.performance_window:
            self.performance_history[symbol] = self.performance_history[symbol][-self.performance_window:]

    def _should_retrain(self, symbol: str) -> bool:
        """Determine if retraining is needed based on performance history"""
        if symbol not in self.performance_history:
            return False

        history = self.performance_history[symbol]

        # Check recent drift frequency
        recent_drifts = [h for h in history[-7:] if h['drift_detected']]  # Last 7 evaluations
        if len(recent_drifts) >= 3:  # 3 or more drifts in last week
            return True

        # Check if performance is consistently poor
        recent_maes = [h['mae'] for h in history[-10:]]
        if recent_maes and np.mean(recent_maes) > self.drift_threshold * 1.5:
            return True

        # Check for performance degradation trend
        if len(history) >= 14:
            old_maes = [h['mae'] for h in history[-14:-7]]
            new_maes = [h['mae'] for h in history[-7:]]

            if old_maes and new_maes:
                old_avg = np.mean(old_maes)
                new_avg = np.mean(new_maes)

                # If performance degraded by more than 50%
                if new_avg > old_avg * 1.5:
                    return True

        return False

    def _calculate_overall_drift_rate(self, drift_metrics: Dict[str, Any]) -> float:
        """Calculate overall drift rate across all symbols"""
        if not drift_metrics:
            return 0.0

        total_symbols = len(drift_metrics)
        drifting_symbols = sum(1 for m in drift_metrics.values() if m['drift_detected'])

        return drifting_symbols / total_symbols if total_symbols > 0 else 0.0

    def _generate_performance_summary(self, drift_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics"""
        if not drift_metrics:
            return {}

        maes = [m['overall_mae'] for m in drift_metrics.values()]
        drifts = [m['drift_detected'] for m in drift_metrics.values()]

        return {
            'avg_mae': np.mean(maes),
            'median_mae': np.median(maes),
            'max_mae': np.max(maes),
            'drift_rate': np.mean(drifts),
            'total_symbols': len(drift_metrics),
            'drifting_symbols': sum(drifts)
        }

def run_monitoring_node(state: GraphState) -> GraphState:
    """
    Node function for monitoring and drift detection.
    """
    print("--- MONITORING PERFORMANCE ---")
    predictions = state.get('predictions', {})
    raw_data = state.get('raw_data', {})

    if not predictions or not raw_data:
        print("⏩ Skipping monitoring: No predictions or raw data.")
        state['drift_metrics'] = {}
        state['retrain_trigger'] = False
        return state

    print(f"Monitoring performance for {len(predictions)} symbols...")

    agent = MonitoringDriftAgent()
    monitoring_results = agent.monitor_performance(predictions, raw_data)

    state['drift_metrics'] = monitoring_results['drift_metrics']
    state['retrain_trigger'] = monitoring_results['retrain_trigger']
    state['metrics'] = monitoring_results

    print(f"✅ Monitoring complete. Retrain trigger: {state['retrain_trigger']}")

    return state