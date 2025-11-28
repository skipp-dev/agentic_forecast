"""
Drift Detection Module for Financial Time Series

This module implements various drift detection techniques to identify concept drift
in financial time series data, which is crucial for maintaining model performance
in changing market conditions.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Union
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)

class DriftDetector:
    """
    Comprehensive drift detection system for financial time series.

    Implements multiple statistical tests and monitoring techniques:
    - Kolmogorov-Smirnov test for distribution drift
    - CUSUM control charts for mean change detection
    - Rolling window analysis for gradual drift
    - Prediction error monitoring
    """

    def __init__(self,
                 window_size: int = 50,
                 significance_level: float = 0.05,
                 cusum_threshold: float = 5.0):
        """
        Initialize drift detector.

        Args:
            window_size: Size of rolling window for analysis
            significance_level: Statistical significance level (alpha)
            cusum_threshold: Threshold for CUSUM drift detection
        """
        self.window_size = window_size
        self.significance_level = significance_level
        self.cusum_threshold = cusum_threshold

        # Historical data for drift detection
        self.historical_data = []
        self.prediction_errors = []
        self.baseline_distribution = None
        self.baseline_mean = None
        self.baseline_std = None

        # CUSUM tracking
        self.cusum_positive = 0.0
        self.cusum_negative = 0.0
        self.cusum_history = []

        logger.info(f"Initialized DriftDetector with window_size={window_size}, "
                   f"significance_level={significance_level}, cusum_threshold={cusum_threshold}")

    def establish_baseline(self, data: Union[np.ndarray, pd.Series, List],
                          prediction_errors: Optional[Union[np.ndarray, List]] = None) -> None:
        """
        Establish baseline distribution from stable historical data.

        Args:
            data: Historical data to establish baseline
            prediction_errors: Optional prediction errors for baseline
        """
        data_array = np.asarray(data)

        # Store baseline statistics
        self.baseline_distribution = data_array.copy()
        self.baseline_mean = np.mean(data_array)
        self.baseline_std = np.std(data_array)

        # Initialize CUSUM
        self.cusum_positive = 0.0
        self.cusum_negative = 0.0
        self.cusum_history = []

        if prediction_errors is not None:
            self.prediction_errors = list(prediction_errors)

        logger.info(f"Established baseline with {len(data_array)} samples, "
                   f"mean={self.baseline_mean:.4f}, std={self.baseline_std:.4f}")

    def detect_distribution_drift(self, new_data: Union[np.ndarray, pd.Series, List],
                                 test_type: str = 'ks') -> Dict[str, Union[bool, float]]:
        """
        Detect distribution drift using statistical tests.

        Args:
            new_data: New data to test for drift
            test_type: Type of test ('ks' for Kolmogorov-Smirnov, 'anderson' for Anderson-Darling)

        Returns:
            Dictionary with drift detection results
        """
        if self.baseline_distribution is None:
            raise ValueError("Baseline not established. Call establish_baseline() first.")

        new_data_array = np.asarray(new_data)

        if len(new_data_array) < 10:
            return {
                'drift_detected': False,
                'p_value': 1.0,
                'statistic': 0.0,
                'confidence': 'insufficient_data'
            }

        try:
            if test_type == 'ks':
                # Kolmogorov-Smirnov test
                statistic, p_value = stats.ks_2samp(self.baseline_distribution, new_data_array)
                test_name = 'Kolmogorov-Smirnov'

            elif test_type == 'anderson':
                # Anderson-Darling test (one-sample version)
                try:
                    result = stats.anderson_ksamp([self.baseline_distribution, new_data_array])
                    statistic = result.statistic
                    p_value = 1.0  # Anderson gives critical values, not p-values
                    test_name = 'Anderson-Darling'
                except Exception:
                    # Fallback to KS test
                    statistic, p_value = stats.ks_2samp(self.baseline_distribution, new_data_array)
                    test_name = 'Kolmogorov-Smirnov (fallback)'

            else:
                raise ValueError(f"Unknown test type: {test_type}")

            drift_detected = p_value < self.significance_level

            confidence = 'high' if p_value < 0.01 else 'medium' if p_value < 0.05 else 'low'

            return {
                'drift_detected': drift_detected,
                'p_value': p_value,
                'statistic': statistic,
                'test_type': test_name,
                'confidence': confidence
            }

        except Exception as e:
            logger.warning(f"Error in distribution drift detection: {e}")
            return {
                'drift_detected': False,
                'p_value': 1.0,
                'statistic': 0.0,
                'error': str(e),
                'confidence': 'error'
            }

    def detect_mean_drift_cusum(self, new_value: float) -> Dict[str, Union[bool, float]]:
        """
        Detect mean drift using CUSUM control chart.

        Args:
            new_value: New observation to test

        Returns:
            Dictionary with CUSUM drift detection results
        """
        if self.baseline_mean is None or self.baseline_std is None:
            raise ValueError("Baseline not established. Call establish_baseline() first.")

        # Standardize the value
        standardized_value = (new_value - self.baseline_mean) / self.baseline_std

        # Update CUSUM statistics
        self.cusum_positive = max(0, self.cusum_positive + standardized_value - 0.5)
        self.cusum_negative = max(0, self.cusum_negative - standardized_value - 0.5)

        # Check for drift
        drift_positive = self.cusum_positive > self.cusum_threshold
        drift_negative = self.cusum_negative > self.cusum_threshold

        # Store history
        self.cusum_history.append({
            'value': new_value,
            'standardized': standardized_value,
            'cusum_pos': self.cusum_positive,
            'cusum_neg': self.cusum_negative
        })

        return {
            'drift_detected': drift_positive or drift_negative,
            'drift_direction': 'positive' if drift_positive else 'negative' if drift_negative else 'none',
            'cusum_positive': self.cusum_positive,
            'cusum_negative': self.cusum_negative,
            'threshold': self.cusum_threshold
        }

    def detect_error_drift(self, predictions: np.ndarray, actuals: np.ndarray,
                          window_size: Optional[int] = None) -> Dict[str, Union[bool, float]]:
        """
        Detect drift in prediction errors.

        Args:
            predictions: Model predictions
            actuals: Actual values
            window_size: Window size for rolling analysis

        Returns:
            Dictionary with error drift detection results
        """
        if window_size is None:
            window_size = self.window_size

        errors = actuals - predictions
        current_mae = mean_absolute_error(actuals, predictions)

        # Compare with historical errors
        if len(self.prediction_errors) >= window_size:
            historical_mae = np.mean(np.abs(self.prediction_errors[-window_size:]))

            # Simple drift detection based on error increase
            error_increase = (current_mae - historical_mae) / historical_mae
            drift_detected = error_increase > 0.5  # 50% increase in MAE

            return {
                'drift_detected': drift_detected,
                'current_mae': current_mae,
                'historical_mae': historical_mae,
                'error_increase': error_increase,
                'threshold': 0.5
            }
        else:
            # Not enough historical data
            self.prediction_errors.extend(errors)
            return {
                'drift_detected': False,
                'current_mae': current_mae,
                'historical_mae': None,
                'error_increase': 0.0,
                'status': 'building_baseline'
            }

    def rolling_window_analysis(self, data: Union[np.ndarray, pd.Series, List],
                               window_size: Optional[int] = None) -> Dict[str, Union[List, bool]]:
        """
        Perform rolling window analysis for gradual drift detection.

        Args:
            data: Time series data
            window_size: Size of rolling window

        Returns:
            Dictionary with rolling analysis results
        """
        if window_size is None:
            window_size = self.window_size

        data_array = np.asarray(data)

        if len(data_array) < window_size * 2:
            return {'drift_detected': False, 'status': 'insufficient_data'}

        # Calculate rolling statistics
        rolling_means = []
        rolling_stds = []

        for i in range(window_size, len(data_array) - window_size + 1):
            window = data_array[i-window_size:i]
            rolling_means.append(np.mean(window))
            rolling_stds.append(np.std(window))

        # Detect trend in rolling means (gradual drift)
        if len(rolling_means) > 10:
            # Linear regression on rolling means
            x = np.arange(len(rolling_means))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, rolling_means)

            # Significant trend indicates gradual drift
            trend_drift = abs(r_value) > 0.7 and p_value < self.significance_level

            return {
                'drift_detected': trend_drift,
                'trend_slope': slope,
                'trend_r_squared': r_value**2,
                'trend_p_value': p_value,
                'rolling_means': rolling_means,
                'rolling_stds': rolling_stds,
                'drift_type': 'gradual' if trend_drift else 'none'
            }

        return {'drift_detected': False, 'status': 'insufficient_windows'}

    def comprehensive_drift_check(self, new_data: Union[np.ndarray, pd.Series, List],
                                 predictions: Optional[np.ndarray] = None,
                                 actuals: Optional[np.ndarray] = None) -> Dict[str, Dict]:
        """
        Perform comprehensive drift detection using all available methods.

        Args:
            new_data: New data to analyze
            predictions: Optional model predictions
            actuals: Optional actual values

        Returns:
            Dictionary with results from all drift detection methods
        """
        results = {}

        # Distribution drift
        results['distribution_drift'] = self.detect_distribution_drift(new_data)

        # Rolling window analysis
        results['rolling_window'] = self.rolling_window_analysis(new_data)

        # CUSUM for recent values
        if len(new_data) > 0:
            recent_cusum_results = []
            recent_values = new_data[-min(20, len(new_data)):]  # Last 20 values
            for value in recent_values:
                cusum_result = self.detect_mean_drift_cusum(value)
                recent_cusum_results.append(cusum_result)

            # Check if any recent CUSUM detected drift
            cusum_drift = any(r['drift_detected'] for r in recent_cusum_results)
            results['cusum_drift'] = {
                'drift_detected': cusum_drift,
                'recent_results': recent_cusum_results[-5:],  # Last 5 results
                'threshold': self.cusum_threshold
            }
        else:
            results['cusum_drift'] = {'drift_detected': False, 'status': 'no_data'}

        # Error drift (if predictions and actuals provided)
        if predictions is not None and actuals is not None:
            results['error_drift'] = self.detect_error_drift(predictions, actuals)
        else:
            results['error_drift'] = {'drift_detected': False, 'status': 'no_predictions'}

        # Overall assessment
        any_drift = any(method_result.get('drift_detected', False)
                       for method_result in results.values()
                       if isinstance(method_result, dict))

        results['overall_assessment'] = {
            'drift_detected': any_drift,
            'methods_used': list(results.keys()),
            'recommendation': 'retrain_model' if any_drift else 'continue_monitoring'
        }

        return results

    def plot_drift_analysis(self, save_path: str = "drift_analysis.png") -> None:
        """
        Plot comprehensive drift analysis visualization.

        Args:
            save_path: Path to save the plot
        """
        if not self.cusum_history:
            logger.warning("No CUSUM history available for plotting")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: CUSUM chart
        history_df = pd.DataFrame(self.cusum_history)
        ax1.plot(history_df.index, history_df['cusum_pos'], 'r-', label='CUSUM+', linewidth=2)
        ax1.plot(history_df.index, history_df['cusum_neg'], 'b-', label='CUSUM-', linewidth=2)
        ax1.axhline(y=self.cusum_threshold, color='k', linestyle='--', alpha=0.7, label='Threshold')
        ax1.set_title('CUSUM Control Chart')
        ax1.set_xlabel('Observation')
        ax1.set_ylabel('CUSUM Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Standardized values
        ax2.plot(history_df.index, history_df['standardized'], 'g-', linewidth=1)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        ax2.set_title('Standardized Values')
        ax2.set_xlabel('Observation')
        ax2.set_ylabel('Standardized Value')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Original values
        ax3.plot(history_df.index, history_df['value'], 'b-', linewidth=1)
        ax3.axhline(y=self.baseline_mean, color='r', linestyle='--', alpha=0.7, label='Baseline Mean')
        ax3.set_title('Original Values with Baseline Mean')
        ax3.set_xlabel('Observation')
        ax3.set_ylabel('Value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Distribution comparison (if available)
        if self.baseline_distribution is not None and len(self.baseline_distribution) > 10:
            ax4.hist(self.baseline_distribution, bins=30, alpha=0.7, label='Baseline', density=True)
            if len(history_df) > 10:
                ax4.hist(history_df['value'], bins=20, alpha=0.7, label='Recent', density=True)
            ax4.set_title('Distribution Comparison')
            ax4.set_xlabel('Value')
            ax4.set_ylabel('Density')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Drift analysis plot saved to {save_path}")
        plt.close()

    def reset(self) -> None:
        """Reset the drift detector state."""
        self.historical_data = []
        self.prediction_errors = []
        self.baseline_distribution = None
        self.baseline_mean = None
        self.baseline_std = None
        self.cusum_positive = 0.0
        self.cusum_negative = 0.0
        self.cusum_history = []
        logger.info("Drift detector reset")


class FinancialDriftMonitor:
    """
    Specialized drift monitor for financial forecasting systems.

    Integrates with forecasting models to monitor for concept drift
    in financial markets and trigger retraining when necessary.
    """

    def __init__(self,
                 model_name: str = "financial_forecaster",
                 drift_detector: Optional[DriftDetector] = None):
        """
        Initialize financial drift monitor.

        Args:
            model_name: Name of the model being monitored
            drift_detector: Optional custom drift detector
        """
        self.model_name = model_name
        self.drift_detector = drift_detector or DriftDetector()
        self.monitoring_history = []
        self.drift_alerts = []

        logger.info(f"Initialized FinancialDriftMonitor for {model_name}")

    def monitor_predictions(self, predictions: np.ndarray, actuals: np.ndarray,
                           timestamps: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None) -> Dict:
        """
        Monitor model predictions for drift.

        Args:
            predictions: Model predictions
            actuals: Actual values
            timestamps: Optional timestamps for the predictions

        Returns:
            Monitoring results
        """
        mae = mean_absolute_error(actuals, predictions)

        # Perform comprehensive drift check
        drift_results = self.drift_detector.comprehensive_drift_check(
            actuals, predictions, actuals
        )

        # Record monitoring event
        monitoring_event = {
            'timestamp': pd.Timestamp.now() if timestamps is None else timestamps[-1] if len(timestamps) > 0 else pd.Timestamp.now(),
            'mae': mae,
            'n_samples': len(actuals),
            'drift_results': drift_results,
            'predictions_mean': np.mean(predictions),
            'actuals_mean': np.mean(actuals)
        }

        self.monitoring_history.append(monitoring_event)

        # Check for drift alerts
        if drift_results['overall_assessment']['drift_detected']:
            alert = {
                'timestamp': monitoring_event['timestamp'],
                'alert_type': 'drift_detected',
                'severity': 'high',
                'message': f"Concept drift detected in {self.model_name}",
                'drift_methods': [k for k, v in drift_results.items()
                                if isinstance(v, dict) and v.get('drift_detected', False)],
                'mae': mae
            }
            self.drift_alerts.append(alert)
            logger.warning(f"🚨 DRIFT ALERT: {alert['message']} - Methods: {alert['drift_methods']}")

        return {
            'current_mae': mae,
            'drift_detected': drift_results['overall_assessment']['drift_detected'],
            'drift_details': drift_results,
            'alerts': len(self.drift_alerts)
        }

    def get_monitoring_summary(self) -> Dict:
        """Get summary of monitoring history and alerts."""
        if not self.monitoring_history:
            return {'status': 'no_monitoring_data'}

        recent_history = self.monitoring_history[-10:]  # Last 10 monitoring events

        return {
            'total_monitoring_events': len(self.monitoring_history),
            'total_alerts': len(self.drift_alerts),
            'recent_mae_trend': [h['mae'] for h in recent_history],
            'drift_detected_count': sum(1 for h in self.monitoring_history
                                      if h['drift_results']['overall_assessment']['drift_detected']),
            'latest_event': self.monitoring_history[-1] if self.monitoring_history else None,
            'latest_alert': self.drift_alerts[-1] if self.drift_alerts else None
        }

    def plot_monitoring_history(self, save_path: str = "monitoring_history.png") -> None:
        """
        Plot monitoring history and drift alerts.

        Args:
            save_path: Path to save the plot
        """
        if not self.monitoring_history:
            logger.warning("No monitoring history available for plotting")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Plot MAE over time
        timestamps = [h['timestamp'] for h in self.monitoring_history]
        maes = [h['mae'] for h in self.monitoring_history]

        ax1.plot(timestamps, maes, 'b-', linewidth=2, label='MAE')
        ax1.set_title(f'{self.model_name} - Monitoring History')
        ax1.set_ylabel('Mean Absolute Error')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)

        # Mark drift alerts
        alert_timestamps = [a['timestamp'] for a in self.drift_alerts]
        alert_maes = []
        for alert_ts in alert_timestamps:
            # Find corresponding MAE
            for h in self.monitoring_history:
                if h['timestamp'] == alert_ts:
                    alert_maes.append(h['mae'])
                    break

        if alert_timestamps:
            ax1.scatter(alert_timestamps, alert_maes, color='red', s=100,
                       marker='x', linewidth=3, label='Drift Alerts')

        ax1.legend()

        # Plot drift detection methods
        method_counts = {}
        for h in self.monitoring_history:
            drift_results = h['drift_results']
            for method, result in drift_results.items():
                if isinstance(result, dict) and result.get('drift_detected', False):
                    method_counts[method] = method_counts.get(method, 0) + 1

        if method_counts:
            methods = list(method_counts.keys())
            counts = list(method_counts.values())

            ax2.bar(methods, counts, color='orange', alpha=0.7)
            ax2.set_title('Drift Detection Methods Triggered')
            ax2.set_ylabel('Count')
            ax2.tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for i, v in enumerate(counts):
                ax2.text(i, v + 0.1, str(v), ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Monitoring history plot saved to {save_path}")
        plt.close()