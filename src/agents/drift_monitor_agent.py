"""
Enhanced Drift Monitor Agent

Extends the existing monitoring agent with advanced drift detection:
- Performance drift monitoring
- Data drift detection
- Spectral drift analysis using cuFFT
- Multi-dimensional drift assessment
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Add paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.agents.monitoring_agent import MonitoringAgent
from src.gpu_services import get_gpu_services
from src.data_pipeline import DataPipeline

logger = logging.getLogger(__name__)

class DriftMonitorAgent(MonitoringAgent):
    """
    Enhanced drift monitoring with spectral analysis and multi-dimensional detection.

    Extends the existing MonitoringAgent with:
    - Performance drift (MAPE, RMSE, Sharpe ratio)
    - Data drift (distribution shifts, statistical tests)
    - Spectral drift (frequency domain changes using cuFFT)
    - Regime change detection
    """

    def __init__(self, gpu_services=None, data_pipeline=None):
        """
        Initialize enhanced drift monitor agent.

        Args:
            gpu_services: GPU services instance for spectral analysis
            data_pipeline: Data pipeline instance
        """
        super().__init__()
        self.gpu_services = gpu_services or get_gpu_services()
        self.data_pipeline = data_pipeline or DataPipeline()

        # Drift detection thresholds
        self.thresholds = {
            'performance_drift': 0.15,  # 15% degradation in metrics
            'data_drift': 0.20,         # 20% distribution shift
            'spectral_drift': 0.25,     # 25% spectral change
            'regime_change': 0.30       # 30% combined drift
        }

        # Historical data for drift detection
        self.performance_history = {}
        self.data_distribution_history = {}
        self.spectral_history = {}

        # Monitoring windows
        self.performance_window_days = 30
        self.drift_detection_window = 100  # Data points for comparison

        logger.info("Enhanced Drift Monitor Agent initialized")

    def comprehensive_drift_check(self, symbol: str) -> Dict[str, Any]:
        """
        Perform comprehensive drift analysis across multiple dimensions.

        Args:
            symbol: Stock symbol to monitor

        Returns:
            Comprehensive drift analysis results
        """
        logger.info(f"Running comprehensive drift check for {symbol}")

        results = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'performance_drift': self._check_performance_drift(symbol),
            'data_drift': self._check_data_drift(symbol),
            'spectral_drift': self._check_spectral_drift(symbol),
            'regime_change': False,
            'overall_drift_score': 0.0,
            'recommendations': []
        }

        # Calculate overall drift score
        drift_scores = [
            results['performance_drift']['drift_score'],
            results['data_drift']['drift_score'],
            results['spectral_drift']['drift_score']
        ]

        results['overall_drift_score'] = np.mean(drift_scores)
        results['regime_change'] = results['overall_drift_score'] > self.thresholds['regime_change']

        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results)

        logger.info(f"Drift analysis complete for {symbol}: score={results['overall_drift_score']:.3f}")

        return results

    def _check_performance_drift(self, symbol: str) -> Dict[str, Any]:
        """
        Check for performance drift in model predictions.

        Args:
            symbol: Stock symbol

        Returns:
            Performance drift analysis
        """
        try:
            # Get recent predictions and actuals (mock data for now)
            recent_predictions = self._get_recent_predictions(symbol)
            recent_actuals = self._get_recent_actuals(symbol)

            if len(recent_predictions) < 10 or len(recent_actuals) < 10:
                return {
                    'drift_detected': False,
                    'drift_score': 0.0,
                    'message': 'Insufficient data for performance drift analysis'
                }

            # Calculate current metrics
            current_mae = mean_absolute_error(recent_actuals, recent_predictions)
            current_rmse = np.sqrt(mean_squared_error(recent_actuals, recent_predictions))

            # Get historical baseline
            baseline_metrics = self._get_performance_baseline(symbol)
            if not baseline_metrics:
                # Store current as baseline
                self.performance_history[symbol] = {
                    'mae': current_mae,
                    'rmse': current_rmse,
                    'timestamp': datetime.now()
                }
                return {
                    'drift_detected': False,
                    'drift_score': 0.0,
                    'message': 'Baseline established'
                }

            # Calculate drift
            mae_drift = abs(current_mae - baseline_metrics['mae']) / baseline_metrics['mae']
            rmse_drift = abs(current_rmse - baseline_metrics['rmse']) / baseline_metrics['rmse']

            drift_score = (mae_drift + rmse_drift) / 2
            drift_detected = drift_score > self.thresholds['performance_drift']

            return {
                'drift_detected': drift_detected,
                'drift_score': drift_score,
                'current_mae': current_mae,
                'current_rmse': current_rmse,
                'baseline_mae': baseline_metrics['mae'],
                'baseline_rmse': baseline_metrics['rmse'],
                'mae_drift_pct': mae_drift * 100,
                'rmse_drift_pct': rmse_drift * 100
            }

        except Exception as e:
            logger.error(f"Performance drift check failed for {symbol}: {e}")
            return {
                'drift_detected': False,
                'drift_score': 0.0,
                'error': str(e)
            }

    def _check_data_drift(self, symbol: str) -> Dict[str, Any]:
        """
        Check for data distribution drift using statistical tests.

        Args:
            symbol: Stock symbol

        Returns:
            Data drift analysis
        """
        try:
            # Get current and historical data
            current_data = self._get_current_market_data(symbol)
            historical_data = self._get_historical_market_data(symbol)

            if len(current_data) < 50 or len(historical_data) < 50:
                return {
                    'drift_detected': False,
                    'drift_score': 0.0,
                    'message': 'Insufficient data for statistical tests'
                }

            # Statistical tests for drift
            drift_tests = {}

            # Kolmogorov-Smirnov test for distribution differences
            for column in ['close', 'volume', 'returns']:
                if column in current_data.columns and column in historical_data.columns:
                    current_vals = current_data[column].dropna().values
                    historical_vals = historical_data[column].dropna().values

                    try:
                        ks_stat, ks_p_value = stats.ks_2samp(current_vals, historical_vals)
                        drift_tests[column] = {
                            'ks_statistic': ks_stat,
                            'p_value': ks_p_value,
                            'drift_detected': ks_p_value < 0.05  # 5% significance
                        }
                    except Exception as e:
                        logger.warning(f"KS test failed for {column}: {e}")

            # Population Stability Index (PSI)
            psi_scores = {}
            for column in ['close', 'volume']:
                if column in current_data.columns and column in historical_data.columns:
                    psi_scores[column] = self._calculate_psi(
                        current_data[column], historical_data[column]
                    )

            # Overall drift score
            ks_drifts = [test['drift_detected'] for test in drift_tests.values()]
            psi_drifts = [psi > 0.1 for psi in psi_scores.values()]  # PSI > 0.1 indicates drift

            drift_score = (np.mean(ks_drifts) + np.mean(psi_drifts)) / 2
            drift_detected = drift_score > self.thresholds['data_drift']

            return {
                'drift_detected': drift_detected,
                'drift_score': drift_score,
                'ks_tests': drift_tests,
                'psi_scores': psi_scores,
                'significant_drifts': sum(ks_drifts) + sum(psi_drifts)
            }

        except Exception as e:
            logger.error(f"Data drift check failed for {symbol}: {e}")
            return {
                'drift_detected': False,
                'drift_score': 0.0,
                'error': str(e)
            }

    def _check_spectral_drift(self, symbol: str) -> Dict[str, Any]:
        """
        Check for spectral drift using cuFFT-based frequency analysis.

        Args:
            symbol: Stock symbol

        Returns:
            Spectral drift analysis
        """
        try:
            if not self.gpu_services:
                return {
                    'drift_detected': False,
                    'drift_score': 0.0,
                    'message': 'GPU services not available for spectral analysis'
                }

            # Get price data for spectral analysis
            price_data = self._get_price_data_for_spectral_analysis(symbol)

            if len(price_data) < 100:
                return {
                    'drift_detected': False,
                    'drift_score': 0.0,
                    'message': 'Insufficient data for spectral analysis'
                }

            # Extract current spectral features
            current_spectra = self.gpu_services.spectral_service.extract_spectral_features(price_data)

            # Get historical spectral features
            historical_spectra = self.spectral_history.get(symbol, [])

            if len(historical_spectra) < 5:
                # Store current as baseline
                self.spectral_history[symbol] = [current_spectra]
                return {
                    'drift_detected': False,
                    'drift_score': 0.0,
                    'message': 'Spectral baseline established'
                }

            # Calculate spectral drift
            drift_scores = {}
            for feature in current_spectra.keys():
                if feature in historical_spectra[0]:
                    # Average historical values
                    hist_avg = np.mean([spec[feature] for spec in historical_spectra])

                    if hist_avg != 0:
                        drift_scores[feature] = abs(current_spectra[feature] - hist_avg) / abs(hist_avg)
                    else:
                        drift_scores[feature] = abs(current_spectra[feature])

            # Overall spectral drift score
            spectral_drift_score = np.mean(list(drift_scores.values()))
            drift_detected = spectral_drift_score > self.thresholds['spectral_drift']

            # Update history
            self.spectral_history[symbol].append(current_spectra)
            if len(self.spectral_history[symbol]) > 10:  # Keep last 10
                self.spectral_history[symbol] = self.spectral_history[symbol][-10:]

            return {
                'drift_detected': drift_detected,
                'drift_score': spectral_drift_score,
                'current_spectra': current_spectra,
                'feature_drifts': drift_scores,
                'regime_change_detected': self.gpu_services.spectral_service.detect_regime_change(
                    historical_spectra, current_spectra
                )
            }

        except Exception as e:
            logger.error(f"Spectral drift check failed for {symbol}: {e}")
            return {
                'drift_detected': False,
                'drift_score': 0.0,
                'error': str(e)
            }

    def _calculate_psi(self, current_data: pd.Series, historical_data: pd.Series,
                       bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI) for drift detection.

        Args:
            current_data: Current distribution
            historical_data: Historical distribution
            bins: Number of bins for discretization

        Returns:
            PSI score
        """
        try:
            # Create bins
            all_data = pd.concat([current_data, historical_data])
            bin_edges = pd.qcut(all_data, q=bins, duplicates='drop', retbins=True)[1]

            # Calculate distributions
            current_dist = np.histogram(current_data, bins=bin_edges)[0]
            historical_dist = np.histogram(historical_data, bins=bin_edges)[0]

            # Convert to proportions
            current_prop = current_dist / len(current_data)
            historical_prop = historical_dist / len(historical_data)

            # Avoid division by zero
            current_prop = np.where(current_prop == 0, 1e-10, current_prop)
            historical_prop = np.where(historical_prop == 0, 1e-10, historical_prop)

            # Calculate PSI
            psi = np.sum((current_prop - historical_prop) * np.log(current_prop / historical_prop))

            return psi

        except Exception as e:
            logger.warning(f"PSI calculation failed: {e}")
            return 0.0

    def _generate_recommendations(self, drift_results: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on drift analysis.

        Args:
            drift_results: Results from comprehensive drift check

        Returns:
            List of recommendations
        """
        recommendations = []

        if drift_results['regime_change']:
            recommendations.append("[CRITICAL] Regime change detected - immediate retraining required")
            recommendations.append("Consider using different model architecture for new market conditions")

        if drift_results['performance_drift']['drift_detected']:
            pct_drift = drift_results['performance_drift']['drift_score'] * 100
            recommendations.append(f"[PERF] Performance degraded by {pct_drift:.1f}% - consider model retraining")

        if drift_results['data_drift']['drift_detected']:
            recommendations.append("[DATA] Data distribution shifted - validate feature engineering")

        if drift_results['spectral_drift']['drift_detected']:
            recommendations.append("[SPECTRAL] Spectral patterns changed - market regime may have shifted")

        if not recommendations:
            recommendations.append("[OK] No significant drift detected - continue monitoring")

        return recommendations

    # Mock data methods (replace with actual data access)
    def _get_recent_predictions(self, symbol: str) -> np.ndarray:
        """Get recent model predictions (mock implementation)."""
        # Mock data - replace with actual prediction retrieval
        return np.random.randn(50) * 0.02 + 0.01

    def _get_recent_actuals(self, symbol: str) -> np.ndarray:
        """Get recent actual values (mock implementation)."""
        # Mock data - replace with actual market data
        return np.random.randn(50) * 0.02

    def _get_performance_baseline(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get performance baseline metrics."""
        return self.performance_history.get(symbol)

    def _get_current_market_data(self, symbol: str) -> pd.DataFrame:
        """Get current market data for drift analysis."""
        # Mock data - replace with actual data pipeline calls
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        data = {
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000000, 5000000, 100),
            'returns': np.random.randn(100) * 0.02
        }
        return pd.DataFrame(data, index=dates)

    def _get_historical_market_data(self, symbol: str) -> pd.DataFrame:
        """Get historical market data for comparison."""
        # Mock historical data
        dates = pd.date_range(end=datetime.now() - timedelta(days=100), periods=200, freq='D')
        data = {
            'close': np.random.randn(200).cumsum() + 95,  # Slightly different baseline
            'volume': np.random.randint(800000, 4000000, 200),
            'returns': np.random.randn(200) * 0.025
        }
        return pd.DataFrame(data, index=dates)

    def _get_price_data_for_spectral_analysis(self, symbol: str) -> np.ndarray:
        """Get price data suitable for spectral analysis."""
        # Mock price data
        return np.random.randn(256) * 0.02 + 100  # 256 points for FFT

# Convenience functions
def create_drift_monitor_agent():
    """Create and configure drift monitor agent."""
    return DriftMonitorAgent()

def run_drift_analysis(symbol: str):
    """Run comprehensive drift analysis for a symbol."""
    agent = create_drift_monitor_agent()
    return agent.comprehensive_drift_check(symbol)
