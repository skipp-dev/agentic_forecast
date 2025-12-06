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
from src.services.model_registry_service import ModelRegistryService
from src.services.inference_service import InferenceService
from src.monitoring.metrics import DRIFT_MONITOR_RUNS, DRIFT_FLAGS_RAISED, DRIFT_SCORE, SYSTEM_ERRORS

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

    def __init__(self, gpu_services=None, data_pipeline=None, model_registry=None, inference_service=None):
        """
        Initialize enhanced drift monitor agent.

        Args:
            gpu_services: GPU services instance for spectral analysis
            data_pipeline: Data pipeline instance
            model_registry: Model registry service
            inference_service: Inference service
        """
        super().__init__()
        self.gpu_services = gpu_services or get_gpu_services()
        self.data_pipeline = data_pipeline or DataPipeline()
        self.model_registry = model_registry or ModelRegistryService()
        self.inference_service = inference_service or InferenceService()

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
        DRIFT_MONITOR_RUNS.labels(symbol=symbol).inc()

        try:
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
            DRIFT_SCORE.labels(symbol=symbol).set(results['overall_drift_score'])
            
            results['regime_change'] = results['overall_drift_score'] > self.thresholds['regime_change']

            if results['regime_change']:
                DRIFT_FLAGS_RAISED.labels(symbol=symbol, drift_type='regime_change').inc()

            # Generate recommendations
            results['recommendations'] = self._generate_recommendations(results)

            logger.info(f"Drift analysis complete for {symbol}: score={results['overall_drift_score']:.3f}")

            return results
            
        except Exception as e:
            logger.error(f"Drift Monitor failed for {symbol}: {e}")
            SYSTEM_ERRORS.labels(component='drift_monitor').inc()
            # Guardrail: Return safe default instead of crashing or returning None
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'performance_drift': {'drift_detected': False, 'drift_score': 0.0},
                'data_drift': {'drift_detected': False, 'drift_score': 0.0},
                'spectral_drift': {'drift_detected': False, 'drift_score': 0.0},
                'regime_change': False,
                'overall_drift_score': 0.0,
                'recommendations': [],
                'error': str(e)
            }

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

    def _get_recent_predictions(self, symbol: str) -> np.ndarray:
        """Get recent model predictions using InferenceService."""
        try:
            # Get best model
            best_model = self.model_registry.get_best_model(symbol, metric='mae', mode='min')
            if not best_model:
                logger.warning(f"No model found for {symbol}")
                return np.array([])
            
            model_id = best_model['model_id']
            model_type = best_model['model_type']
            
            # Fetch recent data for prediction context
            # We need enough history for the model to predict
            # Fetch last 60 days
            df = self.data_pipeline.fetch_stock_data(symbol, period='3mo')
            if df.empty:
                return np.array([])
                
            # Predict for the last 30 days (simulating "recent" predictions)
            # We iterate or just predict the last window?
            # For simplicity, let's predict the last 'horizon' steps using the data before it.
            # But drift monitoring usually looks at a window of predictions vs actuals.
            # Let's try to predict the last 30 days using a rolling window or just one shot if horizon allows.
            # If horizon is small (e.g. 5), we can't predict 30 days in one shot without autoregression.
            # InferenceService.predict handles horizon.
            
            # Let's just predict the last 'horizon' days for now, as that's what we can reliably do without complex rolling logic here.
            # Or better, just return empty if we can't easily reconstruct predictions.
            # But we need predictions for performance drift.
            
            # Let's use the validation predictions stored in the model metadata if available?
            # No, that's validation at training time. We want *recent* performance.
            
            # Let's skip complex prediction reconstruction and return empty for now, 
            # effectively disabling performance drift check unless we have a better way.
            # OR, we can use the 'val_preds' if we consider the training run "recent".
            
            # For now, let's try to predict the last 14 days.
            horizon = 14
            if len(df) > horizon * 2:
                # Use data up to -horizon as context
                context_df = df.iloc[:-horizon]
                
                result = self.inference_service.predict(
                    symbol=symbol,
                    model_id=model_id,
                    model_type=model_type,
                    data=context_df,
                    horizon=horizon
                )
                
                if result['status'] in ['success', 'success_fallback']:
                    preds = result['predictions']
                    if isinstance(preds, pd.DataFrame):
                        # Find prediction column
                        cols = [c for c in preds.columns if c not in ['ds', 'unique_id', 'y']]
                        if cols:
                            return preds[cols[0]].values
                    else:
                        return np.array(preds)
            
            return np.array([])
            
        except Exception as e:
            logger.error(f"Failed to get recent predictions for {symbol}: {e}")
            return np.array([])

    def _get_recent_actuals(self, symbol: str) -> np.ndarray:
        """Get recent actual values."""
        try:
            # Fetch last 14 days (matching prediction horizon above)
            df = self.data_pipeline.fetch_stock_data(symbol, period='1mo')
            if not df.empty:
                # Return last 14 days 'close' or 'y'
                if 'y' in df.columns:
                    return df['y'].iloc[-14:].values
                elif 'close' in df.columns:
                    return df['close'].iloc[-14:].values
            return np.array([])
        except Exception as e:
            logger.error(f"Failed to get recent actuals for {symbol}: {e}")
            return np.array([])

    def _get_performance_baseline(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get performance baseline metrics from Model Registry."""
        try:
            best_model = self.model_registry.get_best_model(symbol, metric='mae', mode='min')
            if best_model and 'metrics' in best_model:
                metrics = best_model['metrics']
                # Ensure we have mae and rmse
                if 'mae' in metrics:
                    if 'rmse' not in metrics:
                        metrics['rmse'] = metrics.get('mse', 0.0) ** 0.5
                    return metrics
            return None
        except Exception as e:
            logger.error(f"Failed to get performance baseline for {symbol}: {e}")
            return None

    def _get_current_market_data(self, symbol: str) -> pd.DataFrame:
        """Get current market data for drift analysis."""
        try:
            # Fetch last 100 days
            return self.data_pipeline.fetch_stock_data(symbol, period='6mo').iloc[-100:]
        except Exception as e:
            logger.error(f"Failed to get current market data for {symbol}: {e}")
            return pd.DataFrame()

    def _get_historical_market_data(self, symbol: str) -> pd.DataFrame:
        """Get historical market data for comparison."""
        try:
            # Fetch 2 years, take the first year (or older part)
            df = self.data_pipeline.fetch_stock_data(symbol, period='2y')
            if len(df) > 200:
                # Return data from 100 days ago and older
                return df.iloc[:-100]
            return df
        except Exception as e:
            logger.error(f"Failed to get historical market data for {symbol}: {e}")
            return pd.DataFrame()

    def _get_price_data_for_spectral_analysis(self, symbol: str) -> np.ndarray:
        """Get price data suitable for spectral analysis."""
        try:
            df = self._get_current_market_data(symbol)
            if not df.empty:
                if 'close' in df.columns:
                    return df['close'].values
                elif 'y' in df.columns:
                    return df['y'].values
            return np.array([])
        except Exception as e:
            logger.error(f"Failed to get spectral data for {symbol}: {e}")
            return np.array([])

# Convenience functions
def create_drift_monitor_agent():
    """Create and configure drift monitor agent."""
    return DriftMonitorAgent()

def run_drift_analysis(symbol: str):
    """Run comprehensive drift analysis for a symbol."""
    agent = create_drift_monitor_agent()
    return agent.comprehensive_drift_check(symbol)
