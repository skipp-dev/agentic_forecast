"""
Risk Analytics

Advanced risk analytics for IB Forecast system.
Provides portfolio risk assessment, value-at-risk calculations, and risk management tools.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import scipy.stats as stats
from scipy.optimize import minimize

# Add paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.metrics_database import MetricsDatabase, MetricQuery

logger = logging.getLogger(__name__)

class RiskAnalytics:
    """
    Advanced risk analytics for portfolio and forecast risk management.

    Provides:
    - Value-at-Risk (VaR) calculations
    - Expected Shortfall (ES) analysis
    - Portfolio optimization
    - Stress testing
    - Risk factor analysis
    """

    def __init__(self, metrics_db: Optional[MetricsDatabase] = None):
        """
        Initialize risk analytics.

        Args:
            metrics_db: Metrics database instance
        """
        self.metrics_db = metrics_db or MetricsDatabase()

        # Risk configuration
        self.risk_config = {
            'confidence_levels': [0.95, 0.99],
            'time_horizons': [1, 5, 10, 20],  # days
            'risk_free_rate': 0.02,  # 2% annual risk-free rate
            'max_portfolio_weight': 0.3,  # Maximum weight per asset
            'min_portfolio_weight': -0.2  # Minimum weight (allows short selling)
        }

        logger.info("Risk Analytics initialized")

    def calculate_portfolio_var(self, returns: pd.DataFrame,
                               weights: np.ndarray,
                               confidence_level: float = 0.95,
                               method: str = 'historical') -> Dict[str, Any]:
        """
        Calculate portfolio Value-at-Risk.

        Args:
            returns: Historical returns DataFrame (assets as columns)
            weights: Portfolio weights array
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
            method: Calculation method ('historical', 'parametric', 'monte_carlo')

        Returns:
            Dictionary with VaR calculations
        """
        if method == 'historical':
            return self._calculate_historical_var(returns, weights, confidence_level)
        elif method == 'parametric':
            return self._calculate_parametric_var(returns, weights, confidence_level)
        elif method == 'monte_carlo':
            return self._calculate_monte_carlo_var(returns, weights, confidence_level)
        else:
            raise ValueError(f"Unknown VaR method: {method}")

    def calculate_portfolio_es(self, returns: pd.DataFrame,
                              weights: np.ndarray,
                              confidence_level: float = 0.95,
                              method: str = 'historical') -> Dict[str, Any]:
        """
        Calculate portfolio Expected Shortfall (Conditional VaR).

        Args:
            returns: Historical returns DataFrame
            weights: Portfolio weights array
            confidence_level: Confidence level
            method: Calculation method

        Returns:
            Dictionary with ES calculations
        """
        if method == 'historical':
            return self._calculate_historical_es(returns, weights, confidence_level)
        elif method == 'parametric':
            return self._calculate_parametric_es(returns, weights, confidence_level)
        else:
            raise ValueError(f"ES method {method} not implemented")

    def optimize_portfolio(self, returns: pd.DataFrame,
                          target_return: Optional[float] = None,
                          risk_free_rate: Optional[float] = None) -> Dict[str, Any]:
        """
        Optimize portfolio weights using Modern Portfolio Theory.

        Args:
            returns: Historical returns DataFrame
            target_return: Target portfolio return (optional)
            risk_free_rate: Risk-free rate (optional)

        Returns:
            Dictionary with optimal portfolio weights and metrics
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_config['risk_free_rate']

        # Calculate mean returns and covariance matrix
        mean_returns = returns.mean()
        cov_matrix = returns.cov()

        # Number of assets
        num_assets = len(mean_returns)

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]

        # Bounds for each weight
        bounds = [(self.risk_config['min_portfolio_weight'],
                  self.risk_config['max_portfolio_weight']) for _ in range(num_assets)]

        # Add target return constraint if specified
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.sum(mean_returns * x) - target_return
            })

        # Objective function: minimize portfolio variance
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))

        # Initial guess: equal weights
        initial_weights = np.array([1/num_assets] * num_assets)

        # Optimize
        result = minimize(portfolio_variance, initial_weights,
                         method='SLSQP', bounds=bounds, constraints=constraints)

        if not result.success:
            logger.warning(f"Portfolio optimization failed: {result.message}")
            return {'status': 'failed', 'message': result.message}

        optimal_weights = result.x

        # Calculate portfolio metrics
        portfolio_return = np.sum(mean_returns * optimal_weights)
        portfolio_volatility = np.sqrt(portfolio_variance(optimal_weights))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

        return {
            'status': 'success',
            'weights': optimal_weights.tolist(),
            'expected_return': float(portfolio_return),
            'volatility': float(portfolio_volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'asset_contributions': {
                asset: float(weight) for asset, weight in zip(returns.columns, optimal_weights)
            }
        }

    def perform_stress_test(self, returns: pd.DataFrame,
                           weights: np.ndarray,
                           scenarios: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Perform stress testing on portfolio.

        Args:
            returns: Historical returns DataFrame
            weights: Portfolio weights
            scenarios: Dictionary of stress scenarios

        Returns:
            Dictionary with stress test results
        """
        results = {}

        for scenario_name, shocks in scenarios.items():
            # Apply shocks to returns
            stressed_returns = returns.copy()

            for asset, shock in shocks.items():
                if asset in stressed_returns.columns:
                    stressed_returns[asset] = stressed_returns[asset] + shock

            # Calculate portfolio returns under stress
            portfolio_returns = stressed_returns.dot(weights)

            # Calculate losses (negative returns)
            losses = -portfolio_returns

            # Calculate VaR and ES under stress
            var_95 = np.percentile(losses, 95)
            var_99 = np.percentile(losses, 99)
            es_95 = losses[losses > var_95].mean()
            es_99 = losses[losses > var_99].mean()

            results[scenario_name] = {
                'var_95': float(var_95),
                'var_99': float(var_99),
                'es_95': float(es_95) if not np.isnan(es_95) else None,
                'es_99': float(es_99) if not np.isnan(es_99) else None,
                'max_loss': float(losses.max()),
                'worst_day': stressed_returns.index[losses.idxmax()].isoformat()
            }

        return results

    def analyze_risk_factors(self, returns: pd.DataFrame,
                           factors: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform risk factor analysis using multi-factor model.

        Args:
            returns: Asset returns DataFrame
            factors: Risk factors DataFrame

        Returns:
            Dictionary with factor analysis results
        """
        results = {}

        for asset in returns.columns:
            asset_returns = returns[asset]

            # Add constant term for intercept
            X = pd.concat([pd.Series(1, index=factors.index, name='constant'), factors], axis=1)
            X = X.loc[asset_returns.index]  # Align indices

            # Remove NaN values
            valid_data = pd.concat([asset_returns, X], axis=1).dropna()
            y = valid_data[asset]
            X_clean = valid_data.drop(asset, axis=1)

            if len(X_clean) < len(factors.columns) + 1:
                logger.warning(f"Insufficient data for factor analysis of {asset}")
                continue

            # Perform regression
            try:
                beta = np.linalg.inv(X_clean.T @ X_clean) @ X_clean.T @ y
                residuals = y - X_clean @ beta
                r_squared = 1 - (residuals.var() / y.var())

                results[asset] = {
                    'intercept': float(beta[0]),
                    'factor_betas': {
                        factor: float(beta[i+1]) for i, factor in enumerate(factors.columns)
                    },
                    'r_squared': float(r_squared),
                    'residual_volatility': float(residuals.std())
                }
            except np.linalg.LinAlgError:
                logger.warning(f"Singular matrix in factor analysis for {asset}")
                continue

        return results

    def calculate_risk_metrics_from_db(self, start_date: datetime,
                                     end_date: datetime) -> Dict[str, Any]:
        """
        Calculate risk metrics from stored forecast data.

        Args:
            start_date: Start date for analysis
            end_date: End date for analysis

        Returns:
            Dictionary with risk metrics
        """
        # Query forecast returns from metrics database
        query = MetricQuery(
            metric_names=['forecast.returns', 'forecast.volatility'],
            start_time=start_date,
            end_time=end_date,
            aggregation=None,
            interval='1d'
        )

        metrics = self.metrics_db.query_metrics(query)

        if metrics.empty:
            return {'status': 'no_data'}

        # Extract returns data
        returns_data = metrics[metrics['metric_name'] == 'forecast.returns']
        if returns_data.empty:
            return {'status': 'no_returns_data'}

        # Calculate risk metrics
        returns = returns_data['value'].values

        # Basic statistics
        mean_return = float(np.mean(returns))
        volatility = float(np.std(returns))
        skewness = float(stats.skew(returns))
        kurtosis = float(stats.kurtosis(returns))

        # VaR calculations
        var_95 = float(np.percentile(-returns, 95))  # Loss VaR
        var_99 = float(np.percentile(-returns, 99))

        # Expected Shortfall
        losses = -returns
        es_95 = float(losses[losses > var_95].mean()) if len(losses[losses > var_95]) > 0 else var_95
        es_99 = float(losses[losses > var_99].mean()) if len(losses[losses > var_99]) > 0 else var_99

        # Sharpe ratio (assuming risk-free rate)
        risk_free_daily = self.risk_config['risk_free_rate'] / 252  # Daily risk-free rate
        sharpe_ratio = float((mean_return - risk_free_daily) / volatility) if volatility > 0 else 0

        # Maximum drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = float(np.min(drawdown))

        return {
            'status': 'success',
            'period': f"{start_date.date()} to {end_date.date()}",
            'basic_stats': {
                'mean_return': mean_return,
                'volatility': volatility,
                'skewness': skewness,
                'kurtosis': kurtosis
            },
            'risk_metrics': {
                'var_95': var_95,
                'var_99': var_99,
                'es_95': es_95,
                'es_99': es_99,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            },
            'data_points': len(returns)
        }

    def _calculate_historical_var(self, returns: pd.DataFrame,
                                weights: np.ndarray,
                                confidence_level: float) -> Dict[str, Any]:
        """Calculate VaR using historical simulation."""
        # Calculate portfolio returns
        portfolio_returns = returns.dot(weights)

        # Calculate losses (negative returns)
        losses = -portfolio_returns

        # Calculate VaR as percentile of losses
        var = np.percentile(losses, (1 - confidence_level) * 100)

        return {
            'method': 'historical',
            'confidence_level': confidence_level,
            'var': float(var),
            'expected_shortfall': float(losses[losses > var].mean()) if len(losses[losses > var]) > 0 else var
        }

    def _calculate_parametric_var(self, returns: pd.DataFrame,
                                weights: np.ndarray,
                                confidence_level: float) -> Dict[str, Any]:
        """Calculate VaR using parametric method (normal distribution assumption)."""
        # Calculate portfolio mean and volatility
        portfolio_mean = returns.dot(weights).mean()
        portfolio_volatility = np.sqrt(weights.T @ returns.cov() @ weights)

        # Calculate VaR using normal distribution
        z_score = stats.norm.ppf(confidence_level)
        var = -(portfolio_mean + z_score * portfolio_volatility)

        return {
            'method': 'parametric',
            'confidence_level': confidence_level,
            'var': float(var),
            'portfolio_mean': float(portfolio_mean),
            'portfolio_volatility': float(portfolio_volatility)
        }

    def _calculate_monte_carlo_var(self, returns: pd.DataFrame,
                                 weights: np.ndarray,
                                 confidence_level: float,
                                 num_simulations: int = 10000) -> Dict[str, Any]:
        """Calculate VaR using Monte Carlo simulation."""
        # Fit multivariate normal distribution to returns
        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values

        # Generate random scenarios
        np.random.seed(42)  # For reproducibility
        scenarios = np.random.multivariate_normal(mean_returns, cov_matrix, num_simulations)

        # Calculate portfolio returns for each scenario
        portfolio_returns = scenarios.dot(weights)

        # Calculate losses
        losses = -portfolio_returns

        # Calculate VaR
        var = np.percentile(losses, (1 - confidence_level) * 100)

        return {
            'method': 'monte_carlo',
            'confidence_level': confidence_level,
            'var': float(var),
            'num_simulations': num_simulations,
            'expected_shortfall': float(losses[losses > var].mean()) if len(losses[losses > var]) > 0 else var
        }

    def _calculate_historical_es(self, returns: pd.DataFrame,
                               weights: np.ndarray,
                               confidence_level: float) -> Dict[str, Any]:
        """Calculate Expected Shortfall using historical method."""
        portfolio_returns = returns.dot(weights)
        losses = -portfolio_returns

        var = np.percentile(losses, (1 - confidence_level) * 100)
        es = losses[losses > var].mean() if len(losses[losses > var]) > 0 else var

        return {
            'method': 'historical',
            'confidence_level': confidence_level,
            'es': float(es),
            'var': float(var)
        }

    def _calculate_parametric_es(self, returns: pd.DataFrame,
                               weights: np.ndarray,
                               confidence_level: float) -> Dict[str, Any]:
        """Calculate Expected Shortfall using parametric method."""
        portfolio_mean = returns.dot(weights).mean()
        portfolio_volatility = np.sqrt(weights.T @ returns.cov() @ weights)

        z_score = stats.norm.ppf(confidence_level)
        var = -(portfolio_mean + z_score * portfolio_volatility)

        # For normal distribution, ES = mean + volatility * (pdf(z) / (1-confidence))
        pdf_z = stats.norm.pdf(z_score)
        es = -(portfolio_mean + portfolio_volatility * (pdf_z / (1 - confidence_level)))

        return {
            'method': 'parametric',
            'confidence_level': confidence_level,
            'es': float(es),
            'var': float(var)
        }