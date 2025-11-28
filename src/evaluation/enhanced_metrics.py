"""
Enhanced Evaluation Metrics for Financial Forecasting

This module provides comprehensive evaluation metrics for financial forecasting models,
including directional accuracy, Sharpe ratio, drawdown analysis, and other trading-related
performance measures.
"""

import numpy as np
from typing import Dict, List, Optional, Union
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

logger = logging.getLogger(__name__)


class EnhancedFinancialMetrics:
    """
    Enhanced evaluation metrics for financial forecasting models.

    Provides comprehensive performance analysis including directional accuracy,
    risk-adjusted returns (Sharpe ratio), and drawdown analysis.
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize enhanced financial metrics calculator.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation (default: 2%)
        """
        self.risk_free_rate = risk_free_rate

    def calculate_directional_accuracy(self,
                                     y_true: np.ndarray,
                                     y_pred: np.ndarray,
                                     threshold: float = 0.0) -> Dict[str, float]:
        """
        Calculate directional accuracy - how well the model predicts price direction.

        Args:
            y_true: Actual price changes/values
            y_pred: Predicted price changes/values
            threshold: Minimum change threshold to consider (default: 0.0)

        Returns:
            Dictionary with directional accuracy metrics
        """
        # Calculate actual and predicted directions
        actual_directions = np.sign(np.diff(y_true))
        predicted_directions = np.sign(np.diff(y_pred))

        # Remove zero changes if threshold is set
        if threshold > 0:
            actual_changes = np.abs(np.diff(y_true))
            valid_indices = actual_changes >= threshold
            actual_directions = actual_directions[valid_indices]
            predicted_directions = predicted_directions[valid_indices]

        # Calculate directional accuracy
        correct_predictions = np.sum(actual_directions == predicted_directions)
        total_predictions = len(actual_directions)

        if total_predictions == 0:
            directional_accuracy = 0.0
        else:
            directional_accuracy = correct_predictions / total_predictions

        # Calculate hit rates for up and down moves
        up_moves = actual_directions == 1
        down_moves = actual_directions == -1

        up_accuracy = np.mean(predicted_directions[up_moves] == 1) if np.any(up_moves) else 0.0
        down_accuracy = np.mean(predicted_directions[down_moves] == -1) if np.any(down_moves) else 0.0

        return {
            'directional_accuracy': directional_accuracy,
            'up_move_accuracy': up_accuracy,
            'down_move_accuracy': down_accuracy,
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions
        }

    def calculate_sharpe_ratio(self,
                              returns: np.ndarray,
                              annualize: bool = True,
                              periods_per_year: int = 252) -> Dict[str, float]:
        """
        Calculate Sharpe ratio - risk-adjusted return measure.

        Args:
            returns: Array of returns (can be daily, weekly, etc.)
            annualize: Whether to annualize the Sharpe ratio
            periods_per_year: Number of periods in a year (252 for daily, 52 for weekly, etc.)

        Returns:
            Dictionary with Sharpe ratio and related metrics
        """
        if len(returns) < 2:
            return {
                'sharpe_ratio': 0.0,
                'annualized_sharpe': 0.0,
                'excess_return': 0.0,
                'volatility': 0.0,
                'risk_free_rate': self.risk_free_rate
            }

        # Calculate excess returns
        excess_returns = returns - (self.risk_free_rate / periods_per_year)

        # Calculate Sharpe ratio
        mean_excess_return = np.mean(excess_returns)
        volatility = np.std(excess_returns, ddof=1)

        if volatility == 0:
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = mean_excess_return / volatility

        # Annualize if requested
        if annualize:
            annualized_sharpe = sharpe_ratio * np.sqrt(periods_per_year)
        else:
            annualized_sharpe = sharpe_ratio

        return {
            'sharpe_ratio': sharpe_ratio,
            'annualized_sharpe': annualized_sharpe,
            'excess_return': mean_excess_return,
            'volatility': volatility,
            'risk_free_rate': self.risk_free_rate,
            'periods_per_year': periods_per_year
        }

    def calculate_drawdown_analysis(self,
                                  portfolio_values: np.ndarray,
                                  return_threshold: float = -0.01) -> Dict[str, Union[float, int, List]]:
        """
        Calculate comprehensive drawdown analysis.

        Args:
            portfolio_values: Array of portfolio values over time
            return_threshold: Minimum return threshold to consider as drawdown

        Returns:
            Dictionary with drawdown metrics
        """
        if len(portfolio_values) < 2:
            return {
                'max_drawdown': 0.0,
                'max_drawdown_duration': 0,
                'average_drawdown': 0.0,
                'drawdown_count': 0,
                'current_drawdown': 0.0,
                'recovery_time': 0
            }

        # Calculate returns
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + returns)

        # Calculate drawdowns
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max

        # Find drawdown periods
        drawdown_periods = []
        in_drawdown = False
        drawdown_start = 0

        for i, dd in enumerate(drawdowns):
            if dd < return_threshold and not in_drawdown:
                # Start of drawdown
                in_drawdown = True
                drawdown_start = i
            elif dd >= 0 and in_drawdown:
                # End of drawdown
                in_drawdown = False
                drawdown_periods.append({
                    'start_idx': drawdown_start,
                    'end_idx': i,
                    'max_drawdown': np.min(drawdowns[drawdown_start:i+1]),
                    'duration': i - drawdown_start + 1,
                    'recovery_time': i - drawdown_start + 1
                })

        # Handle ongoing drawdown
        if in_drawdown:
            current_dd = np.min(drawdowns[drawdown_start:])
            drawdown_periods.append({
                'start_idx': drawdown_start,
                'end_idx': len(drawdowns) - 1,
                'max_drawdown': current_dd,
                'duration': len(drawdowns) - drawdown_start,
                'recovery_time': 0  # Still in drawdown
            })

        # Calculate metrics
        if drawdown_periods:
            max_dd = min(dd['max_drawdown'] for dd in drawdown_periods)
            max_dd_duration = max(dd['duration'] for dd in drawdown_periods)
            avg_dd = np.mean([dd['max_drawdown'] for dd in drawdown_periods])
            dd_count = len(drawdown_periods)
        else:
            max_dd = 0.0
            max_dd_duration = 0
            avg_dd = 0.0
            dd_count = 0

        # Current drawdown
        current_dd = drawdowns[-1] if len(drawdowns) > 0 else 0.0

        return {
            'max_drawdown': abs(max_dd),  # Return as positive value
            'max_drawdown_duration': max_dd_duration,
            'average_drawdown': abs(avg_dd),
            'drawdown_count': dd_count,
            'current_drawdown': abs(current_dd),
            'drawdown_periods': drawdown_periods,
            'return_threshold': return_threshold
        }

    def calculate_trading_returns(self,
                                y_true: np.ndarray,
                                y_pred: np.ndarray,
                                transaction_cost: float = 0.001) -> Dict[str, float]:
        """
        Calculate trading returns based on directional predictions.

        Args:
            y_true: Actual price values
            y_pred: Predicted price values
            transaction_cost: Transaction cost per trade (default: 0.1%)

        Returns:
            Dictionary with trading performance metrics
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")

        # Calculate predicted directions
        predicted_directions = np.sign(np.diff(y_pred))

        # Calculate actual returns
        actual_returns = np.diff(y_true) / y_true[:-1]

        # Simple trading strategy: buy if prediction > 0, sell/short if prediction < 0
        strategy_returns = predicted_directions * actual_returns

        # Apply transaction costs (simplified: cost per trade)
        # Direction changes indicate trades (shift by one to align with returns)
        direction_changes = np.abs(np.diff(predicted_directions)) > 0
        trade_costs = np.zeros_like(strategy_returns)
        if len(direction_changes) > 0:
            trade_costs[1:] = direction_changes * transaction_cost  # Align with returns

        # Net returns after costs
        net_returns = strategy_returns - trade_costs

        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + net_returns)

        # Calculate metrics
        total_return = cumulative_returns[-1] - 1 if len(cumulative_returns) > 0 else 0
        annualized_return = (1 + total_return) ** (252 / len(net_returns)) - 1 if len(net_returns) > 252 else total_return
        volatility = np.std(net_returns, ddof=1) * np.sqrt(252) if len(net_returns) > 1 else 0

        # Win rate
        winning_trades = np.sum(net_returns > 0)
        total_trades = len(net_returns)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Profit factor
        gross_profit = np.sum(net_returns[net_returns > 0])
        gross_loss = abs(np.sum(net_returns[net_returns < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': (annualized_return - self.risk_free_rate) / volatility if volatility > 0 else 0,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_return': np.max(net_returns) if len(net_returns) > 0 else 0,
            'min_return': np.min(net_returns) if len(net_returns) > 0 else 0,
            'total_trades': total_trades,
            'transaction_cost': transaction_cost
        }

    def calculate_comprehensive_metrics(self,
                                      y_true: np.ndarray,
                                      y_pred: np.ndarray,
                                      portfolio_values: Optional[np.ndarray] = None) -> Dict[str, Union[float, Dict]]:
        """
        Calculate comprehensive set of financial evaluation metrics.

        Args:
            y_true: Actual values
            y_pred: Predicted values
            portfolio_values: Optional portfolio values for drawdown analysis

        Returns:
            Dictionary with all available metrics
        """
        # Basic regression metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

        # Directional accuracy
        directional_metrics = self.calculate_directional_accuracy(y_true, y_pred)

        # Trading returns (if we have enough data)
        if len(y_true) > 30:  # Need minimum data for meaningful trading metrics
            trading_metrics = self.calculate_trading_returns(y_true, y_pred)
        else:
            trading_metrics = {}

        # Sharpe ratio from returns
        if len(y_true) > 1:
            returns = np.diff(y_true) / y_true[:-1]
            sharpe_metrics = self.calculate_sharpe_ratio(returns)
        else:
            sharpe_metrics = {}

        # Drawdown analysis
        if portfolio_values is not None and len(portfolio_values) > 10:
            drawdown_metrics = self.calculate_drawdown_analysis(portfolio_values)
        else:
            # Use predicted values as proxy for portfolio values
            drawdown_metrics = self.calculate_drawdown_analysis(y_pred)

        return {
            'regression_metrics': {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'r2': r2,
                'mse': mse
            },
            'directional_metrics': directional_metrics,
            'trading_metrics': trading_metrics,
            'sharpe_metrics': sharpe_metrics,
            'drawdown_metrics': drawdown_metrics,
            'summary': {
                'best_metric': 'directional_accuracy' if directional_metrics['directional_accuracy'] > 0.5 else 'mae',
                'overall_score': directional_metrics['directional_accuracy'] * (1 - abs(drawdown_metrics['max_drawdown'])),
                'risk_adjusted_score': directional_metrics['directional_accuracy'] / (1 + drawdown_metrics['max_drawdown'])
            }
        }

    def print_comprehensive_report(self, metrics_dict: Dict) -> None:
        """
        Print a comprehensive evaluation report.

        Args:
            metrics_dict: Dictionary returned by calculate_comprehensive_metrics
        """
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE FINANCIAL EVALUATION REPORT")
        print("="*80)

        # Regression Metrics
        reg = metrics_dict['regression_metrics']
        print("\n📊 REGRESSION METRICS:")
        print(f"  MAE: {reg['mae']:.4f}")
        print(f"  RMSE: {reg['rmse']:.4f}")
        print(f"  MAPE: {reg['mape']:.2f}%")
        print(f"  R²: {reg['r2']:.4f}")

        # Directional Metrics
        dir_metrics = metrics_dict['directional_metrics']
        print("\n📈 DIRECTIONAL METRICS:")
        print(f"  Directional Accuracy: {dir_metrics['directional_accuracy']:.1%}")
        print(f"  Up Move Accuracy: {dir_metrics['up_move_accuracy']:.1%}")
        print(f"  Down Move Accuracy: {dir_metrics['down_move_accuracy']:.1%}")
        print(f"  Total Predictions: {dir_metrics['total_predictions']}")

        # Trading Metrics
        if metrics_dict['trading_metrics']:
            trade = metrics_dict['trading_metrics']
            print("\n💰 TRADING METRICS:")
            print(f"  Total Return: {trade['total_return']:.2%}")
            print(f"  Annualized Return: {trade['annualized_return']:.2%}")
            print(f"  Win Rate: {trade['win_rate']:.1%}")
            print(f"  Profit Factor: {trade['profit_factor']:.2f}")
            print(f"  Total Trades: {trade['total_trades']}")

        # Sharpe Ratio
        if metrics_dict['sharpe_metrics']:
            sharpe = metrics_dict['sharpe_metrics']
            print("\n📉 RISK METRICS:")
            print(f"  Sharpe Ratio: {sharpe['annualized_sharpe']:.3f}")
            print(f"  Annualized Volatility: {sharpe['volatility']*np.sqrt(sharpe['periods_per_year']):.2%}")
            print(f"  Risk-Free Rate: {sharpe['risk_free_rate']:.2%}")

        # Drawdown Analysis
        dd = metrics_dict['drawdown_metrics']
        print("\n📉 DRAWDOWN ANALYSIS:")
        print(f"  Max Drawdown: {dd['max_drawdown']:.2%}")
        print(f"  Max Drawdown Duration: {dd['max_drawdown_duration']} periods")
        print(f"  Average Drawdown: {dd['average_drawdown']:.2%}")
        print(f"  Drawdown Count: {dd['drawdown_count']}")
        print(f"  Current Drawdown: {dd['current_drawdown']:.2%}")

        # Summary
        summary = metrics_dict['summary']
        print("\n🏆 SUMMARY:")
        print(f"  Overall Score: {summary['overall_score']:.3f}")
        print(f"  Risk-Adjusted Score: {summary['risk_adjusted_score']:.3f}")

        print("="*80)