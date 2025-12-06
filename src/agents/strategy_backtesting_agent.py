"""
Strategy Backtesting Agent

Dedicated agent for simulating strategy performance over historical data.
Calculates key financial metrics (Sharpe, Drawdown, etc.) to validate model recommendations.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.backtesting.backtest_executor import BacktestExecutor
from src.core.state import PipelineGraphState

logger = logging.getLogger(__name__)

class StrategyBacktestingAgent:
    """
    Agent for running backtests and analyzing strategy performance.
    """

    def __init__(self, graph: Any, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the backtesting agent.
        
        Args:
            graph: The compiled LangGraph executable to backtest.
            config: Configuration dictionary.
        """
        self.graph = graph
        self.config = config or {}
        
    def run_backtest(self, 
                     start_date: str, 
                     end_date: str, 
                     initial_cash: float = 100000.0) -> Dict[str, Any]:
        """
        Execute a backtest for the given period.
        
        Args:
            start_date: Start date (ISO format).
            end_date: End date (ISO format).
            initial_cash: Initial portfolio cash.
            
        Returns:
            Dictionary containing backtest results and metrics.
        """
        logger.info(f"Starting backtest from {start_date} to {end_date}...")
        
        executor = BacktestExecutor(
            graph=self.graph,
            start_date=start_date,
            end_date=end_date,
            initial_cash=initial_cash
        )
        
        # Run the backtest
        results = self._execute_backtest_loop(executor)
        
        # Merge portfolio history into results for metric calculation
        if executor.portfolio_history:
            portfolio_df = pd.DataFrame(executor.portfolio_history)
            # Ensure date is datetime
            portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
            
            # If results has date, merge. If results is empty (no forecasts), use portfolio history.
            if not results.empty and 'date' in results.columns:
                 results['date'] = pd.to_datetime(results['date'])
                 # Merge on date
                 results = pd.merge(results, portfolio_df, on='date', how='outer')
            else:
                results = portfolio_df
        
        metrics = self._calculate_metrics(results)
        
        return {
            'metrics': metrics,
            'results': results
        }

    def _execute_backtest_loop(self, executor: BacktestExecutor) -> pd.DataFrame:
        """
        Internal method to drive the backtest loop.
        """
        logger.info("Executing backtest simulation...")
        
        # Construct initial state
        initial_state = {
            "symbols": self.config.get("symbols", ["SPY"]), # Default to SPY if not set
            "start_date": executor.start_date.strftime('%Y-%m-%d'),
            "end_date": executor.end_date.strftime('%Y-%m-%d'),
            "run_id": f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "config": self.config,
            "run_type": "BACKTEST",
            "data": {},
            "features": {},
            "macro_data": {},
            "regimes": {},
            "best_models": {},
            "forecasts": {},
            "analytics_results": {},
            "drift_detected": [],
            "drift_metrics": {},
            "retrained_models": [],
            "hpo_triggered": False,
            "hpo_results": {},
            "errors": [],
            "run_status": "STARTING",
            "next_step": "init",
            "deep_research_conducted": False,
            "horizon_forecasts": {},
            "interpreted_forecasts": False,
            "report_metadata": {},
            "report_generated": False,
            "signals": {}
        }
        
        # Run the executor
        results_df = executor.run(initial_state)
        
        return results_df

    def _calculate_metrics(self, results: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate performance metrics from backtest results.
        """
        if results.empty:
            return {
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_return': 0.0,
                'win_rate': 0.0
            }
            
        # Assuming results contains 'total_value' and 'date'
        # If results comes from executor.results, it might be forecasts.
        # We need portfolio history from executor.
        # But executor.run returns self.results which is forecasts/metrics per step.
        # We need to access executor.portfolio_history for financial metrics.
        
        # Since we passed executor to _execute_backtest_loop, we can access it here if we change signature
        # or we rely on what is returned.
        # The current implementation of executor.run returns pd.DataFrame(self.results).
        # We should probably modify executor to return both or access history.
        
        # For now, let's calculate basic metrics if 'total_value' is in results.
        # If not, we return zeros.
        
        if 'total_value' not in results.columns:
             return {
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_return': 0.0,
                'win_rate': 0.0
            }
            
        # Calculate Returns
        results['returns'] = results['total_value'].pct_change()
        
        # Total Return
        start_val = results['total_value'].iloc[0]
        end_val = results['total_value'].iloc[-1]
        total_return = (end_val - start_val) / start_val if start_val != 0 else 0.0
        
        # Sharpe Ratio (Annualized, assuming daily data)
        mean_return = results['returns'].mean()
        std_return = results['returns'].std()
        sharpe = (mean_return / std_return) * np.sqrt(252) if std_return != 0 else 0.0
        
        # Max Drawdown
        rolling_max = results['total_value'].cummax()
        drawdown = (results['total_value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        return {
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_drawdown),
            'total_return': float(total_return),
            'win_rate': 0.0 # Placeholder
        }
