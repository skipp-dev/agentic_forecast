import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from src.agents.strategy_backtesting_agent import StrategyBacktestingAgent
from src.backtesting.backtest_executor import BacktestExecutor

class TestStrategyBacktesting(unittest.TestCase):
    
    def test_metrics_calculation(self):
        # Create dummy agent
        agent = StrategyBacktestingAgent(graph=None)
        
        # Create dummy results with total_value
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        # Simulate a 1% daily return trend with some noise
        returns = np.random.normal(0.001, 0.01, 100)
        values = [100000]
        for r in returns[1:]:
            values.append(values[-1] * (1 + r))
            
        results = pd.DataFrame({
            'date': dates,
            'total_value': values
        })
        
        metrics = agent._calculate_metrics(results)
        
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        self.assertIn('total_return', metrics)
        
        # Basic sanity checks
        self.assertIsInstance(metrics['sharpe_ratio'], float)
        self.assertLessEqual(metrics['max_drawdown'], 0.0) # Drawdown is negative or zero
        
    def test_integration_flow(self):
        # Mock the graph and executor
        mock_graph = MagicMock()
        agent = StrategyBacktestingAgent(graph=mock_graph)
        
        # Mock executor behavior
        # We can't easily mock the internal BacktestExecutor instantiation inside run_backtest
        # without dependency injection or patching.
        # However, we can test _execute_backtest_loop directly if we pass a mock executor.
        
        mock_executor = MagicMock()
        mock_executor.start_date = pd.Timestamp('2023-01-01')
        mock_executor.end_date = pd.Timestamp('2023-01-05')
        mock_executor.run.return_value = pd.DataFrame({'col': [1, 2]})
        mock_executor.portfolio_history = [{'date': '2023-01-01', 'total_value': 100}]
        
        results = agent._execute_backtest_loop(mock_executor)
        
        self.assertFalse(results.empty)
        mock_executor.run.assert_called_once()

if __name__ == '__main__':
    unittest.main()
