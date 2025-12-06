import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.backtesting.backtest_executor import BacktestExecutor
from src.brokers.paper_broker import PaperBroker

class TestBacktestExecutor(unittest.TestCase):
    
    @patch('src.backtesting.backtest_executor.PaperBroker')
    def test_initialization(self, MockBroker):
        # Setup Mock Broker
        mock_broker_instance = MockBroker.return_value
        mock_broker_instance.db_service = MagicMock()
        
        # Mock Graph
        mock_graph = MagicMock()
        
        # Initialize Executor
        executor = BacktestExecutor(
            graph=mock_graph,
            start_date="2023-01-01",
            end_date="2023-01-05",
            initial_cash=50000.0
        )
        
        # Verify Broker Initialization
        MockBroker.assert_called_once()
        args, kwargs = MockBroker.call_args
        self.assertEqual(kwargs['initial_cash'], 50000.0)
        self.assertTrue("backtest" in kwargs['state_file'])
        
        # Verify DB Clearing
        mock_broker_instance.db_service.clear_backtest_data.assert_called_once()
        mock_broker_instance._save_state.assert_called_once()
        
    @patch('src.backtesting.backtest_executor.PaperBroker')
    def test_run_loop(self, MockBroker):
        # Setup Mock Broker
        mock_broker_instance = MockBroker.return_value
        mock_broker_instance.db_service = MagicMock()
        mock_broker_instance.get_cash.return_value = 50000.0
        mock_broker_instance.get_positions.return_value = {}
        
        # Mock Graph
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            'forecasts': {},
            'performance_summary': pd.DataFrame()
        }
        
        # Initialize Executor
        executor = BacktestExecutor(
            graph=mock_graph,
            start_date="2023-01-01",
            end_date="2023-01-03", # 3 days
            step_days=1
        )
        
        # Run
        initial_state = {"some": "state"}
        results = executor.run(initial_state)
        
        # Verify Graph Invocation Count (3 days: 01, 02, 03)
        self.assertEqual(mock_graph.invoke.call_count, 3)
        
        # Verify State Injection
        call_args = mock_graph.invoke.call_args_list[0][0][0]
        self.assertEqual(call_args['run_type'], 'BACKTEST')
        self.assertEqual(call_args['cutoff_date'], '2023-01-01')
        self.assertEqual(call_args['broker'], mock_broker_instance)

if __name__ == '__main__':
    unittest.main()
