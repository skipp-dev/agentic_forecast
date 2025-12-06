
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from src.pipeline_orchestrator import run_pipeline
from src.core.run_context import RunContext, RunType

class TestGoldenPath(unittest.TestCase):
    def setUp(self):
        # Load golden data
        self.golden_df = pd.read_csv('tests/data/golden_aapl.csv', index_col='date', parse_dates=True)
        self.symbols = ['AAPL']
        self.config = {
            'alpha_vantage': {'api_key': 'mock'},
            'orchestrator': {'dynamic_routing_enabled': False},
            'model': {'type': 'statistical'} # Use simple models to avoid GPU/heavy compute
        }

    @patch('src.nodes.reporting_nodes.generate_report_node')
    @patch('src.nodes.agent_nodes.forecast_agent_node')
    @patch('src.nodes.data_nodes_optimized.load_data_node')
    @patch('src.nodes.agent_nodes.market_calendar_node')
    def test_pipeline_integration(self, mock_market_calendar, mock_load_data, mock_forecast_agent, mock_reporting):
        # Setup mock market calendar
        def market_calendar_side_effect(state):
            state['market_status'] = {'is_trading_day': True}
            return state
        mock_market_calendar.side_effect = market_calendar_side_effect

        # Setup mock data loader
        def load_data_side_effect(state):
            state['raw_data'] = {'AAPL': self.golden_df}
            return state
        mock_load_data.side_effect = load_data_side_effect

        # Setup mock forecast agent to set interpreted_forecasts (Definition of Done for Supervisor)
        def forecast_agent_side_effect(state):
            state['interpreted_forecasts'] = {'AAPL': {'action': 'HOLD', 'confidence': 'High'}}
            return state
        mock_forecast_agent.side_effect = forecast_agent_side_effect

        # Setup mock reporting to just pass through
        def reporting_side_effect(state):
            state['report_generated'] = True
            return state
        mock_reporting.side_effect = reporting_side_effect

        # Create context
        from datetime import datetime
        ctx = RunContext(run_id="test_golden", run_type=RunType.DAILY, started_at=datetime.now())
        
        # Run pipeline using graph directly to get state
        from src.graphs.main_graph import create_main_graph
        from src.pipeline_orchestrator import build_initial_state
        
        app = create_main_graph(self.config)
        initial_state = build_initial_state(self.symbols, self.config, ctx)
        
        try:
            final_state = app.invoke(initial_state, config={"recursion_limit": 100})
        except Exception as e:
            self.fail(f"Pipeline failed: {e}")

        # Assertions
        self.assertIn('AAPL', final_state['raw_data'])

if __name__ == '__main__':
    unittest.main()
