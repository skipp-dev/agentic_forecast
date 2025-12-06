"""
Regression Test Suite

Runs a backtest using the Golden Dataset and asserts performance stability.
"""

import pytest
import pandas as pd
import os
import sys
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.golden_dataset_manager import GoldenDatasetManager
from src.agents.strategy_backtesting_agent import StrategyBacktestingAgent
# Mock graph for testing
from unittest.mock import MagicMock

class TestRegression:
    
    def setup_method(self):
        self.manager = GoldenDatasetManager()
        
    def test_golden_dataset_existence(self):
        """
        Ensure golden dataset exists. If not, warn user to generate it.
        """
        if not self.manager.metadata_file.exists():
            pytest.skip("Golden Dataset not generated. Run 'python scripts/generate_golden_dataset.py' first.")
            
        import json
        with open(self.manager.metadata_file, 'r') as f:
            meta = json.load(f)
            
        assert "symbols" in meta
        assert len(meta["symbols"]) > 0
        
    def test_backtest_stability(self):
        """
        Run a backtest on golden data and check metrics.
        """
        if not self.manager.metadata_file.exists():
            pytest.skip("Golden Dataset not generated.")
            
        # Load metadata
        import json
        with open(self.manager.metadata_file, 'r') as f:
            meta = json.load(f)
            
        symbols = meta["symbols"]
        start_date = meta["start_date"]
        end_date = meta["end_date"]
        
        # Mock the Graph to return deterministic results based on data
        # In a real integration test, we would load the actual LangGraph
        # For this example, we assume StrategyBacktestingAgent can accept a mock graph
        # or we use a simplified graph that just runs the model.
        
        # TODO: Instantiate the real graph here
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "signals": {s: "BUY" for s in symbols}, # Dummy signal
            "portfolio": {"cash": 100000, "value": 100000}
        }
        
        agent = StrategyBacktestingAgent(graph=mock_graph)
        
        # Inject Golden Data into the agent/executor
        # This requires the BacktestExecutor to support a custom data source
        # or we patch the DataPipeline to return golden data.
        
        # Patch DataPipeline.fetch_stock_data to use GoldenDatasetManager.load_data
        from unittest.mock import patch
        
        with patch('src.data_pipeline.DataPipeline.fetch_stock_data') as mock_fetch:
            mock_fetch.side_effect = lambda symbol, **kwargs: self.manager.load_data(symbol)
            
            # Run Backtest
            results = agent.run_backtest(start_date, end_date)
            
            # Extract Metrics
            metrics = results.get('metrics', {})
            
            # Validate against baseline
            passed = self.manager.validate_current_performance(metrics)
            
            assert passed, "Regression detected in backtest metrics!"
