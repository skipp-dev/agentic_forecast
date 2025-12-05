
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.nodes.macro_nodes import regime_detection_node
from src.agents.decision_agent import DecisionAgent
from src.graphs.state import GraphState

class TestRegimeIntegration(unittest.TestCase):
    def setUp(self):
        self.mock_macro_data = {
            'raw_data': {
                'INFLATION': pd.Series([0.05] * 100, index=pd.date_range(start='2023-01-01', periods=100)),
                'UNEMPLOYMENT': pd.Series([0.04] * 100, index=pd.date_range(start='2023-01-01', periods=100)),
                'GDP_GROWTH': pd.Series([0.02] * 100, index=pd.date_range(start='2023-01-01', periods=100)),
                'VIX': pd.Series([25.0] * 100, index=pd.date_range(start='2023-01-01', periods=100)) # High Volatility
            }
        }
        self.state = GraphState(
            macro_data=self.mock_macro_data,
            regimes={},
            errors=[],
            config={}
        )

    @patch('src.agents.regime_agent.RegimeAgent.detect_regime')
    def test_regime_detection_flow(self, mock_detect):
        # Setup mock return with standard keys
        mock_detect.return_value = {
            'rate_regime': 'high_rates',
            'oil_regime': 'normal_oil',
            'volatility_regime': 'high_volatility' # New key we want to implement
        }

        # Run the node
        new_state = regime_detection_node(self.state)

        # Verify state update
        self.assertIn('rate_regime', new_state['regimes'])
        self.assertEqual(new_state['regimes']['rate_regime'], 'high_rates')

    def test_decision_agent_regime_impact(self):
        # Test if DecisionAgent actually uses the regime
        agent = DecisionAgent({})
        
        # Case 1: Normal Regime
        regimes_normal = {
            'rate_regime': 'normal_rates',
            'volatility_regime': 'normal_volatility'
        }
        decision_normal = agent.get_trading_decision('AAPL', trust_score=0.8, regimes=regimes_normal)
        
        # Case 2: High Volatility Regime
        regimes_volatile = {
            'rate_regime': 'normal_rates',
            'volatility_regime': 'high_volatility'
        }
        decision_volatile = agent.get_trading_decision('AAPL', trust_score=0.8, regimes=regimes_volatile)

        # We expect the position size or confidence to be lower in the volatile regime
        print(f"Normal Multiplier: {decision_normal.get('position_size_multiplier')}")
        print(f"Volatile Multiplier: {decision_volatile.get('position_size_multiplier')}")
        
        self.assertLess(
            decision_volatile.get('position_size_multiplier', 1.0),
            decision_normal.get('position_size_multiplier', 1.0),
            "Decision Agent should reduce position size in High Volatility regime"
        )

if __name__ == '__main__':
    unittest.main()
