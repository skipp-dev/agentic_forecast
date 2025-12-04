import unittest
from unittest.mock import MagicMock
import pandas as pd
import numpy as np
from src.agents.portfolio_manager_agent import PortfolioManagerAgent

class TestPortfolioManagerAgent(unittest.TestCase):
    def setUp(self):
        self.config = {
            'max_position_size': 0.2,
            'target_cash': 0.1, # 10% cash
            'risk': {
                'max_portfolio_var': 0.05
            }
        }
        self.agent = PortfolioManagerAgent(self.config)
        
        # Mock Risk Agent to always approve unless specified
        self.agent.risk_agent.assess_portfolio_risk = MagicMock(return_value={
            'risk_approved': True,
            'metrics': {'VaR_95': -0.01},
            'reason': 'OK'
        })
        
        self.raw_data = {'AAPL': pd.DataFrame(), 'GOOG': pd.DataFrame()}

    def test_position_limits(self):
        """Test that individual position limits are respected."""
        # Recommend huge position
        actions = ["Promote BaselineLinear for AAPL (Trust: 0.9, Size: 5.0x)"]
        best_models = {'AAPL': {'model_family': 'BaselineLinear'}}
        
        result = self.agent.construct_portfolio(actions, best_models, self.raw_data)
        weights = result['target_weights']
        
        # Should be capped at 0.2
        self.assertLessEqual(weights['AAPL'], 0.2)

    def test_target_cash(self):
        """Test that portfolio leaves room for target cash."""
        # Recommend many assets
        actions = [
            "Promote M1 for AAPL (Size: 1.0x)",
            "Promote M2 for GOOG (Size: 1.0x)",
            "Promote M3 for MSFT (Size: 1.0x)",
            "Promote M4 for AMZN (Size: 1.0x)",
            "Promote M5 for TSLA (Size: 1.0x)"
        ]
        best_models = {
            'AAPL': {}, 'GOOG': {}, 'MSFT': {}, 'AMZN': {}, 'TSLA': {}
        }
        
        # Mock data for all
        raw_data = {k: pd.DataFrame() for k in best_models}
        
        result = self.agent.construct_portfolio(actions, best_models, raw_data)
        weights = result['target_weights']
        
        total_weight = sum(weights.values())
        # Should be <= 1.0 - target_cash (0.9)
        self.assertLessEqual(total_weight, 0.900001)

    def test_risk_rejection_fallback(self):
        """Test fallback logic when risk agent rejects."""
        # Mock rejection
        self.agent.risk_agent.assess_portfolio_risk = MagicMock(side_effect=[
            {'risk_approved': False, 'reason': 'Too risky'}, # First call rejects
            {'risk_approved': True, 'reason': 'OK'}          # Second call (after scaling) accepts
        ])
        
        actions = ["Promote M1 for AAPL (Size: 1.0x)"]
        best_models = {'AAPL': {}}
        
        result = self.agent.construct_portfolio(actions, best_models, self.raw_data)
        
        # Verify assess_portfolio_risk was called twice
        self.assertEqual(self.agent.risk_agent.assess_portfolio_risk.call_count, 2)
        
        # Verify weights were scaled down (fallback logic is 50% reduction)
        # Initial weight for 1.0x is 0.1. Scaled to (1-cash)/total -> 0.9/0.1 = 9x? 
        # Wait, logic is: weight = 0.1 * mult. 
        # Then scale_factor = (1-cash) / total.
        # If total < 1-cash, we don't scale UP usually?
        # Let's check implementation:
        # if total_weight > 0: scale_factor = (1.0 - self.target_cash) / total_weight
        # It scales UP or DOWN to match target exposure exactly?
        # "Normalize weights to sum to (1 - target_cash)"
        # Yes, it scales to fill the bucket.
        
        # So initial: AAPL=0.1. Total=0.1. Target=0.9. Scale=9.
        # AAPL becomes 0.9. But max_position_size is 0.2.
        # So AAPL becomes 0.2.
        
        # Then Risk Agent rejects.
        # Fallback: "Reduce exposure by 50%"
        # AAPL becomes 0.1.
        
        self.assertAlmostEqual(result['target_weights']['AAPL'], 0.1)

if __name__ == '__main__':
    unittest.main()
