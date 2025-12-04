import unittest
import pandas as pd
import numpy as np
from src.agents.risk_management_agent import RiskManagementAgent

class TestRiskManagementAgent(unittest.TestCase):
    def setUp(self):
        self.config = {
            'max_portfolio_var': 0.05,
            'confidence_level': 0.95,
            'max_volatility': 0.30
        }
        self.agent = RiskManagementAgent(self.config)
        
        # Create deterministic data
        # 2 assets, perfectly correlated for simplicity in some tests, or uncorrelated
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Asset A: Steady growth, low vol
        self.data_a = pd.DataFrame({
            'close': np.linspace(100, 110, 100) * (1 + np.random.normal(0, 0.01, 100))
        }, index=dates)
        
        # Asset B: High vol
        self.data_b = pd.DataFrame({
            'close': np.linspace(100, 110, 100) * (1 + np.random.normal(0, 0.05, 100))
        }, index=dates)
        
        self.raw_data = {
            'A': self.data_a,
            'B': self.data_b
        }

    def test_var_calculation_correctness(self):
        """Test that VaR is calculated correctly for a single asset."""
        # For a single asset, Portfolio VaR should match Asset VaR
        positions = {'A': 1.0}
        result = self.agent.assess_portfolio_risk(positions, self.raw_data)
        
        returns = self.data_a['close'].pct_change().dropna()
        expected_var = np.percentile(returns, 5) # 95% confidence -> 5th percentile
        
        self.assertAlmostEqual(result['metrics']['VaR_95'], expected_var, places=4)

    def test_var_scaling(self):
        """Test that VaR scales with position size (roughly, for linear approximation)."""
        # Note: In this agent, weights are normalized to 1.0 inside assess_portfolio_risk
        # So passing {'A': 0.5} is treated same as {'A': 1.0} if it's the only asset.
        # To test scaling, we need to check if the agent handles cash/leverage?
        # The current implementation normalizes weights: weights = weights / np.sum(weights)
        # So it assumes fully invested portfolio of the provided assets.
        # This test verifies that behavior.
        
        res1 = self.agent.assess_portfolio_risk({'A': 1.0}, self.raw_data)
        res2 = self.agent.assess_portfolio_risk({'A': 0.5}, self.raw_data)
        
        self.assertAlmostEqual(res1['metrics']['VaR_95'], res2['metrics']['VaR_95'])

    def test_risk_rejection(self):
        """Test that high risk portfolios are rejected."""
        # Asset B has high vol, should trigger rejection if limit is tight
        
        # Create a very volatile asset
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        volatile_data = pd.DataFrame({
            'close': 100 * (1 + np.random.normal(0, 0.10, 100)) # 10% daily vol
        }, index=dates)
        
        raw_data = {'VOL': volatile_data}
        
        # 10% daily vol -> VaR 95% approx -16%
        # Limit is 5%
        
        result = self.agent.assess_portfolio_risk({'VOL': 1.0}, raw_data)
        self.assertFalse(result['risk_approved'])
        self.assertIn("VaR", result['reason'])

    def test_risk_acceptance(self):
        """Test that low risk portfolios are accepted."""
        # Asset A is low vol
        result = self.agent.assess_portfolio_risk({'A': 1.0}, self.raw_data)
        self.assertTrue(result['risk_approved'])

if __name__ == '__main__':
    unittest.main()
