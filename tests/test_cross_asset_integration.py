
import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.agents.feature_agent import FeatureAgent
from src.nodes.agent_nodes import feature_agent_node
from src.graphs.state import GraphState

class TestCrossAssetIntegration(unittest.TestCase):
    def setUp(self):
        # Create synthetic data for 2 symbols
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Symbol 1: Uptrend
        df1 = pd.DataFrame(index=dates)
        df1['close'] = np.linspace(100, 200, 100) + np.random.normal(0, 2, 100)
        df1['open'] = df1['close']
        df1['high'] = df1['close']
        df1['low'] = df1['close']
        df1['volume'] = 1000
        
        # Symbol 2: Downtrend (Negative correlation)
        df2 = pd.DataFrame(index=dates)
        df2['close'] = np.linspace(200, 100, 100) + np.random.normal(0, 2, 100)
        df2['open'] = df2['close']
        df2['high'] = df2['close']
        df2['low'] = df2['close']
        df2['volume'] = 1000
        
        self.raw_data = {
            'SYM1': df1,
            'SYM2': df2
        }
        
        self.state = GraphState(
            raw_data=self.raw_data,
            features={},
            regimes={},
            historical_regimes={},
            macro_data={},
            news_features={},
            cutoff_date=None
        )

    def test_cross_asset_features_generation(self):
        """Test that cross-asset features are generated when multiple symbols exist."""
        
        # Run the feature agent node
        # This should trigger the logic I added to agent_nodes.py
        new_state = feature_agent_node(self.state)
        
        features = new_state['features']
        
        self.assertIn('SYM1', features)
        self.assertIn('SYM2', features)
        
        # Check for cross-asset columns in the first row of data
        # features is a dict of dicts (index -> columns)
        sym1_feats = pd.DataFrame.from_dict(features['SYM1'], orient='index')
        sym2_feats = pd.DataFrame.from_dict(features['SYM2'], orient='index')
        
        print("SYM1 Columns:", sym1_feats.columns.tolist())
        
        # Check for specific columns
        expected_cols = ['market_correlation_20', 'relative_strength_1d', 'beta_60']
        for col in expected_cols:
            self.assertIn(col, sym1_feats.columns, f"Missing {col} in SYM1")
            self.assertIn(col, sym2_feats.columns, f"Missing {col} in SYM2")
            
        # Check values
        # Since they are negatively correlated, correlation should be negative
        # We need to check the last few rows as rolling windows need data
        last_corr = sym1_feats['market_correlation_20'].iloc[-1]
        print(f"Market Correlation for SYM1 (should be positive vs market index?): {last_corr}")
        
        # Market index is (SYM1 + SYM2) / 2. 
        # SYM1 goes up, SYM2 goes down. Market might be flat or noisy.
        # If Market is flat, correlation might be weird.
        # But the point is that the columns exist and are populated.
        
        self.assertFalse(sym1_feats['beta_60'].isnull().all(), "Beta should not be all NaN")

if __name__ == '__main__':
    unittest.main()
