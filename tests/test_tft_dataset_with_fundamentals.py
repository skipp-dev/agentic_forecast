import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from models.model_zoo import ModelZoo, DataSpec, HPOConfig

class TestTFTDatasetWithFundamentals(unittest.TestCase):
    def setUp(self):
        self.zoo = ModelZoo()
        
    @patch('models.model_zoo.AutoTFT')
    @patch('models.model_zoo.NeuralForecast')
    def test_tft_initialization_with_fundamentals(self, mock_nf, mock_autotft):
        # Create dummy data with fundamentals
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'ds': dates,
            'y': np.random.rand(100),
            'pe_ttm': np.random.rand(100),
            'revenue_growth': np.random.rand(100),
            'sector': ['Tech'] * 100
        })
        
        # Create DataSpec
        data_spec = DataSpec(
            target_col='y',
            date_col='ds',
            train_df=df.iloc[:80],
            val_df=df.iloc[80:],
            horizon=10,
            exog_cols=['pe_ttm', 'revenue_growth', 'sector']
        )
        
        # Mock HPO config
        hpo_config = HPOConfig(n_trials=1)
        
        # Call train_autotft
        # We need to ensure _HAS_NEURALFORECAST is True or we mock it
        with patch('models.model_zoo._HAS_NEURALFORECAST', True):
            self.zoo.train_autotft(data_spec, hpo_config)
            
        # Verify AutoTFT was initialized with futr_exog_list
        mock_autotft.assert_called_once()
        call_kwargs = mock_autotft.call_args[1]
        
        self.assertIn('futr_exog_list', call_kwargs)
        self.assertEqual(call_kwargs['futr_exog_list'], ['pe_ttm', 'revenue_growth', 'sector'])
        self.assertEqual(call_kwargs['h'], 10)

if __name__ == '__main__':
    unittest.main()
