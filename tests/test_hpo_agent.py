import unittest
from unittest.mock import MagicMock
from agents.hyperparameter_search_agent import HyperparameterSearchAgent
from models.model_zoo import ModelZoo, DataSpec
import pandas as pd
import numpy as np

class TestHPOAgentRiskMode(unittest.TestCase):
    def setUp(self):
        idx = pd.date_range('2024-02-01', periods=72, freq='h')
        y = 50 + np.sin(np.linspace(0, 10, len(idx)))
        frame = pd.DataFrame({'ds': idx, 'y': y, 'unique_id': 'TEST'})
        self.data_spec = DataSpec(
            train_df=frame.iloc[:48],
            val_df=frame.iloc[48:],
            test_df=frame.iloc[48:],
            symbol='TEST',
            horizon=24
        )

    def test_risk_mode_family_injection(self):
        # Mock ModelZoo
        mock_zoo = MagicMock(spec=ModelZoo)
        
        # Mock results
        mock_res_nhits = MagicMock()
        mock_res_nhits.model_family = 'AutoNHITS'
        mock_res_nhits.best_val_mape = 0.1
        mock_zoo.train_autonhits.return_value = mock_res_nhits
        
        mock_res_nbeats = MagicMock()
        mock_res_nbeats.model_family = 'AutoNBEATS'
        mock_res_nbeats.best_val_mape = 0.15
        mock_zoo.train_autonbeats.return_value = mock_res_nbeats
        
        mock_res_baseline = MagicMock()
        mock_res_baseline.model_family = 'BaselineLinear'
        mock_res_baseline.best_val_mape = 0.2
        mock_zoo.train_baseline_linear.return_value = mock_res_baseline

        agent = HyperparameterSearchAgent(risk_mode=True, model_zoo=mock_zoo)
        result = agent.run_model_zoo_hpo(symbol='TEST', data_spec=self.data_spec, model_families=['AutoNHITS','AutoNBEATS'])
        
        self.assertTrue(result['success'])
        tried = [r['model_family'] for r in result['all_results']]
        # DeepAR stub should be skipped but AutoNHITS present
        self.assertIn('AutoNHITS', tried)
        # BaselineLinear should appear from risk families injection
        self.assertIn('BaselineLinear', tried)

    def test_best_family_selection(self):
        # Mock ModelZoo
        mock_zoo = MagicMock(spec=ModelZoo)
        
        # Mock results - AutoNHITS is better (lower MAPE)
        mock_res_nhits = MagicMock()
        mock_res_nhits.model_family = 'AutoNHITS'
        mock_res_nhits.best_val_mape = 0.05
        mock_zoo.train_autonhits.return_value = mock_res_nhits
        
        mock_res_nbeats = MagicMock()
        mock_res_nbeats.model_family = 'AutoNBEATS'
        mock_res_nbeats.best_val_mape = 0.15
        mock_zoo.train_autonbeats.return_value = mock_res_nbeats

        agent = HyperparameterSearchAgent(risk_mode=False, model_zoo=mock_zoo)
        result = agent.run_model_zoo_hpo(symbol='XYZ', data_spec=self.data_spec, model_families=['AutoNHITS','AutoNBEATS'])
        
        self.assertTrue(result['success'])
        self.assertEqual(result['best_family'], 'AutoNHITS')
        self.assertEqual(result['best_val_mape'], 0.05)

if __name__ == '__main__':
    unittest.main()
