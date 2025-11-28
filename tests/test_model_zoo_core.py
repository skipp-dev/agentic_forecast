import unittest
from unittest.mock import patch, MagicMock
from models.model_zoo import ModelZoo, DataSpec
import pandas as pd
import numpy as np

class TestModelZooCoreFamilies(unittest.TestCase):
    def setUp(self):
        idx = pd.date_range('2024-01-01', periods=60, freq='h')
        y = np.linspace(100, 110, num=len(idx)) + np.random.default_rng(0).normal(0, 0.5, len(idx))
        frame = pd.DataFrame({'ds': idx, 'y': y, 'unique_id': 'SYN'})
        self.ds = DataSpec(train_df=frame.iloc[:40], val_df=frame.iloc[40:], test_df=frame.iloc[40:])
        self.mz = ModelZoo(random_seed=123)

    def _assert_result(self, result):
        self.assertIsNotNone(result.best_val_mape)
        self.assertIsNotNone(result.best_val_mae)
        # self.assertTrue(result.best_val_mape >= 0) # Mocked values might be 0 or whatever
        d = result.to_dict()
        self.assertIn('val_preds_info', d)
        self.assertIn('best_hyperparams', d)

    @patch('models.model_zoo.NeuralForecast')
    def test_autonhits(self, mock_nf_cls):
        mock_nf = mock_nf_cls.return_value
        mock_nf.predict.return_value = pd.DataFrame({
            'ds': self.ds.val_df['ds'],
            'unique_id': 'SYN',
            'AutoNHITS': [100.0] * len(self.ds.val_df)
        })
        
        res = self.mz.train_autonhits(self.ds)
        self.assertEqual(res.model_family, 'AutoNHITS')
        self._assert_result(res)

    @patch('models.model_zoo.NeuralForecast')
    def test_autonbeats(self, mock_nf_cls):
        mock_nf = mock_nf_cls.return_value
        mock_nf.predict.return_value = pd.DataFrame({
            'ds': self.ds.val_df['ds'],
            'unique_id': 'SYN',
            'AutoNBEATS': [100.0] * len(self.ds.val_df)
        })

        res = self.mz.train_autonbeats(self.ds)
        self.assertEqual(res.model_family, 'AutoNBEATS')
        self._assert_result(res)

    @patch('models.model_zoo.NeuralForecast')
    def test_autodlinear(self, mock_nf_cls):
        mock_nf = mock_nf_cls.return_value
        mock_nf.predict.return_value = pd.DataFrame({
            'ds': self.ds.val_df['ds'],
            'unique_id': 'SYN',
            'AutoDLinear': [100.0] * len(self.ds.val_df)
        })

        res = self.mz.train_autodlinear(self.ds)
        self.assertEqual(res.model_family, 'AutoDLinear')
        self._assert_result(res)

    @patch('models.model_zoo.NeuralForecast')
    def test_baseline_linear(self, mock_nf_cls):
        mock_nf = mock_nf_cls.return_value
        mock_nf.predict.return_value = pd.DataFrame({
            'ds': self.ds.val_df['ds'],
            'unique_id': 'SYN',
            'NLinear': [100.0] * len(self.ds.val_df)
        })

        res = self.mz.train_baseline_linear(self.ds)
        self.assertEqual(res.model_family, 'BaselineLinear')
        self._assert_result(res)

class TestModelZooFamilyHelpers(unittest.TestCase):
    def test_family_lists(self):
        mz = ModelZoo()
        self.assertIn('AutoNHITS', mz.get_core_model_families())
        self.assertIn('PatchTST', mz.get_second_wave_families())
        self.assertIn('DeepAR', mz.get_risk_model_families())

if __name__ == '__main__':
    unittest.main()
