import unittest
from unittest.mock import patch, MagicMock
from models.model_zoo import ModelZoo, DataSpec
import pandas as pd
import numpy as np

class TestModelZooCoreFamilies(unittest.TestCase):
    def setUp(self):
        idx = pd.date_range("2024-01-01", periods=60, freq="h")
        y = np.linspace(100, 110, num=len(idx)) + np.random.default_rng(0).normal(0, 0.5, len(idx))
        frame = pd.DataFrame({"ds": idx, "y": y, "unique_id": "SYN", "feat": np.random.random(len(idx))})
        self.ds = DataSpec(
            job_id="test_job",
            symbol_scope="SYN",
            train_df=frame.iloc[:40], 
            val_df=frame.iloc[40:], 
            feature_cols=["feat"],
            target_col="y",
            horizon=20
        )
        self.mz = ModelZoo()

    def _assert_result(self, result):
        self.assertIsNotNone(result.best_val_mape)
        self.assertIsNotNone(result.best_val_mae)
        d = result.to_dict()
        self.assertIn("best_hyperparams", d)

    @patch("models.model_zoo.NeuralForecast")
    def test_nhits(self, mock_nf_cls):
        mock_nf = mock_nf_cls.return_value
        mock_nf.predict.return_value = pd.DataFrame({
            "ds": self.ds.val_df["ds"],
            "unique_id": "SYN",
            "NHITS": [100.0] * len(self.ds.val_df)
        })
        
        # train_autonhits uses NHITS (aliased to AutoNHITS)
        res = self.mz.train_autonhits(self.ds)
        self.assertEqual(res.model_family, "AutoNHITS")
        self._assert_result(res)

    @patch("models.model_zoo.NeuralForecast")
    def test_tft(self, mock_nf_cls):
        mock_nf = mock_nf_cls.return_value
        mock_nf.predict.return_value = pd.DataFrame({
            "ds": self.ds.val_df["ds"],
            "unique_id": "SYN",
            "TFT": [100.0] * len(self.ds.val_df)
        })

        res = self.mz.train_tft(self.ds)
        self.assertEqual(res.model_family, "TFT")
        self._assert_result(res)

    @patch("models.model_zoo.NeuralForecast")
    def test_autodlinear(self, mock_nf_cls):
        mock_nf = mock_nf_cls.return_value
        mock_nf.predict.return_value = pd.DataFrame({
            "ds": self.ds.val_df["ds"],
            "unique_id": "SYN",
            "DLinear": [100.0] * len(self.ds.val_df)
        })

        res = self.mz.train_autodlinear(self.ds)
        self.assertEqual(res.model_family, "AutoDLinear")
        self._assert_result(res)

    def test_baseline_linear(self):
        # BaselineLinear uses sklearn, no need to mock NeuralForecast
        res = self.mz.train_baseline_linear(self.ds)
        self.assertEqual(res.model_family, "BaselineLinear")
        self._assert_result(res)

class TestModelZooFamilyHelpers(unittest.TestCase):
    def test_family_lists(self):
        mz = ModelZoo()
        families = mz.get_core_model_families()
        self.assertIn("AutoNHITS", families)
        self.assertIn("AutoTFT", families)
        self.assertIn("AutoDLinear", families)
        self.assertIn("BaselineLinear", families)

if __name__ == "__main__":
    unittest.main()
