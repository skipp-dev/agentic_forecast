import unittest
import numpy as np
import pandas as pd
from models.model_zoo import ModelZoo, DataSpec

class TestComputeValMapeSuccess(unittest.TestCase):
    def setUp(self):
        # Create train and validation data
        idx_train = pd.date_range("2024-01-01", periods=5, freq="D")
        idx_val = pd.date_range("2024-01-06", periods=4, freq="D")
        self.train_df = pd.DataFrame({"feat": [1,2,3,4,5], "target": [10,12,11,13,15]}, index=idx_train)
        self.val_df = pd.DataFrame({"feat": [6,7,8,9], "target": [16,18,17,19]}, index=idx_val)
        self.spec = DataSpec(job_id="job-mape-ok", symbol_scope="SYM:MAPEOK", train_df=self.train_df, val_df=self.val_df, feature_cols=["feat"], target_col="target", horizon=4)
        self.zoo = ModelZoo()
        # Prepare NF frames to get val_nf
        _, _, self.val_nf, _ = self.zoo._prepare_nf_frames(self.spec)

    def test_success_mape(self):
        # Build a predictions frame overlapping by ds values
        # Suppose model produces target * 0.95 as forecast
        preds = pd.DataFrame({
            "unique_id": self.val_nf["unique_id"],
            "ds": self.val_nf["ds"],
            "MyModel": (self.val_nf["y"] * 0.95).to_numpy(),
        })
        mape = self.zoo._compute_val_mape(preds, self.val_nf, "MyModel")
        # Expected MAPE manual calculation
        expected = np.mean(np.abs(self.val_nf["y"].to_numpy() - preds["MyModel"].to_numpy()) / self.val_nf["y"].to_numpy())
        self.assertAlmostEqual(mape, expected, places=10)


class TestExtractBestParams(unittest.TestCase):
    class FakeTrial:
        def __init__(self):
            # Contains numpy types that should be converted to Python/native forms
            self.config = {
                "learning_rate": np.float64(0.01),
                "layers": np.array([32, 64, 128]),
                "activation": "relu",
            }
    class FakeTrials:
        def __init__(self):
            self.best_trial = TestExtractBestParams.FakeTrial()
    class FakeModel:
        def __init__(self):
            self.trials = TestExtractBestParams.FakeTrials()

    def test_extract_best_params_conversion(self):
        zoo = ModelZoo()
        fake_model = self.FakeModel()
        params = zoo._extract_best_params(fake_model)
        self.assertIn("learning_rate", params)
        self.assertEqual(params["learning_rate"], 0.01)
        self.assertIn("layers", params)
        self.assertIsInstance(params["layers"], list)
        self.assertEqual(params["layers"], [32, 64, 128])
        self.assertEqual(params["activation"], "relu")

if __name__ == "__main__":
    unittest.main()
