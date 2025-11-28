import unittest
import pandas as pd
from models.model_zoo import ModelZoo, DataSpec

class TestComputeValMapeErrors(unittest.TestCase):
    def setUp(self):
        # Train frames
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        self.train_df = pd.DataFrame({"feat": [1,2,3,4,5], "target": [10,11,12,13,14]}, index=idx)
        # Validation frames
        val_idx = pd.date_range("2024-01-06", periods=3, freq="D")
        self.val_df = pd.DataFrame({"feat": [6,7,8], "target": [15,16,17]}, index=val_idx)
        self.spec = DataSpec(job_id="job-mape", symbol_scope="SYM:MAPE", train_df=self.train_df, val_df=self.val_df, feature_cols=["feat"], target_col="target", horizon=3)
        self.zoo = ModelZoo()
        # Build frames using private helper
        _, _, self.val_nf, _ = self.zoo._prepare_nf_frames(self.spec)

    def test_compute_val_mape_raises_on_empty_predictions(self):
        empty_preds = pd.DataFrame(columns=["unique_id","ds","EmptyModel"])  # no rows
        with self.assertRaises(ValueError) as ctx:
            self.zoo._compute_val_mape(empty_preds, self.val_nf, "EmptyModel")
        self.assertIn("Prediction dataframe empty", str(ctx.exception))

    def test_compute_val_mape_raises_on_no_overlap(self):
        # Predictions with ds entirely outside validation ds range
        preds = pd.DataFrame({
            "unique_id": ["SYM_MAPE"]*3,
            "ds": pd.date_range("2023-12-20", periods=3, freq="D"),
            "SomeModel": [1.0,2.0,3.0]
        })
        with self.assertRaises(ValueError) as ctx:
            self.zoo._compute_val_mape(preds, self.val_nf, "SomeModel")
        self.assertIn("No overlap", str(ctx.exception))

if __name__ == "__main__":
    unittest.main()
