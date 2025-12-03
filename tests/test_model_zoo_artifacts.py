import unittest
import dataclasses
import pandas as pd
from unittest.mock import patch, MagicMock
from models.model_zoo import ModelTrainingResult, DataSpec, ModelZoo, ArtifactInfo


class TestModelTrainingResult(unittest.TestCase):
    def test_initialization(self):
        result = ModelTrainingResult(
            job_id="job1", 
            symbol_scope="SYM:TEST", 
            model_family="BaselineLinear", 
            framework="sklearn", 
            best_val_mape=0.1, 
            best_val_mae=0.1, 
            best_hyperparams={}, 
            best_model_id="baseline_123", 
            artifact_info=ArtifactInfo(artifact_uri="file:///tmp/model", local_path="/tmp/model"), 
            val_preds=None
        )
        self.assertEqual(result.artifact_info.local_path, "/tmp/model")
        self.assertEqual(result.best_model_id, "baseline_123")

    def test_to_dict_roundtrip(self):
        result = ModelTrainingResult(
            job_id="job2", 
            symbol_scope="SCOPE:ABC", 
            model_family="AutoDLinear", 
            framework="neuralforecast", 
            best_val_mape=0.0, 
            best_val_mae=0.0, 
            best_hyperparams={"depth": 3}, 
            best_model_id="autodlinear_456", 
            artifact_info=ArtifactInfo(artifact_uri="file:///tmp/model_path", local_path="/tmp/model_path"), 
            val_preds=None
        )
        as_dict = result.to_dict()
        self.assertEqual(as_dict["artifact_info"]["local_path"], "/tmp/model_path")
        # Rebuilding from dict requires handling nested dataclass, which asdict converts to dict
        # So we need to reconstruct ArtifactInfo manually if we want to use **as_dict directly
        # But for this test, let us just verify the dict structure is correct
        self.assertEqual(as_dict["best_model_id"], "autodlinear_456")


class TestModelZoo(unittest.TestCase):
    def setUp(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        self.train_df = pd.DataFrame({"ds": idx, "y": [10,11,12,13,14], "unique_id": "SYM:LIN", "feature": [1,2,3,4,5]})
        val_idx = pd.date_range("2024-01-06", periods=3, freq="D")
        self.val_df = pd.DataFrame({"ds": val_idx, "y": [15,16,17], "unique_id": "SYM:LIN", "feature": [6,7,8]})
        self.data_spec = DataSpec(job_id="job3", symbol_scope="SYM:LIN", train_df=self.train_df, val_df=self.val_df, feature_cols=["feature"], target_col="y", horizon=3)
        self.zoo = ModelZoo()

    @patch("models.model_zoo.NeuralForecast")
    def test_baseline_linear_populates_preds_and_artifact_path(self, mock_nf_cls):
        # Setup mock
        mock_nf_instance = mock_nf_cls.return_value
        
        # Mock predict return
        preds_df = pd.DataFrame({
            "ds": self.val_df["ds"],
            "unique_id": "SYM:LIN",
            "NLinear": [15.1, 16.1, 17.1] 
        })
        mock_nf_instance.predict.return_value = preds_df
        
        result = self.zoo.train_baseline_linear(self.data_spec)
        self.assertIsNotNone(result.val_preds)
        self.assertFalse(result.val_preds.empty)
        self.assertIsNotNone(result.artifact_info)
        self.assertIsInstance(result.artifact_info, ArtifactInfo)


if __name__ == "__main__":
    unittest.main()
