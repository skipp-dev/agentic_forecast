import unittest
import dataclasses
import pandas as pd
from unittest.mock import patch, MagicMock
from models.model_zoo import ArtifactInfo, ModelTrainingResult, DataSpec, ModelZoo


class TestArtifactInfo(unittest.TestCase):
    def test_defaults_and_properties(self):
        empty_info = ArtifactInfo(artifact_uri="", local_path=None)
        self.assertEqual(empty_info.artifact_uri, "")
        self.assertIsNone(empty_info.local_path)

        populated = ArtifactInfo(artifact_uri="file://tmp/run/artifacts/model", local_path="/tmp/run/artifacts/model.pkl")
        self.assertTrue(populated.artifact_uri.startswith("file://"))
        self.assertTrue(populated.local_path.endswith("model.pkl"))

    def test_local_path_none_handling(self):
        info = ArtifactInfo(artifact_uri="mlflow://runs/xyz/artifacts/model", local_path=None)
        result = ModelTrainingResult(job_id="job4", symbol_scope="SYM:NULL", model_family="AutoNHITS", framework="neuralforecast", best_val_mape=0.0, best_val_mae=0.0, best_hyperparams={}, best_model_id="autonhits_789", artifact_info=info, val_preds=None)
        self.assertIsNone(result.local_artifact_path)


class TestModelTrainingResult(unittest.TestCase):
    def test_auto_populates_artifact_info(self):
        result = ModelTrainingResult(job_id="job1", symbol_scope="SYM:TEST", model_family="BaselineLinear", framework="sklearn", best_val_mape=0.1, best_val_mae=0.1, best_hyperparams={}, best_model_id="baseline_123", artifact_info=None, val_preds=None)
        self.assertIsInstance(result.artifact_info, ArtifactInfo)
        self.assertEqual(result.artifact_uri, "")
        self.assertIsNone(result.local_artifact_path)

    def test_to_dict_roundtrip(self):
        info = ArtifactInfo(artifact_uri="mlflow://runs/abc/artifacts/model", local_path=None)
        result = ModelTrainingResult(job_id="job2", symbol_scope="SCOPE:ABC", model_family="AutoDLinear", framework="neuralforecast", best_val_mape=0.0, best_val_mae=0.0, best_hyperparams={"depth": 3}, best_model_id="autodlinear_456", artifact_info=info, val_preds=None)
        as_dict = result.to_dict()
        self.assertEqual(as_dict["artifact_info"]["artifact_uri"], info.artifact_uri)
        self.assertEqual(as_dict["artifact_info"]["local_path"], info.local_path)
        rebuilt = ModelTrainingResult(**{k: v for k, v in as_dict.items() if k != "artifact_info"}, artifact_info=ArtifactInfo(**as_dict["artifact_info"]))
        self.assertEqual(dataclasses.asdict(rebuilt), dataclasses.asdict(result))


class TestModelZoo(unittest.TestCase):
    def setUp(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        self.train_df = pd.DataFrame({"ds": idx, "y": [10,11,12,13,14], "unique_id": "SYM:LIN", "feature": [1,2,3,4,5]})
        val_idx = pd.date_range("2024-01-06", periods=3, freq="D")
        self.val_df = pd.DataFrame({"ds": val_idx, "y": [15,16,17], "unique_id": "SYM:LIN", "feature": [6,7,8]})
        self.data_spec = DataSpec(job_id="job3", symbol_scope="SYM:LIN", train_df=self.train_df, val_df=self.val_df, feature_cols=["feature"], target_col="y", horizon=3)
        self.zoo = ModelZoo()

    @patch('models.model_zoo.NeuralForecast')
    def test_baseline_linear_populates_preds_and_artifact_info(self, mock_nf_cls):
        # Setup mock
        mock_nf_instance = mock_nf_cls.return_value
        
        # Mock predict return
        preds_df = pd.DataFrame({
            'ds': self.val_df['ds'],
            'unique_id': 'SYM:LIN',
            'NLinear': [15.1, 16.1, 17.1] 
        })
        mock_nf_instance.predict.return_value = preds_df
        
        result = self.zoo.train_baseline_linear(self.data_spec)
        self.assertIsNotNone(result.val_preds)
        self.assertFalse(result.val_preds.empty)
        self.assertIsNotNone(result.artifact_info)
        self.assertNotEqual(result.artifact_info.artifact_uri, "")


if __name__ == "__main__":
    unittest.main()
