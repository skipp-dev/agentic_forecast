import unittest
from unittest.mock import patch
import os
import shutil
from pathlib import Path

import pandas as pd

import models.model_zoo as mz
from models.model_zoo import ModelZoo, DataSpec, HPOConfig


class TestNeuralForecastStubs(unittest.TestCase):
    def setUp(self):
        idx = pd.date_range("2024-01-01", periods=4, freq="D")
        val_idx = pd.date_range("2024-01-05", periods=2, freq="D")
        self.train_df = pd.DataFrame({"feat": [1,2,3,4], "target": [10,11,12,13]}, index=idx)
        self.val_df = pd.DataFrame({"feat": [5,6], "target": [14,15]}, index=val_idx)
        self.spec = DataSpec(job_id="job-nf", symbol_scope="SYM:NF", train_df=self.train_df, val_df=self.val_df, feature_cols=["feat"], target_col="target", horizon=2)
        self.hpo = HPOConfig(max_trials=1, max_epochs=1)

    def tearDown(self):
        # Ensure models directory cleaned between tests to avoid residue.
        if Path("models").exists():
            shutil.rmtree("models", ignore_errors=True)

    def test_autodlinear_importerror_when_neuralforecast_missing(self):
        # Simulate absence of neuralforecast
        with patch('models.model_zoo._HAS_NEURALFORECAST', False):
            zoo = ModelZoo()
            with self.assertRaises(ImportError):
                zoo.train_autodlinear(self.spec, self.hpo)

    def test_autonbeats_importerror_when_neuralforecast_missing(self):
        # AutoNBEATS
        with patch('models.model_zoo._HAS_NEURALFORECAST', False):
            zoo = ModelZoo()
            with self.assertRaises(ImportError):
                zoo.train_autonbeats(self.spec, self.hpo)

    def test_autonhits_importerror_when_neuralforecast_missing(self):
        # NHITS
        with patch('models.model_zoo._HAS_NEURALFORECAST', False):
            zoo = ModelZoo()
            with self.assertRaises(ImportError):
                zoo.train_autonhits(self.spec, self.hpo)

    def test_persist_nf_model_returns_artifact_path(self):
        zoo = ModelZoo()
        # Mock model object with save method
        class MockModel:
            def save(self, path):
                os.makedirs(path, exist_ok=True)
                
        dummy_model = MockModel()
        model_id = "dummy123"
        artifact_info = zoo._persist_nf_model(model_id=model_id, model=dummy_model, model_family="DummyFam")
        
        self.assertIsInstance(artifact_info, mz.ArtifactInfo)
        self.assertTrue(artifact_info.artifact_uri.startswith("file://"))
        self.assertTrue(artifact_info.local_path.endswith(model_id))
        self.assertTrue(Path(artifact_info.local_path).exists())


if __name__ == "__main__":
    unittest.main()

