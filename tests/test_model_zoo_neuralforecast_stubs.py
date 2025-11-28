import unittest
import os
import shutil
from pathlib import Path

import pandas as pd

import src.models.model_zoo as mz
from models.model_zoo import ModelZoo, DataSpec, HPOConfig, ArtifactInfo


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
        original_flag = mz._HAS_NEURALFORECAST
        original_cls = mz.AutoDLinear
        try:
            mz._HAS_NEURALFORECAST = False
            mz.AutoDLinear = None
            zoo = ModelZoo()
            with self.assertRaises(ImportError):
                zoo.train_autodlinear(self.spec, self.hpo)
        finally:
            mz._HAS_NEURALFORECAST = original_flag
            mz.AutoDLinear = original_cls

    def test_autonbeats_importerror_when_neuralforecast_missing(self):
        original_flag = mz._HAS_NEURALFORECAST
        original_cls = mz.AutoNBEATS
        try:
            mz._HAS_NEURALFORECAST = False
            mz.AutoNBEATS = None
            zoo = ModelZoo()
            with self.assertRaises(ImportError):
                zoo.train_autonbeats(self.spec, self.hpo)
        finally:
            mz._HAS_NEURALFORECAST = original_flag
            mz.AutoNBEATS = original_cls

    def test_autonhits_importerror_when_neuralforecast_missing(self):
        original_flag = mz._HAS_NEURALFORECAST
        original_cls = mz.AutoNHITS
        try:
            mz._HAS_NEURALFORECAST = False
            mz.AutoNHITS = None
            zoo = ModelZoo()
            with self.assertRaises(ImportError):
                zoo.train_autonhits(self.spec, self.hpo)
        finally:
            mz._HAS_NEURALFORECAST = original_flag
            mz.AutoNHITS = original_cls

    def test_persist_nf_model_returns_artifact_info(self):
        zoo = ModelZoo()
        dummy_model = "dummy_model_content"
        model_id = "dummy123"
        artifact = zoo._persist_nf_model(model_id=model_id, model_obj=dummy_model, model_family="DummyFam")
        
        self.assertIsInstance(artifact, ArtifactInfo)
        self.assertTrue(artifact.local_path.endswith(model_id))
        self.assertTrue(Path(artifact.local_path).exists())
        self.assertTrue(artifact.artifact_uri.startswith("file://"))


if __name__ == "__main__":
    unittest.main()
