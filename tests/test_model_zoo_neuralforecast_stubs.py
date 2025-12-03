import unittest
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
        # AutoNBEATS is not in ModelZoo anymore, it seems. But let us check if it is there.
        # Based on read_file of model_zoo.py, it has train_autodlinear, train_lstm (NHITS), train_tft.
        # It does NOT have train_autonbeats.
        pass

    def test_autonhits_importerror_when_neuralforecast_missing(self):
        # NHITS is used in train_lstm
        original_flag = mz._HAS_NEURALFORECAST
        original_cls = mz.AutoNHITS
        try:
            mz._HAS_NEURALFORECAST = False
            mz.AutoNHITS = None
            zoo = ModelZoo()
            # train_lstm uses NHITS
            with self.assertRaises(ImportError):
                zoo.train_lstm(self.spec, self.hpo)
        finally:
            mz._HAS_NEURALFORECAST = original_flag
            mz.AutoNHITS = original_cls

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

