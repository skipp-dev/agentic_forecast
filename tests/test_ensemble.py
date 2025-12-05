import unittest
from unittest.mock import MagicMock
from src.agents.hyperparameter_search_agent import HyperparameterSearchAgent
from models.model_zoo import ModelZoo, DataSpec
import pandas as pd
import numpy as np

class TestEnsembleCombinations(unittest.TestCase):
    def setUp(self):
        idx = pd.date_range("2024-04-01", periods=72, freq="h")
        y = 50 + np.cos(np.linspace(0, 8, len(idx)))
        frame = pd.DataFrame({"ds": idx, "y": y, "unique_id": "ENS", "feat": np.random.random(len(idx))})
        self.data_spec = DataSpec(
            job_id="test_job",
            symbol_scope="ENS",
            train_df=frame.iloc[:48],
            val_df=frame.iloc[48:],
            feature_cols=["feat"],
            target_col="y",
            horizon=24
        )

    def test_ensemble_two_models(self):
        # Mock ModelZoo
        mock_zoo = MagicMock(spec=ModelZoo)
        
        # Mock results
        mock_res_nhits = MagicMock()
        mock_res_nhits.val_preds = pd.DataFrame({
            "ds": self.data_spec.val_df["ds"],
            "AutoNHITS": self.data_spec.val_df["y"] * 1.1  # 10% error
        })
        mock_zoo.train_autonhits.return_value = mock_res_nhits
        
        mock_res_nbeats = MagicMock()
        mock_res_nbeats.val_preds = pd.DataFrame({
            "ds": self.data_spec.val_df["ds"],
            "AutoNBEATS": self.data_spec.val_df["y"] * 0.9  # -10% error
        })
        mock_zoo.train_autonbeats.return_value = mock_res_nbeats
        
        agent = HyperparameterSearchAgent(model_zoo=mock_zoo)
        result = agent.test_ensemble_combinations(symbol="ENS", data_spec=self.data_spec, families=["AutoNHITS","AutoNBEATS"])
        
        self.assertTrue(result["success"])
        self.assertIn("ensemble_mape", result)
        # Ensemble of 1.1 and 0.9 should be close to 1.0 (perfect), so MAPE should be low
        self.assertLess(result["ensemble_mape"], 10.0)

if __name__ == "__main__":
    unittest.main()

