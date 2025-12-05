import unittest
from unittest.mock import MagicMock
from src.agents.hyperparameter_search_agent import HyperparameterSearchAgent
from models.model_zoo import ModelZoo, DataSpec
import pandas as pd
import numpy as np

class TestRiskModeInjection(unittest.TestCase):
    def setUp(self):
        idx = pd.date_range("2024-03-01", periods=48, freq="h")
        y = 100 + np.random.default_rng(0).normal(0, 1, len(idx))
        frame = pd.DataFrame({"ds": idx, "y": y, "unique_id": "XYZ", "feat": np.random.random(len(idx))})
        self.data_spec = DataSpec(
            job_id="test_job",
            symbol_scope="XYZ",
            train_df=frame.iloc[:32], 
            val_df=frame.iloc[32:], 
            feature_cols=["feat"],
            target_col="y",
            horizon=16
        )

    def test_risk_mode_injects_baseline(self):
        # Mock ModelZoo
        mock_zoo = MagicMock(spec=ModelZoo)
        
        # Setup mock returns
        mock_res_nhits = MagicMock()
        mock_res_nhits.model_family = "AutoNHITS"
        mock_res_nhits.best_val_mape = 0.1
        mock_zoo.train_autonhits.return_value = mock_res_nhits
        
        mock_res_nbeats = MagicMock()
        mock_res_nbeats.model_family = "AutoNBEATS"
        mock_res_nbeats.best_val_mape = 0.15
        mock_zoo.train_autonbeats.return_value = mock_res_nbeats
        
        mock_res_baseline = MagicMock()
        mock_res_baseline.model_family = "BaselineLinear"
        mock_res_baseline.best_val_mape = 0.2
        mock_zoo.train_baseline_linear.return_value = mock_res_baseline

        agent = HyperparameterSearchAgent(risk_mode=True, model_zoo=mock_zoo)
        
        res = agent.run_model_zoo_hpo(symbol="XYZ", data_spec=self.data_spec, model_families=["AutoNHITS", "AutoNBEATS"])
        
        self.assertTrue(res["success"])
        fams = [r["model_family"] for r in res["all_results"]]
        self.assertIn("BaselineLinear", fams)  # injected from risk families
        self.assertIn("AutoNHITS", fams)
        self.assertIn("AutoNBEATS", fams)
        
        # Verify calls
        mock_zoo.train_baseline_linear.assert_called_once()
        mock_zoo.train_autonhits.assert_called_once()
        mock_zoo.train_autonbeats.assert_called_once()

if __name__ == "__main__":
    unittest.main()

