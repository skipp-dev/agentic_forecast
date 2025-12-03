import unittest
from unittest.mock import MagicMock
from src.agents.hyperparameter_search_agent import HyperparameterSearchAgent
from models.model_zoo import ModelZoo, DataSpec
import pandas as pd
import numpy as np

class TestHPOAgentRiskMode(unittest.TestCase):
    def setUp(self):
        idx = pd.date_range("2024-02-01", periods=72, freq="h")
        y = 50 + np.sin(np.linspace(0, 10, len(idx)))
        frame = pd.DataFrame({"ds": idx, "y": y, "unique_id": "TEST", "feat": np.random.random(len(idx))})
        self.data_spec = DataSpec(
            job_id="test_job",
            symbol_scope="TEST",
            train_df=frame.iloc[:48],
            val_df=frame.iloc[48:],
            feature_cols=["feat"],
            target_col="y",
            horizon=24
        )

    def test_risk_mode_family_injection(self):
        # Mock ModelZoo
        mock_zoo = MagicMock(spec=ModelZoo)
        
        # Mock results
        mock_res_nhits = MagicMock()
        mock_res_nhits.model_family = "NHITS"
        mock_res_nhits.best_val_mape = 0.1
        mock_zoo.train_lstm.return_value = mock_res_nhits
        mock_zoo.train_autonhits.return_value = mock_res_nhits
        
        mock_res_tft = MagicMock()
        mock_res_tft.model_family = "TFT"
        mock_res_tft.best_val_mape = 0.15
        mock_zoo.train_tft.return_value = mock_res_tft
        
        mock_res_baseline = MagicMock()
        mock_res_baseline.model_family = "BaselineLinear"
        mock_res_baseline.best_val_mape = 0.2
        mock_zoo.train_baseline_linear.return_value = mock_res_baseline

        agent = HyperparameterSearchAgent(risk_mode=True, model_zoo=mock_zoo)
        result = agent.run_model_zoo_hpo(symbol="TEST", data_spec=self.data_spec, model_families=["NHITS","TFT"])
        
        self.assertTrue(result["success"])
        tried = [r["model_family"] for r in result["all_results"]]
        
        self.assertIn("NHITS", tried)
        # BaselineLinear should appear from risk families injection
        self.assertIn("BaselineLinear", tried)

    def test_best_family_selection(self):
        # Mock ModelZoo
        mock_zoo = MagicMock(spec=ModelZoo)
        
        # Mock results - NHITS is better (lower MAPE)
        mock_res_nhits = MagicMock()
        mock_res_nhits.model_family = "NHITS"
        mock_res_nhits.best_val_mape = 0.05
        mock_zoo.train_lstm.return_value = mock_res_nhits
        mock_zoo.train_autonhits.return_value = mock_res_nhits
        
        mock_res_tft = MagicMock()
        mock_res_tft.model_family = "TFT"
        mock_res_tft.best_val_mape = 0.15
        mock_zoo.train_tft.return_value = mock_res_tft

        agent = HyperparameterSearchAgent(risk_mode=False, model_zoo=mock_zoo)
        result = agent.run_model_zoo_hpo(symbol="XYZ", data_spec=self.data_spec, model_families=["NHITS","TFT"])
        
        self.assertTrue(result["success"])
        self.assertEqual(result["best_family"], "NHITS")
        self.assertEqual(result["best_val_mape"], 0.05)

if __name__ == "__main__":
    unittest.main()

