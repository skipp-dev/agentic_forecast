
import os
import json
import time
import unittest
from prometheus_client import REGISTRY

# Mock environment variables before importing the module
os.environ["TRUST_SCORES_PATH"] = "data/metrics/test_trust_scores.json"
os.environ["FORECAST_AGENT_OUTPUT_PATH"] = "data/metrics/test_forecast_output.json"

from services.metrics_exporter import build_metrics_registry

class TestTrustScoreExport(unittest.TestCase):
    def setUp(self):
        # Create dummy trust scores file
        self.trust_scores = {
            "AAPL": 0.85,
            "GOOGL": 0.42,
            "TSLA": 0.15
        }
        os.makedirs("data/metrics", exist_ok=True)
        with open("data/metrics/test_trust_scores.json", "w") as f:
            json.dump(self.trust_scores, f)
            
        # Create dummy forecast output (to ensure it doesn't overwrite if we handled that logic)
        self.forecast_output = {
            "symbol": "AAPL",
            "risk_assessment": {
                "trust_score": 0.11  # Old score, should be overwritten or ignored if we prioritize the new file
            }
        }
        with open("data/metrics/test_forecast_output.json", "w") as f:
            json.dump(self.forecast_output, f)

    def tearDown(self):
        # Cleanup
        if os.path.exists("data/metrics/test_trust_scores.json"):
            os.remove("data/metrics/test_trust_scores.json")
        if os.path.exists("data/metrics/test_forecast_output.json"):
            os.remove("data/metrics/test_forecast_output.json")

    def test_trust_score_export(self):
        registry = build_metrics_registry()
        
        # Get the metric sample
        trust_metric = registry.get_sample_value('forecast_trust_score', labels={'symbol': 'AAPL'})
        self.assertEqual(trust_metric, 0.85, "AAPL trust score should match trust_scores.json")
        
        trust_metric_googl = registry.get_sample_value('forecast_trust_score', labels={'symbol': 'GOOGL'})
        self.assertEqual(trust_metric_googl, 0.42, "GOOGL trust score should match trust_scores.json")
        
        trust_metric_tsla = registry.get_sample_value('forecast_trust_score', labels={'symbol': 'TSLA'})
        self.assertEqual(trust_metric_tsla, 0.15, "TSLA trust score should match trust_scores.json")

if __name__ == '__main__':
    unittest.main()
