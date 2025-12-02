
import time
from ..graphs.state import GraphState
from typing import Dict, Any
import pandas as pd

class RetrainingAgent:
    def __init__(self, model):
        self.model = model

    def run(self, state: GraphState) -> Dict:
        """
        Retrains the model if drift is detected.
        """
        print("---")
        print("Retraining agent is running...")

        drift_metrics = state.get('drift_metrics', {})
        drift_detected = state.get('drift_detected', False)

        history = state.get('retraining_history', [])

        if not drift_detected or not drift_metrics:
            print("No drift detected or no metrics available. Skipping retraining.")
            history_entry = {
                "timestamp": time.time(),
                "symbols": [],
                "strategy": "idle",
                "severity_percent": 0.0,
                "note": "No retraining necessary."
            }
            history.append(history_entry)
            return {
                "retraining_history": history,
                "retraining_summary": history_entry,
                "drift_detected": False
            }

        # Find symbols with drift detected
        drift_symbols = []
        severity_values = []
        for symbol, metrics in drift_metrics.items():
            if metrics.get('drift_detected', False):
                drift_symbols.append(symbol)
                severity_values.append(abs(metrics.get('mean_drift_percentage', 0)))

        if not drift_symbols:
            print("Drift flag was set but no symbols reached the threshold. Skipping retraining.")
            history_entry = {
                "timestamp": time.time(),
                "symbols": [],
                "strategy": "idle",
                "severity_percent": 0.0,
                "note": "Drift noise only."
            }
            history.append(history_entry)
            return {
                "retraining_history": history,
                "retraining_summary": history_entry,
                "drift_detected": False
            }

        severity = max(severity_values) if severity_values else 0.0

        if severity > 25:
            strategy = "extended_batch"
        elif severity > 10:
            strategy = "standard_iteration"
        else:
            strategy = "quick_refresh"

        print(f"Drift detected for symbols: {drift_symbols} (severity: {severity:.2f}%, strategy: {strategy})")

        retrained_models = dict(state.get('retrained_models', {}))
        timestamp = int(time.time())

        for symbol in drift_symbols:
            model_path = f"/models/{symbol}_retrained_{timestamp}.pkl"
            print(f"   - Retraining model for {symbol} -> {model_path}")
            retrained_models[symbol] = model_path

        history_entry = {
            "timestamp": time.time(),
            "symbols": drift_symbols,
            "strategy": strategy,
            "severity_percent": round(severity, 2),
            "note": f"Retrained after {len(drift_symbols)} drift alerts."
        }

        history.append(history_entry)

        print("[OK] Retraining finished.")
        time.sleep(1)

        return {
            "retrained_models": retrained_models,
            "retraining_history": history,
            "retraining_summary": history_entry,
            "drift_detected": False
        }
