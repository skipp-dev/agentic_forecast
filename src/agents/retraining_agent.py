
import time
from ..graphs.state import GraphState
from typing import Dict, Any
import pandas as pd
import logging
from src.services.training_service import GPUTrainingService
from models.model_zoo import DataSpec

logger = logging.getLogger(__name__)

class RetrainingAgent:
    def __init__(self, model=None):
        self.model = model
        self.training_service = GPUTrainingService()

    def run(self, state: GraphState) -> Dict:
        """
        Retrains the model if drift is detected.
        """
        logger.info("--- Retraining agent is running ---")

        drift_metrics = state.get('drift_metrics', {})
        drift_detected = state.get('drift_detected', False)
        features = state.get('features', {})

        history = state.get('retraining_history', [])

        if not drift_detected or not drift_metrics:
            logger.info("No drift detected or no metrics available. Skipping retraining.")
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
            logger.info("Drift flag was set but no symbols reached the threshold. Skipping retraining.")
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

        logger.info(f"Drift detected for symbols: {drift_symbols} (severity: {severity:.2f}%, strategy: {strategy})")

        retrained_models = dict(state.get('retrained_models', {}))
        
        for symbol in drift_symbols:
            logger.info(f"Retraining model for {symbol}...")
            
            # Prepare data
            if symbol not in features:
                logger.warning(f"No features found for {symbol}, skipping retraining.")
                continue
                
            data = features[symbol]
            if isinstance(data, dict):
                data = pd.DataFrame.from_dict(data, orient='index')
                data.index = pd.to_datetime(data.index)
            
            # Ensure 'y' column
            if 'y' not in data.columns:
                data['y'] = data['close'].pct_change(1).shift(-1)
                data = data.dropna()
                
            # Split data (simple split for retraining)
            horizon = 3 # Default horizon
            train_df = data.iloc[:-horizon]
            val_df = data.iloc[-horizon:]
            
            # Create DataSpec
            data_spec = DataSpec(
                job_id=f"retrain_{symbol}_{int(time.time())}",
                symbol_scope=symbol,
                train_df=train_df,
                val_df=val_df,
                target_col='y'
            )
            
            # Train using GPUTrainingService
            # For retraining, we might want to use the best model family or a default
            # For now, let's default to BaselineLinear or check if we have a previous best
            model_type = "BaselineLinear" # Default
            
            res = self.training_service.train_model(
                symbol=symbol,
                model_type=model_type,
                data=data_spec,
                hyperparams={'horizon': horizon}
            )
            
            if res['status'] == 'success':
                model_id = res['model_id']
                retrained_models[symbol] = model_id
                logger.info(f"Successfully retrained {model_type} for {symbol}: {model_id}")
            else:
                logger.error(f"Failed to retrain {symbol}: {res.get('error')}")

        history_entry = {
            "timestamp": time.time(),
            "symbols": drift_symbols,
            "strategy": strategy,
            "severity_percent": round(severity, 2),
            "note": f"Retrained after {len(drift_symbols)} drift alerts."
        }

        history.append(history_entry)

        logger.info("[OK] Retraining finished.")

        return {
            "retrained_models": retrained_models,
            "retraining_history": history,
            "retraining_summary": history_entry,
            "drift_detected": False
        }
