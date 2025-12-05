import os
import logging
from typing import Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import mlflow
    import mlflow.pytorch
    import mlflow.sklearn
    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False
    logger.warning("MLflow not installed. Experiment tracking will be disabled.")

class DummyContextManager:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class MLflowManager:
    """
    Manages MLflow experiments, runs, and artifact logging.
    Replaces ad-hoc file saving with structured experiment tracking.
    """
    def __init__(self, experiment_name: str = "agentic_forecast_v1", tracking_uri: Optional[str] = None):
        self.experiment_name = experiment_name
        self.enabled = _HAS_MLFLOW
        self.experiment = None
        
        if self.enabled:
            # Set tracking URI (default to local mlruns if not provided)
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            
            # Create or set experiment
            try:
                self.experiment = mlflow.set_experiment(experiment_name)
                logger.info(f"MLflow experiment set to: {experiment_name}")
            except Exception as e:
                logger.warning(f"Could not set MLflow experiment: {e}")
                self.enabled = False

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """Start a new MLflow run."""
        if self.enabled:
            return mlflow.start_run(run_name=run_name, tags=tags)
        return DummyContextManager()

    def log_params(self, params: Dict[str, Any]):
        """Log parameters to the current run."""
        if self.enabled:
            try:
                mlflow.log_params(params)
            except Exception as e:
                logger.error(f"Failed to log params: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to the current run."""
        if self.enabled:
            try:
                mlflow.log_metrics(metrics, step=step)
            except Exception as e:
                logger.error(f"Failed to log metrics: {e}")

    def log_model(self, model: Any, artifact_path: str, model_type: str = "pytorch"):
        """
        Log a model to MLflow.
        Supported types: "pytorch", "sklearn".
        """
        if self.enabled:
            try:
                if model_type == "pytorch":
                    mlflow.pytorch.log_model(model, artifact_path)
                elif model_type == "sklearn":
                    mlflow.sklearn.log_model(model, artifact_path)
                else:
                    logger.warning(f"Unsupported model type: {model_type}")
            except Exception as e:
                logger.error(f"Failed to log model: {e}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log a local file as an artifact."""
        if self.enabled:
            try:
                mlflow.log_artifact(local_path, artifact_path)
            except Exception as e:
                logger.error(f"Failed to log artifact: {e}")

    def load_model(self, run_id: str, artifact_path: str, model_type: str = "pytorch") -> Any:
        """Load a model from a specific run."""
        if not self.enabled:
            logger.warning("MLflow not enabled, cannot load model.")
            return None
            
        model_uri = f"runs:/{run_id}/{artifact_path}"
        try:
            if model_type == "pytorch":
                return mlflow.pytorch.load_model(model_uri)
            elif model_type == "sklearn":
                return mlflow.sklearn.load_model(model_uri)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
        except Exception as e:
            logger.error(f"Failed to load model from {model_uri}: {e}")
            return None

    def get_best_run(self, metric_name: str, mode: str = "min") -> Optional[Any]:
        """
        Find the best run for the current experiment based on a metric.
        mode: "min" (e.g. RMSE) or "max" (e.g. Accuracy)
        """
        if not self.enabled:
            return None
            
        try:
            order_by = f"metrics.{metric_name} ASC" if mode == "min" else f"metrics.{metric_name} DESC"
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment.experiment_id],
                order_by=[order_by],
                max_results=1
            )
            if not runs.empty:
                return runs.iloc[0]
            return None
        except Exception as e:
            logger.error(f"Failed to get best run: {e}")
            return None
