"""
Model Registry Service

Centralized management for model versioning, storage, and retrieval.
Handles metadata tracking and ensures reproducibility using MLflow.
"""

import os
import json
import logging
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import torch
import joblib
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

class ModelRegistryService:
    """
    Service for managing model artifacts and metadata using MLflow.
    
    Features:
    - Versioned model storage via MLflow
    - Metadata management (hyperparameters, metrics, training config)
    - Model retrieval by ID or criteria (e.g., best performance)
    """
    
    def __init__(self, storage_path: str = "mlruns"):
        """
        Initialize the model registry with MLflow.
        
        Args:
            storage_path: Base directory for MLflow runs (if using local file store).
        """
        # Ensure absolute path for local storage
        self.storage_path = os.path.abspath(storage_path)
        self.tracking_uri = f"file:///{self.storage_path.replace(os.sep, '/')}"
        
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        
        logger.info(f"ModelRegistryService initialized with MLflow at {self.tracking_uri}")
        
        # Create default experiment if not exists
        self.experiment_name = "agentic_forecast"
        
        # Use client to check for experiment to avoid caching issues
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        
        if experiment is None:
            self.experiment_id = self.client.create_experiment(self.experiment_name)
        else:
            self.experiment_id = experiment.experiment_id
            # Verify it exists in this store
            try:
                self.client.get_experiment(self.experiment_id)
            except Exception:
                logger.warning(f"Experiment {self.experiment_name} found but ID {self.experiment_id} not accessible. Recreating.")
                # Try to create a new one with a unique name if the old one is stuck
                self.experiment_name = f"agentic_forecast_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                self.experiment_id = self.client.create_experiment(self.experiment_name)

    def register_model(self, model: Any, symbol: str, model_type: str, 
                       training_results: Dict[str, Any], training_config: Dict[str, Any], 
                       feature_names: List[str], framework: str = "pytorch") -> str:
        """Alias for save_model to support legacy tests."""
        metadata = {
            'metrics': training_results.get('final_metrics', {}),
            'hyperparameters': training_config,
            'feature_names': feature_names
        }
        return self.save_model(model, symbol, model_type, metadata, framework)

    def save_model(self, 
                   model: Any, 
                   symbol: str, 
                   model_type: str, 
                   metadata: Dict[str, Any], 
                   framework: str = "pytorch") -> str:
        """
        Save a model and its metadata to the registry.
        
        Args:
            model: The model object to save.
            symbol: The symbol the model was trained on.
            model_type: The type/family of the model (e.g., 'NLinear', 'LSTM').
            metadata: Dictionary containing metrics, hyperparameters, etc.
            framework: The framework used ('pytorch', 'sklearn', 'neuralforecast').
            
        Returns:
            str: The unique model ID (MLflow Run ID).
        """
        run_name = f"{symbol}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Use client to create run to avoid global state issues with experiment resolution
        run = self.client.create_run(experiment_id=self.experiment_id, run_name=run_name)
        run_id = run.info.run_id
        
        try:
            # Use start_run with the specific run_id to enable fluent API (log_model, etc.)
            # We must ensure tracking URI is set correctly for this to work
            mlflow.set_tracking_uri(self.tracking_uri)
            
            with mlflow.start_run(run_id=run_id):
                # Log Parameters
                params = metadata.get('hyperparameters', {})
                params.update(metadata.get('training_config', {}))
                # Flatten nested dicts if necessary or log as json string
                for k, v in params.items():
                    mlflow.log_param(k, str(v)) # Convert to str to be safe
                
                # Log Metrics
                metrics = metadata.get('metrics', {})
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, v)
                
                # Log Tags
                mlflow.set_tag("symbol", symbol)
                mlflow.set_tag("model_type", model_type)
                mlflow.set_tag("framework", framework)
                mlflow.set_tag("created_at", datetime.now().isoformat())
                
                # Log Model Artifact
                if framework == "pytorch":
                    mlflow.pytorch.log_model(model, "model")
                elif framework == "sklearn":
                    mlflow.sklearn.log_model(model, "model")
                elif framework == "neuralforecast":
                    # NeuralForecast integration with MLflow is custom
                    # We save locally then log artifacts
                    local_path = f"temp_nf_{run_id}"
                    if hasattr(model, 'save'):
                        model.save(local_path)
                        mlflow.log_artifacts(local_path, artifact_path="model")
                        shutil.rmtree(local_path, ignore_errors=True)
                    else:
                        # Fallback
                        mlflow.pytorch.log_model(model, "model")
                else:
                    logger.warning(f"Unknown framework {framework}, logging as generic artifact")
                    # Try pickling
                    mlflow.log_text(str(model), "model_repr.txt")
                    
            logger.info(f"Saved model {run_name} with Run ID {run_id}")
            return run_id
            
        except Exception as e:
            logger.error(f"Failed to save model artifact: {e}")
            self.client.set_terminated(run_id, status="FAILED")
            raise e

    def get_model_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for a specific model ID (Run ID)."""
        try:
            run = self.client.get_run(model_id)
            return {
                'model_id': run.info.run_id,
                'symbol': run.data.tags.get('symbol'),
                'model_type': run.data.tags.get('model_type'),
                'framework': run.data.tags.get('framework'),
                'created_at': run.data.tags.get('created_at'),
                'metrics': run.data.metrics,
                'hyperparameters': run.data.params,
                'status': run.info.status
            }
        except Exception as e:
            logger.error(f"Failed to get metadata for {model_id}: {e}")
            return None

    def list_models(self, symbol: Optional[str] = None, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List models matching criteria."""
        filter_string = ""
        conditions = []
        if symbol:
            conditions.append(f"tags.symbol = '{symbol}'")
        if model_type:
            conditions.append(f"tags.model_type = '{model_type}'")
            
        filter_string = " AND ".join(conditions) if conditions else None
        
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=filter_string,
            order_by=["attribute.start_time DESC"]
        )
        
        models = []
        for run in runs:
            models.append({
                'model_id': run.info.run_id,
                'symbol': run.data.tags.get('symbol'),
                'model_type': run.data.tags.get('model_type'),
                'metrics': run.data.metrics,
                'hyperparameters': run.data.params,
                'stage': run.data.tags.get('stage', 'None')
            })
        return models

    def get_best_model(self, symbol: str, metric: str = "val_loss", mode: str = "min") -> Optional[Dict[str, Any]]:
        """
        Get the best model for a symbol based on a metric.
        """
        filter_string = f"tags.symbol = '{symbol}'"
        order = f"metrics.{metric} ASC" if mode == "min" else f"metrics.{metric} DESC"
        
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=filter_string,
            order_by=[order],
            max_results=1
        )
        
        if not runs:
            return None
            
        run = runs[0]
        return {
            'model_id': run.info.run_id,
            'symbol': run.data.tags.get('symbol'),
            'model_type': run.data.tags.get('model_type'),
            'metrics': run.data.metrics,
            'hyperparameters': run.data.params
        }

    def transition_model_stage(self, model_id: str, stage: str):
        """
        Transition a model to a new stage (e.g., 'Staging', 'Production', 'Archived').
        """
        valid_stages = ['None', 'Staging', 'Production', 'Archived']
        if stage not in valid_stages:
            raise ValueError(f"Invalid stage: {stage}. Must be one of {valid_stages}")
            
        # If promoting to Production, demote others for the same symbol FIRST
        if stage == 'Production':
            metadata = self.get_model_metadata(model_id)
            if metadata and metadata.get('symbol'):
                self._demote_other_production_models(metadata['symbol'], model_id)

        self.client.set_tag(model_id, "stage", stage)
        logger.info(f"Transitioned model {model_id} to stage {stage}")

    def _demote_other_production_models(self, symbol: str, new_prod_id: str):
        """Demote existing Production models for a symbol to Archived."""
        # Find ALL production models
        filter_string = f"tags.symbol = '{symbol}' AND tags.stage = 'Production'"
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=filter_string
        )
        
        for run in runs:
            if run.info.run_id != new_prod_id:
                self.client.set_tag(run.info.run_id, "stage", "Archived")
                logger.info(f"Demoted previous production model {run.info.run_id} to Archived")

    def get_production_model(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Retrieve the current Production model for a symbol."""
        filter_string = f"tags.symbol = '{symbol}' AND tags.stage = 'Production'"
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=filter_string,
            max_results=1,
            order_by=["attribute.start_time DESC"]
        )
        
        if not runs:
            return None
            
        run = runs[0]
        return {
            'model_id': run.info.run_id,
            'symbol': run.data.tags.get('symbol'),
            'model_type': run.data.tags.get('model_type'),
            'metrics': run.data.metrics,
            'hyperparameters': run.data.params,
            'stage': 'Production'
        }

    def load_model(self, model_id: str) -> Any:
        """
        Load a model artifact into memory.
        """
        try:
            run = self.client.get_run(model_id)
            framework = run.data.tags.get('framework', 'pytorch')
            
            model_uri = f"runs:/{model_id}/model"
            
            if framework == "pytorch":
                return mlflow.pytorch.load_model(model_uri)
            elif framework == "sklearn":
                return mlflow.sklearn.load_model(model_uri)
            elif framework == "neuralforecast":
                # For NF, we need to download artifacts and load
                local_path = mlflow.artifacts.download_artifacts(run_id=model_id, artifact_path="model")
                from neuralforecast import NeuralForecast
                return NeuralForecast.load(local_path)
            else:
                raise ValueError(f"Unsupported framework: {framework}")
                
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise e

    def get_last_hpo_run(self, symbol: str) -> Optional[float]:
        """
        Get the timestamp of the last HPO run for a symbol.
        """
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=f"tags.symbol = '{symbol}'",
            order_by=["attribute.start_time DESC"],
            max_results=1
        )
        
        if runs:
            # Return timestamp in seconds
            return runs[0].info.start_time / 1000.0
        return None

    def set_last_hpo_run(self, symbol: str, timestamp: float):
        """
        Set the timestamp of the last HPO run for a symbol.
        """
        with mlflow.start_run(experiment_id=self.experiment_id, run_name=f"HPO_MARKER_{symbol}") as run:
            mlflow.set_tag("symbol", symbol)
            mlflow.set_tag("type", "hpo_marker")
            mlflow.log_param("timestamp", timestamp)
