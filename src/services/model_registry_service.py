"""
Model Registry Service

Centralized management for model versioning, storage, and retrieval.
Handles metadata tracking and ensures reproducibility.
"""

import os
import json
import logging
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, List
import torch
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelRegistryService:
    """
    Service for managing model artifacts and metadata.
    
    Features:
    - Versioned model storage
    - Metadata management (hyperparameters, metrics, training config)
    - Model retrieval by ID or criteria (e.g., best performance)
    """
    
    def __init__(self, storage_path: str = "models/registry"):
        """
        Initialize the model registry.
        
        Args:
            storage_path: Base directory for storing models and metadata.
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.metadata_db_path = self.storage_path / "metadata_db.json"
        self._load_metadata_db()
        
    def _load_metadata_db(self):
        """Load the metadata database from disk."""
        if self.metadata_db_path.exists():
            try:
                with open(self.metadata_db_path, 'r') as f:
                    self.metadata_db = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load metadata DB: {e}")
                self.metadata_db = {}
        else:
            self.metadata_db = {}
            
    def _save_metadata_db(self):
        """Save the metadata database to disk."""
        try:
            with open(self.metadata_db_path, 'w') as f:
                json.dump(self.metadata_db, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata DB: {e}")

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
            str: The unique model ID.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_id = f"{symbol}_{model_type}_{timestamp}"
        
        # Create directory for this model
        model_dir = self.storage_path / symbol / model_type / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model artifact
        artifact_path = model_dir / "model.pt" # Default extension
        
        try:
            if framework == "pytorch":
                if hasattr(model, 'state_dict'):
                    torch.save(model.state_dict(), artifact_path)
                else:
                    torch.save(model, artifact_path)
            elif framework == "sklearn":
                artifact_path = model_dir / "model.joblib"
                joblib.dump(model, artifact_path)
            elif framework == "neuralforecast":
                # NeuralForecast models usually save to a directory
                # If 'model' is a NeuralForecast object, use its save method
                if hasattr(model, 'save'):
                    model.save(str(model_dir / "checkpoints"))
                    artifact_path = model_dir / "checkpoints"
                else:
                    # If it's a raw PyTorch model inside NF
                    torch.save(model, artifact_path)
            else:
                logger.warning(f"Unknown framework {framework}, attempting torch save")
                torch.save(model, artifact_path)
                
            logger.info(f"Saved model artifact to {artifact_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model artifact: {e}")
            raise e
            
        # Enrich and save metadata
        full_metadata = {
            'model_id': model_id,
            'symbol': symbol,
            'model_type': model_type,
            'framework': framework,
            'created_at': datetime.now().isoformat(),
            'artifact_path': str(artifact_path),
            'metrics': metadata.get('metrics', {}),
            'hyperparameters': metadata.get('hyperparameters', {}),
            'training_config': metadata.get('training_config', {})
        }
        
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(full_metadata, f, indent=2)
            
        # Update central DB
        self.metadata_db[model_id] = full_metadata
        self._save_metadata_db()
        
        return model_id

    def get_model_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for a specific model ID."""
        return self.metadata_db.get(model_id)

    def list_models(self, symbol: Optional[str] = None, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List models matching criteria."""
        models = list(self.metadata_db.values())
        
        if symbol:
            models = [m for m in models if m['symbol'] == symbol]
        if model_type:
            models = [m for m in models if m['model_type'] == model_type]
            
        return models

    def get_best_model(self, symbol: str, metric: str = "val_loss", mode: str = "min") -> Optional[Dict[str, Any]]:
        """
        Get the best model for a symbol based on a metric.
        
        Args:
            symbol: The symbol to query.
            metric: The metric key to optimize (e.g., 'val_loss', 'mae').
            mode: 'min' (lower is better) or 'max' (higher is better).
            
        Returns:
            Dict: Metadata of the best model.
        """
        models = self.list_models(symbol=symbol)
        if not models:
            return None
            
        # Filter models that have the metric
        valid_models = [m for m in models if 'metrics' in m and metric in m['metrics']]
        if not valid_models:
            return None
            
        def get_metric_val(m):
            return m['metrics'][metric]
            
        if mode == "min":
            best_model = min(valid_models, key=get_metric_val)
        else:
            best_model = max(valid_models, key=get_metric_val)
            
        return best_model

    def load_model(self, model_id: str) -> Any:
        """
        Load a model artifact into memory.
        
        Args:
            model_id: The ID of the model to load.
            
        Returns:
            The loaded model object.
        """
        metadata = self.get_model_metadata(model_id)
        if not metadata:
            raise ValueError(f"Model ID {model_id} not found")
            
        artifact_path = Path(metadata['artifact_path'])
        framework = metadata.get('framework', 'pytorch')
        
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found at {artifact_path}")
            
        try:
            if framework == "pytorch":
                # This assumes we know the class to instantiate if it's a state_dict
                # For now, just return the loaded object (might be state_dict or full model)
                return torch.load(artifact_path)
            elif framework == "sklearn":
                return joblib.load(artifact_path)
            elif framework == "neuralforecast":
                from neuralforecast import NeuralForecast
                return NeuralForecast.load(str(artifact_path))
            else:
                raise ValueError(f"Unsupported framework: {framework}")
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise e

    def get_last_hpo_run(self, symbol: str) -> Optional[float]:
        """
        Get the timestamp of the last HPO run for a symbol.
        
        Args:
            symbol: The symbol to check.
            
        Returns:
            Timestamp of the last run, or None if never run.
        """
        # We can store this in a separate file or in the metadata DB
        # For simplicity, let's use a separate JSON file for HPO tracking
        hpo_tracker_path = self.storage_path / "hpo_tracker.json"
        if hpo_tracker_path.exists():
            try:
                with open(hpo_tracker_path, 'r') as f:
                    tracker = json.load(f)
                return tracker.get(symbol)
            except Exception as e:
                logger.error(f"Failed to load HPO tracker: {e}")
                return None
        return None

    def set_last_hpo_run(self, symbol: str, timestamp: float):
        """
        Set the timestamp of the last HPO run for a symbol.
        
        Args:
            symbol: The symbol.
            timestamp: The timestamp.
        """
        hpo_tracker_path = self.storage_path / "hpo_tracker.json"
        tracker = {}
        if hpo_tracker_path.exists():
            try:
                with open(hpo_tracker_path, 'r') as f:
                    tracker = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load HPO tracker: {e}")
        
        tracker[symbol] = timestamp
        
        try:
            with open(hpo_tracker_path, 'w') as f:
                json.dump(tracker, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save HPO tracker: {e}")
