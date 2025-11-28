"""
Model Registry Service

Centralized model registry for storing, versioning, and managing trained models.
Provides model lifecycle management for the IB Forecast system.
"""

import os
import sys
import json
import pickle
import hashlib
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
from pathlib import Path
import torch
import tensorflow as tf
from dataclasses import dataclass, asdict
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import shutil

# Add paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    """Model metadata information."""
    model_id: str
    symbol: str
    model_type: str
    framework: str  # 'pytorch', 'tensorflow', 'sklearn'
    version: str
    created_at: datetime
    updated_at: datetime
    training_config: Dict[str, Any]
    performance_metrics: Dict[str, float]
    feature_names: List[str]
    target_name: str
    data_hash: str
    model_hash: str
    status: str  # 'active', 'deprecated', 'archived'
    tags: List[str]

@dataclass
class ModelVersion:
    """Model version information."""
    version: str
    model_id: str
    created_at: datetime
    parent_version: Optional[str]
    changes: str
    performance_delta: Dict[str, float]
    is_active: bool

class ModelRegistryService:
    """
    Centralized model registry service.

    Features:
    - Model versioning and lineage tracking
    - Performance comparison across versions
    - Model metadata management
    - Automatic model archiving and cleanup
    - Model deployment tracking
    - A/B testing support
    """

    def __init__(self, registry_path: str = '/tmp/model_registry'):
        """
        Initialize model registry service.

        Args:
            registry_path: Path to store model registry data
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

        # Registry storage
        self.models_dir = self.registry_path / 'models'
        self.metadata_dir = self.registry_path / 'metadata'
        self.versions_dir = self.registry_path / 'versions'

        self.models_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        self.versions_dir.mkdir(exist_ok=True)

        # In-memory cache
        self.metadata_cache = {}
        self.version_cache = {}

        # Load existing registry
        self._load_registry()

        logger.info(f"Model Registry Service initialized at {registry_path}")

    def register_model(self, model: Any, symbol: str, model_type: str,
                      training_results: Dict[str, Any],
                      training_config: Dict[str, Any],
                      feature_names: List[str],
                      framework: str = 'pytorch') -> str:
        """
        Register a trained model in the registry.

        Args:
            model: Trained model object
            symbol: Stock symbol
            model_type: Type of model (lstm, cnn, etc.)
            training_results: Training results and metrics
            training_config: Training configuration
            feature_names: List of feature names
            framework: ML framework used

        Returns:
            Model ID
        """
        # Generate model ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_id = f"{symbol}_{model_type}_{timestamp}"

        # Calculate hashes
        data_hash = self._calculate_data_hash(training_results)
        model_hash = self._calculate_model_hash(model)

        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            symbol=symbol,
            model_type=model_type,
            framework=framework,
            version='1.0.0',
            created_at=datetime.now(),
            updated_at=datetime.now(),
            training_config=training_config,
            performance_metrics=training_results.get('final_metrics', {}),
            feature_names=feature_names,
            target_name='target',
            data_hash=data_hash,
            model_hash=model_hash,
            status='active',
            tags=[]
        )

        # Save model and metadata
        self._save_model(model, model_id, framework)
        self._save_metadata(metadata)

        # Create initial version
        version = ModelVersion(
            version='1.0.0',
            model_id=model_id,
            created_at=datetime.now(),
            parent_version=None,
            changes='Initial model registration',
            performance_delta={},
            is_active=True
        )
        self._save_version(version)

        # Update cache
        self.metadata_cache[model_id] = metadata
        self.version_cache[model_id] = [version]

        logger.info(f"Model registered: {model_id}")

        return model_id

    def update_model(self, model_id: str, new_model: Any,
                    performance_metrics: Dict[str, float],
                    changes: str = 'Model update') -> Optional[str]:
        """
        Update an existing model with a new version.

        Args:
            model_id: Existing model ID
            new_model: Updated model object
            performance_metrics: New performance metrics
            changes: Description of changes

        Returns:
            New version string or None if failed
        """
        if model_id not in self.metadata_cache:
            logger.error(f"Model {model_id} not found in registry")
            return None

        # Get current metadata
        metadata = self.metadata_cache[model_id]
        current_version = self._get_latest_version(model_id)

        # Calculate new version
        new_version = self._increment_version(current_version.version)

        # Calculate performance delta
        old_metrics = metadata.performance_metrics
        performance_delta = {}
        for metric in performance_metrics:
            if metric in old_metrics:
                delta = performance_metrics[metric] - old_metrics[metric]
                performance_delta[f'{metric}_delta'] = delta

        # Update metadata
        metadata.updated_at = datetime.now()
        metadata.performance_metrics = performance_metrics
        metadata.model_hash = self._calculate_model_hash(new_model)

        # Create new version
        version = ModelVersion(
            version=new_version,
            model_id=model_id,
            created_at=datetime.now(),
            parent_version=current_version.version,
            changes=changes,
            performance_delta=performance_delta,
            is_active=True
        )

        # Deactivate old version
        current_version.is_active = False

        # Save updated model and metadata
        self._save_model(new_model, model_id, metadata.framework, version=new_version)
        self._save_metadata(metadata)
        self._save_version(version)
        self._save_version(current_version)  # Save deactivated version

        # Update cache
        self.version_cache[model_id].append(version)

        logger.info(f"Model {model_id} updated to version {new_version}")

        return new_version

    def load_model(self, model_id: str, version: Optional[str] = None) -> Optional[Any]:
        """
        Load a model from the registry.

        Args:
            model_id: Model ID
            version: Specific version to load (latest if None)

        Returns:
            Loaded model object
        """
        if model_id not in self.metadata_cache:
            return None

        metadata = self.metadata_cache[model_id]

        if version is None:
            version = self._get_latest_version(model_id).version

        model_path = self.models_dir / model_id / f'model_{version}.pkl'

        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return None

        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            logger.info(f"Model loaded: {model_id} v{version}")
            return model

        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            return None

    def get_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Get metadata for a model."""
        return self.metadata_cache.get(model_id)

    def list_models(self, symbol: Optional[str] = None,
                   status: str = 'active') -> List[ModelMetadata]:
        """
        List models in the registry.

        Args:
            symbol: Filter by symbol
            status: Filter by status ('active', 'deprecated', 'archived')

        Returns:
            List of model metadata
        """
        models = list(self.metadata_cache.values())

        if symbol:
            models = [m for m in models if m.symbol == symbol]

        if status:
            models = [m for m in models if m.status == status]

        return models

    def compare_models(self, model_ids: List[str]) -> pd.DataFrame:
        """
        Compare performance of multiple models.

        Args:
            model_ids: List of model IDs to compare

        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = []

        for model_id in model_ids:
            metadata = self.metadata_cache.get(model_id)
            if metadata:
                row = {
                    'model_id': model_id,
                    'symbol': metadata.symbol,
                    'model_type': metadata.model_type,
                    'version': metadata.version,
                    'status': metadata.status
                }
                row.update(metadata.performance_metrics)
                comparison_data.append(row)

        return pd.DataFrame(comparison_data)

    def archive_model(self, model_id: str, reason: str = 'Manual archive') -> bool:
        """
        Archive a model (soft delete).

        Args:
            model_id: Model ID to archive
            reason: Reason for archiving

        Returns:
            Success status
        """
        if model_id not in self.metadata_cache:
            return False

        metadata = self.metadata_cache[model_id]
        metadata.status = 'archived'
        metadata.updated_at = datetime.now()

        # Deactivate all versions
        for version in self.version_cache.get(model_id, []):
            version.is_active = False
            self._save_version(version)

        self._save_metadata(metadata)

        logger.info(f"Model archived: {model_id} - {reason}")

        return True

    def get_model_versions(self, model_id: str) -> List[ModelVersion]:
        """Get version history for a model."""
        return self.version_cache.get(model_id, [])

    def promote_model(self, model_id: str, environment: str = 'production') -> bool:
        """
        Promote a model to a specific environment.

        Args:
            model_id: Model ID
            environment: Target environment

        Returns:
            Success status
        """
        if model_id not in self.metadata_cache:
            return False

        metadata = self.metadata_cache[model_id]
        metadata.tags.append(f'promoted_{environment}')
        metadata.updated_at = datetime.now()

        self._save_metadata(metadata)

        logger.info(f"Model {model_id} promoted to {environment}")

        return True

    def cleanup_old_versions(self, model_id: str, keep_versions: int = 5) -> int:
        """
        Clean up old model versions, keeping only the most recent ones.

        Args:
            model_id: Model ID
            keep_versions: Number of versions to keep

        Returns:
            Number of versions removed
        """
        if model_id not in self.version_cache:
            return 0

        versions = sorted(self.version_cache[model_id],
                         key=lambda v: v.created_at, reverse=True)

        if len(versions) <= keep_versions:
            return 0

        # Keep only the most recent versions
        versions_to_remove = versions[keep_versions:]

        removed_count = 0
        for version in versions_to_remove:
            # Remove model file
            model_path = self.models_dir / model_id / f'model_{version.version}.pkl'
            if model_path.exists():
                model_path.unlink()

            # Remove version file
            version_path = self.versions_dir / f'{model_id}_{version.version}.json'
            if version_path.exists():
                version_path.unlink()

            removed_count += 1

        # Update cache
        self.version_cache[model_id] = versions[:keep_versions]

        logger.info(f"Cleaned up {removed_count} old versions for {model_id}")

        return removed_count

    def export_model(self, model_id: str, export_path: str,
                    version: Optional[str] = None) -> bool:
        """
        Export a model for deployment.

        Args:
            model_id: Model ID
            export_path: Path to export to
            version: Version to export

        Returns:
            Success status
        """
        model = self.load_model(model_id, version)
        metadata = self.get_model_metadata(model_id)

        if not model or not metadata:
            return False

        export_data = {
            'model': model,
            'metadata': asdict(metadata),
            'exported_at': datetime.now().isoformat()
        }

        try:
            with open(export_path, 'wb') as f:
                pickle.dump(export_data, f)

            logger.info(f"Model exported: {model_id} -> {export_path}")
            return True

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False

    def _save_model(self, model: Any, model_id: str, framework: str,
                   version: str = '1.0.0'):
        """Save model to disk."""
        model_dir = self.models_dir / model_id
        model_dir.mkdir(exist_ok=True)

        model_path = model_dir / f'model_{version}.pkl'

        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        except Exception as e:
            logger.error(f"Failed to save model {model_id}: {e}")
            raise

    def _save_metadata(self, metadata: ModelMetadata):
        """Save metadata to disk."""
        metadata_path = self.metadata_dir / f'{metadata.model_id}.json'

        try:
            with open(metadata_path, 'w') as f:
                json.dump(asdict(metadata), f, default=str, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata {metadata.model_id}: {e}")
            raise

    def _save_version(self, version: ModelVersion):
        """Save version to disk."""
        version_path = self.versions_dir / f'{version.model_id}_{version.version}.json'

        try:
            with open(version_path, 'w') as f:
                json.dump(asdict(version), f, default=str, indent=2)
        except Exception as e:
            logger.error(f"Failed to save version {version.model_id}_{version.version}: {e}")
            raise

    def _load_registry(self):
        """Load existing registry from disk."""
        # Load metadata
        for metadata_file in self.metadata_dir.glob('*.json'):
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    # Convert datetime strings back
                    data['created_at'] = datetime.fromisoformat(data['created_at'])
                    data['updated_at'] = datetime.fromisoformat(data['updated_at'])
                    metadata = ModelMetadata(**data)
                    self.metadata_cache[metadata.model_id] = metadata
            except Exception as e:
                logger.warning(f"Failed to load metadata {metadata_file}: {e}")

        # Load versions
        for version_file in self.versions_dir.glob('*.json'):
            try:
                with open(version_file, 'r') as f:
                    data = json.load(f)
                    data['created_at'] = datetime.fromisoformat(data['created_at'])
                    version = ModelVersion(**data)

                    if version.model_id not in self.version_cache:
                        self.version_cache[version.model_id] = []

                    self.version_cache[version.model_id].append(version)
            except Exception as e:
                logger.warning(f"Failed to load version {version_file}: {e}")

        logger.info(f"Loaded {len(self.metadata_cache)} models from registry")

    def _calculate_data_hash(self, training_results: Dict[str, Any]) -> str:
        """Calculate hash of training data."""
        # Use training metrics and config for data hash
        hash_data = json.dumps(training_results, sort_keys=True, default=str)
        return hashlib.sha256(hash_data.encode()).hexdigest()[:16]

    def _calculate_model_hash(self, model: Any) -> str:
        """Calculate hash of model parameters."""
        try:
            # For PyTorch models
            if hasattr(model, 'state_dict'):
                state_dict = model.state_dict()
                hash_data = str(sorted([(k, v.shape if hasattr(v, 'shape') else str(v))
                                       for k, v in state_dict.items()]))
            else:
                # For other models, use pickle representation
                hash_data = pickle.dumps(model)

            return hashlib.sha256(hash_data).hexdigest()[:16]
        except Exception:
            return hashlib.sha256(str(model).encode()).hexdigest()[:16]

    def _get_latest_version(self, model_id: str) -> ModelVersion:
        """Get the latest version for a model."""
        versions = self.version_cache.get(model_id, [])
        if not versions:
            return ModelVersion(
                version='1.0.0',
                model_id=model_id,
                created_at=datetime.now(),
                parent_version=None,
                changes='Default version',
                performance_delta={},
                is_active=True
            )

        return max(versions, key=lambda v: v.created_at)

    def _increment_version(self, version: str) -> str:
        """Increment version number."""
        parts = version.split('.')
        if len(parts) >= 3:
            parts[2] = str(int(parts[2]) + 1)
        else:
            parts = [parts[0], parts[1], '1']

        return '.'.join(parts)

# Convenience functions
def create_model_registry_service(registry_path: str = '/tmp/model_registry'):
    """Create and configure model registry service."""
    return ModelRegistryService(registry_path)

def register_trained_model(model: Any, symbol: str, model_type: str,
                          training_results: Dict[str, Any],
                          training_config: Dict[str, Any],
                          feature_names: List[str]) -> str:
    """Register a trained model with default settings."""
    registry = create_model_registry_service()
    return registry.register_model(
        model, symbol, model_type, training_results,
        training_config, feature_names
    )