"""
GPU Training Service

Scalable GPU-accelerated model training service with distributed capabilities.
Provides training infrastructure for the IB Forecast system.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import GPUtil
from dataclasses import dataclass
import json
import tempfile

# Model classes for PyTorch models
class LSTMModel(nn.Module):
    """LSTM model for time series forecasting."""
    
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                          dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Handle both 2D (batch, features) and 3D (batch, seq, features) input
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension: (batch, 1, features)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use last sequence output
        return out

class CNNModel(nn.Module):
    """CNN model for time series forecasting."""
    
    def __init__(self, input_size):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

class TransformerModel(nn.Module):
    """Transformer model for time series forecasting."""
    
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer_encoder(x)
        x = self.fc(x[:, -1, :])  # Use last sequence output
        return x

# Add paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.gpu_services import get_gpu_services
from agents.hyperparameter_search_agent import HyperparameterSearchAgent
from agents.feature_engineer_agent import FeatureEngineerAgent

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    model_type: str
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    use_gpu: bool = True
    distributed: bool = False
    n_workers: int = 1
    memory_limit_gb: float = 8.0

@dataclass
class TrainingMetrics:
    """Training performance metrics."""
    epoch: int
    train_loss: float
    val_loss: float
    train_mae: float
    val_mae: float
    learning_rate: float
    gpu_memory_used: float
    training_time: float

class GPUDataset(Dataset):
    """GPU-optimized dataset for training."""

    def __init__(self, X: np.ndarray, y: np.ndarray, device: str = 'cuda'):
        self.X = torch.tensor(X, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class GPUTrainingService:
    """
    GPU-accelerated model training service.

    Features:
    - Multi-GPU training support
    - Distributed training capabilities
    - Memory optimization and monitoring
    - Automatic batching and prefetching
    - Early stopping and learning rate scheduling
    - Model checkpointing and recovery
    - Performance monitoring and profiling
    """

    def __init__(self, gpu_services=None, model_registry=None):
        """
        Initialize GPU training service.

        Args:
            gpu_services: GPU services instance
            model_registry: Model registry service instance
        """
        self.gpu_services = gpu_services or get_gpu_services()
        self.hyperparameter_agent = HyperparameterSearchAgent()
        self.feature_agent = FeatureEngineerAgent()
        self.model_registry = model_registry  # Use provided registry or create dict fallback

        # If no registry provided, create a simple dict (for backward compatibility)
        if self.model_registry is None:
            self.model_registry = {}

        # Training state
        self.active_trainings = {}
        self.training_history = {}

        # GPU configuration
        self.gpu_devices = self._detect_gpu_devices()
        self.memory_manager = self._initialize_memory_manager()

        # Distributed training setup
        self.distributed_config = self._setup_distributed_training()

        logger.info(f"GPU Training Service initialized with {len(self.gpu_devices)} GPU devices")

    def train_model(self, symbol: str, model_config: Dict[str, Any],
                   training_data: Optional[Dict[str, Any]] = None,
                   config: Optional[TrainingConfig] = None) -> Dict[str, Any]:
        """
        Train a model for a symbol with GPU acceleration.

        Args:
            symbol: Stock symbol
            model_config: Model configuration
            training_data: Pre-prepared training data (optional)
            config: Training configuration

        Returns:
            Training results and metrics
        """
        training_id = f"{symbol}_{model_config.get('type', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Starting GPU training for {symbol}, ID: {training_id}")

        # Default configuration
        if config is None:
            config = TrainingConfig(
                model_type=model_config.get('type', 'lstm'),
                use_gpu=len(self.gpu_devices) > 0
            )

        # Prepare training data
        if training_data is None:
            training_data = self._prepare_training_data(symbol)

        if not training_data or len(training_data['X_train']) < 100:
            logger.error(f"Insufficient training data for {symbol}")
            return {'error': 'Insufficient training data', 'training_id': training_id}

        # Start training
        self.active_trainings[training_id] = {
            'symbol': symbol,
            'config': config,
            'start_time': datetime.now(),
            'status': 'running'
        }

        try:
            if config.distributed and len(self.gpu_devices) > 1:
                results = self._train_distributed(training_data, model_config, config, training_id)
            else:
                results = self._train_single_gpu(training_data, model_config, config, training_id)

            # Update training state
            self.active_trainings[training_id].update({
                'status': 'completed',
                'end_time': datetime.now(),
                'results': results
            })

            # Store in registry
            if hasattr(self.model_registry, 'register_model'):
                # Use ModelRegistryService
                model_id = self.model_registry.register_model(
                    model=results.get('model'),
                    symbol=symbol,
                    model_type=model_config.get('type', 'unknown'),
                    training_results=results,
                    training_config=config.__dict__ if hasattr(config, '__dict__') else config,
                    feature_names=training_data.get('feature_names', [])
                )
                results['model_id'] = model_id
            else:
                # Fallback to dict storage
                self.model_registry[training_id] = {
                    'symbol': symbol,
                    'model_config': model_config,
                    'training_config': config,
                    'results': results,
                    'trained_at': datetime.now()
                }

            logger.info(f"Training completed for {training_id}")
            return results

        except Exception as e:
            logger.error(f"Training failed for {training_id}: {e}")
            self.active_trainings[training_id].update({
                'status': 'failed',
                'error': str(e),
                'end_time': datetime.now()
            })
            return {'error': str(e), 'training_id': training_id}

    def _prepare_training_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Prepare training data for a symbol."""
        try:
            # Get raw data
            raw_data = self.feature_agent.data_pipeline.av_client.get_daily_data(symbol, outputsize='full')
            if raw_data.empty:
                return None

            data_df = pd.DataFrame(raw_data)

            # Engineer features
            feature_data = self.feature_agent.engineer_features(
                symbol, data_df, feature_sets=['basic', 'spectral', 'volatility']
            )

            # Prepare target (next day return)
            feature_data['target'] = feature_data['close'].shift(-1).pct_change(fill_method=None)

            # Remove NaN values
            feature_data = feature_data.dropna()

            # Remove infinite values
            feature_data = feature_data.replace([np.inf, -np.inf], np.nan).dropna()

            if len(feature_data) < 100:
                return None

            # Split features and target
            feature_cols = [col for col in feature_data.columns
                          if col not in ['close', 'target']]
            X = feature_data[feature_cols].values
            y = feature_data['target'].values

            # Train/validation split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )

            return {
                'X_train': X_train,
                'X_val': X_val,
                'y_train': y_train,
                'y_val': y_val,
                'feature_names': feature_cols
            }

        except Exception as e:
            logger.error(f"Data preparation failed for {symbol}: {e}")
            return None

    def _train_single_gpu(self, training_data: Dict[str, Any],
                         model_config: Dict[str, Any], config: TrainingConfig,
                         training_id: str) -> Dict[str, Any]:
        """Train model on single GPU."""
        device = 'cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu'
        logger.info(f"Training on device: {device}")

        # Prepare data
        X_train, X_val = training_data['X_train'], training_data['X_val']
        y_train, y_val = training_data['y_train'], training_data['y_val']

        # Create datasets
        train_dataset = GPUDataset(X_train, y_train, device)
        val_dataset = GPUDataset(X_val, y_val, device)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                                shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                              shuffle=False, num_workers=0)

        # Create model
        model = self._create_model(model_config, X_train.shape[1])
        model = model.to(device)

        # Optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        training_metrics = []

        for epoch in range(config.epochs):
            start_time = datetime.now()

            # Training phase
            model.train()
            train_loss = 0.0
            train_predictions = []
            train_targets = []

            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_predictions.extend(outputs.squeeze().cpu().detach().numpy())
                train_targets.extend(targets.cpu().detach().numpy())

            train_loss /= len(train_loader)
            train_mae = mean_absolute_error(train_targets, train_predictions)

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets = []

            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), targets)

                    val_loss += loss.item()
                    val_predictions.extend(outputs.squeeze().cpu().detach().numpy())
                    val_targets.extend(targets.cpu().detach().numpy())

            val_loss /= len(val_loader)
            val_mae = mean_absolute_error(val_targets, val_predictions)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # GPU memory monitoring
            gpu_memory = 0.0
            if device == 'cuda':
                gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB

            # Record metrics
            epoch_metrics = TrainingMetrics(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                train_mae=train_mae,
                val_mae=val_mae,
                learning_rate=optimizer.param_groups[0]['lr'],
                gpu_memory_used=gpu_memory,
                training_time=(datetime.now() - start_time).total_seconds()
            )
            training_metrics.append(epoch_metrics)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                temp_dir = tempfile.gettempdir()
                torch.save(model.state_dict(), os.path.join(temp_dir, f'best_model_{training_id}.pth'))
            else:
                patience_counter += 1

            if patience_counter >= config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{config.epochs}, "
                          f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Load best model
        temp_dir = tempfile.gettempdir()
        model_path = os.path.join(temp_dir, f'best_model_{training_id}.pth')
        model.load_state_dict(torch.load(model_path))

        # Final evaluation
        model.eval()
        with torch.no_grad():
            final_predictions = []
            final_targets = []

            for inputs, targets in val_loader:
                outputs = model(inputs)
                final_predictions.extend(outputs.squeeze().cpu().detach().numpy())
                final_targets.extend(targets.cpu().detach().numpy())

        final_mae = mean_absolute_error(final_targets, final_predictions)
        final_rmse = np.sqrt(mean_squared_error(final_targets, final_predictions))

        return {
            'training_id': training_id,
            'final_metrics': {
                'mae': final_mae,
                'rmse': final_rmse,
                'best_val_loss': best_val_loss
            },
            'training_history': [vars(m) for m in training_metrics],
            'model': model,  # Return the trained model
            'model_path': model_path,
            'device': device,
            'epochs_trained': len(training_metrics)
        }

    def _train_distributed(self, training_data: Dict[str, Any],
                          model_config: Dict[str, Any], config: TrainingConfig,
                          training_id: str) -> Dict[str, Any]:
        """Train model with distributed GPUs."""
        logger.info(f"Starting distributed training with {len(self.gpu_devices)} GPUs")

        # For now, implement simple multi-GPU training
        # In production, would use torch.nn.DataParallel or DistributedDataParallel

        if len(self.gpu_devices) < 2:
            logger.warning("Distributed training requested but < 2 GPUs available")
            return self._train_single_gpu(training_data, model_config, config, training_id)

        # Split data across GPUs
        X_train, y_train = training_data['X_train'], training_data['y_train']
        split_size = len(X_train) // len(self.gpu_devices)

        gpu_results = []

        with ThreadPoolExecutor(max_workers=len(self.gpu_devices)) as executor:
            futures = []
            for i, gpu_id in enumerate(self.gpu_devices):
                start_idx = i * split_size
                end_idx = (i + 1) * split_size if i < len(self.gpu_devices) - 1 else len(X_train)

                gpu_data = {
                    'X_train': X_train[start_idx:end_idx],
                    'X_val': training_data['X_val'],
                    'y_train': y_train[start_idx:end_idx],
                    'y_val': training_data['y_val'],
                    'feature_names': training_data['feature_names']
                }

                gpu_config = TrainingConfig(**vars(config))
                gpu_config.use_gpu = True

                future = executor.submit(
                    self._train_single_gpu,
                    gpu_data, model_config, gpu_config,
                    f"{training_id}_gpu_{gpu_id}"
                )
                futures.append(future)

            # Collect results
            for future in futures:
                gpu_results.append(future.result())

        # Aggregate results (simple averaging for now)
        avg_mae = np.mean([r['final_metrics']['mae'] for r in gpu_results])
        avg_rmse = np.mean([r['final_metrics']['rmse'] for r in gpu_results])

        return {
            'training_id': training_id,
            'distributed': True,
            'n_gpus': len(self.gpu_devices),
            'gpu_results': gpu_results,
            'aggregated_metrics': {
                'mae': avg_mae,
                'rmse': avg_rmse
            }
        }

    def _create_model(self, model_config: Dict[str, Any], input_size: int) -> nn.Module:
        """Create model based on configuration."""
        model_type = model_config.get('type', 'lstm')

        if model_type == 'lstm':
            return self._create_lstm_model(input_size, model_config)
        elif model_type == 'cnn':
            return self._create_cnn_model(input_size, model_config)
        elif model_type == 'transformer':
            return self._create_transformer_model(input_size, model_config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _create_lstm_model(self, input_size: int, config: Dict[str, Any]) -> nn.Module:
        """Create LSTM model."""
        hidden_size = config.get('hidden_size', 64)
        num_layers = config.get('num_layers', 2)
        dropout = config.get('dropout', 0.2)

        return LSTMModel(input_size, hidden_size, num_layers, dropout)

    def _create_cnn_model(self, input_size: int, config: Dict[str, Any]) -> nn.Module:
        """Create CNN model."""
        return CNNModel(input_size)

    def _create_transformer_model(self, input_size: int, config: Dict[str, Any]) -> nn.Module:
        """Create Transformer model."""
        d_model = config.get('d_model', 64)
        nhead = config.get('nhead', 4)
        num_layers = config.get('num_layers', 2)
        
        return TransformerModel(input_size, d_model, nhead, num_layers)
        class TransformerModel(nn.Module):
            def __init__(self, input_size):
                super().__init__()
                self.encoder_layer = nn.TransformerEncoderLayer(
                    d_model=input_size, nhead=8, dim_feedforward=512
                )
                self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
                self.fc = nn.Linear(input_size, 1)

            def forward(self, x):
                x = self.transformer_encoder(x)
                x = self.fc(x[:, -1, :])
                return x

        return TransformerModel(input_size)

    def _detect_gpu_devices(self) -> List[int]:
        """Detect available GPU devices."""
        devices = []
        if torch.cuda.is_available():
            devices = list(range(torch.cuda.device_count()))
        return devices

    def _initialize_memory_manager(self) -> Dict[str, Any]:
        """Initialize GPU memory manager."""
        return {
            'max_memory_gb': 8.0,
            'current_memory_gb': 0.0,
            'memory_threshold': 0.9
        }

    def _setup_distributed_training(self) -> Dict[str, Any]:
        """Setup distributed training configuration."""
        return {
            'enabled': False,
            'world_size': 1,
            'rank': 0
        }

    def get_training_status(self, training_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a training job."""
        return self.active_trainings.get(training_id)

    def list_active_trainings(self) -> List[Dict[str, Any]]:
        """List all active training jobs."""
        return list(self.active_trainings.values())

    def get_model_registry(self) -> Dict[str, Any]:
        """Get model registry."""
        return self.model_registry

    def load_model(self, training_id: str) -> Optional[nn.Module]:
        """Load a trained model."""
        if training_id not in self.model_registry:
            return None

        registry_entry = self.model_registry[training_id]
        model_config = registry_entry['model_config']
        model_path = registry_entry['results']['model_path']

        try:
            model = self._create_model(model_config, 1)  # input_size placeholder
            model.load_state_dict(torch.load(model_path))
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Failed to load model {training_id}: {e}")
            return None

# Convenience functions
def create_gpu_training_service():
    """Create and configure GPU training service."""
    return GPUTrainingService()

def train_model_for_symbol(symbol: str, model_type: str = 'lstm',
                          epochs: int = 50) -> Dict[str, Any]:
    """Train a model for a symbol with default settings."""
    service = create_gpu_training_service()

    model_config = {'type': model_type}
    config = TrainingConfig(model_type=model_type, epochs=epochs)

    return service.train_model(symbol, model_config, config=config)