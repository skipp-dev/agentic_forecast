#!/usr/bin/env python3
"""
Global Model Agent for Agentic Forecasting System

Implements global time series models that can learn across multiple symbols:
- NHITS-style architecture using PyTorch
- TFT-style temporal fusion (simplified)
- Cross-symbol learning capabilities

Phase 2 addition for advanced forecasting.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

class TimeSeriesDataset(Dataset):
    """Dataset for time series forecasting with lookback windows."""

    def __init__(self, data: pd.DataFrame, lookback: int = 30, horizon: int = 1,
                 target_col: str = 'close', feature_cols: List[str] = None):
        """
        Initialize time series dataset.

        Args:
            data: DataFrame with time series data
            lookback: Number of past time steps to use as input
            horizon: Number of future time steps to predict
            target_col: Name of target column
            feature_cols: List of feature columns to use
        """
        self.lookback = lookback
        self.horizon = horizon
        self.target_col = target_col
        self.feature_cols = feature_cols or [col for col in data.columns if col != target_col]

        # Prepare sequences
        self.X, self.y = self._create_sequences(data)

    def _create_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create input-output sequences from time series data."""
        X, y = [], []

        for i in range(len(data) - self.lookback - self.horizon + 1):
            # Input sequence
            x_seq = data.iloc[i:i+self.lookback][self.feature_cols].values
            X.append(x_seq)

            # Target sequence (predict horizon steps ahead)
            y_seq = data.iloc[i+self.lookback:i+self.lookback+self.horizon][self.target_col].values
            y.append(y_seq[0])  # Take first horizon step for simplicity

        return np.array(X), np.array(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

class GlobalNHITSModel(nn.Module):
    """
    Simplified NHITS-style model for global forecasting.
    Neural Hierarchical Interpolation for Time Series.
    """

    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 1,
                 output_size: int = 1, dropout: float = 0.1):
        """
        Initialize simplified NHITS-style model for global forecasting.

        Args:
            input_size: Number of input features
            hidden_size: Hidden layer size
            num_layers: Number of LSTM layers
            output_size: Number of output values
            dropout: Dropout rate
        """
        super(GlobalNHITSModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # Simplified single-scale model for faster training
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the simplified model.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        batch_size, seq_len, input_size = x.shape

        # Process each time step through encoder
        encoded_steps = []
        for t in range(seq_len):
            step_input = x[:, t, :]  # (batch_size, input_size)
            encoded = self.encoder(step_input)  # (batch_size, hidden_size)
            encoded_steps.append(encoded)

        # Stack encoded steps: (batch_size, seq_len, hidden_size)
        encoded_seq = torch.stack(encoded_steps, dim=1)

        # LSTM processing
        lstm_out, _ = self.lstm(encoded_seq)  # (batch_size, seq_len, hidden_size)

        # Take last time step output
        last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_size)

        # Decode to prediction
        output = self.decoder(last_hidden)  # (batch_size, output_size)
        return output

class GlobalModelAgent:
    """
    Agent for training and using global time series models.
    Can learn patterns across multiple symbols for better generalization.
    """

    def __init__(self, model_type: str = 'nhits', device: str = 'cpu'):
        """
        Initialize global model agent.

        Args:
            model_type: Type of global model ('nhits', 'tft', etc.)
            device: Device to run model on ('cpu' or 'cuda')
        """
        self.model_type = model_type
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')

        self.model = None
        self.scaler = StandardScaler()
        self.lookback = 30
        self.horizon = 1

        # Model hyperparameters - reduced for faster testing
        self.hidden_size = 32
        self.num_layers = 1
        self.dropout = 0.1
        self.learning_rate = 0.001
        self.batch_size = 32
        self.num_epochs = 10  # Reduced from 50 for faster testing

        logger.info(f"Initialized GlobalModelAgent with {model_type} on {self.device}")

    def prepare_global_dataset(self, symbol_data: Dict[str, pd.DataFrame],
                              feature_cols: List[str] = None) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare global dataset from multiple symbols.

        Args:
            symbol_data: Dictionary mapping symbols to their feature DataFrames
            feature_cols: List of feature columns to use

        Returns:
            Tuple of (train_loader, val_loader)
        """
        all_sequences = []

        # Define available basic features - updated to match generated features
        if feature_cols is None:
            feature_cols = [
                'open', 'high', 'low', 'close', 'volume',
                'returns_1d', 'returns_5d', 'volatility_5d', 'volatility_10d',
                'sma_20', 'sma_50'
            ]

        for symbol, data in symbol_data.items():
            if data.empty or len(data) < self.lookback + self.horizon:
                logger.warning(f"Insufficient data for {symbol}, skipping")
                continue

            # Filter to only available columns
            available_cols = [col for col in feature_cols if col in data.columns]

            # Remove target column from features
            feature_cols_for_model = [col for col in available_cols if col != 'close']
            if len(available_cols) < len(feature_cols):
                logger.warning(f"Some features missing for {symbol}: {[col for col in feature_cols if col not in data.columns]}")

            if not available_cols:
                logger.warning(f"No valid features for {symbol}, skipping")
                continue

            # Create dataset for this symbol
            dataset = TimeSeriesDataset(data, self.lookback, self.horizon, 'close', feature_cols_for_model)

            if len(dataset) > 0:
                all_sequences.extend(list(dataset))
                logger.info(f"Added {len(dataset)} sequences from {symbol}")

        if not all_sequences:
            raise ValueError("No valid sequences found in symbol data")

        # Split into train/val (80/20)
        split_idx = int(len(all_sequences) * 0.8)
        train_sequences = all_sequences[:split_idx]
        val_sequences = all_sequences[split_idx:]

        # Create data loaders
        train_loader = DataLoader(train_sequences, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_sequences, batch_size=self.batch_size, shuffle=False)

        logger.info(f"Prepared global dataset: {len(train_sequences)} train, {len(val_sequences)} val sequences")
        return train_loader, val_loader

    def train_global_model(self, train_loader: DataLoader, val_loader: DataLoader,
                          input_size: int) -> Dict[str, Any]:
        """
        Train the global model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            input_size: Number of input features

        Returns:
            Training history and metrics
        """
        # Initialize model
        if self.model_type == 'nhits':
            self.model = GlobalNHITSModel(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                output_size=1,
                dropout=self.dropout
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.model.to(self.device)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        history = {'train_loss': [], 'val_loss': [], 'val_mae': []}

        for epoch in range(self.num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)

            # Validation
            self.model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                    outputs = self.model(X_batch)
                    loss = criterion(outputs.squeeze(), y_batch)
                    val_loss += loss.item()

                    # Ensure outputs is at least 1D for extend()
                    preds = outputs.squeeze().cpu().numpy()
                    if preds.ndim == 0:  # scalar
                        preds = [preds]
                    val_preds.extend(preds)
                    val_targets.extend(y_batch.cpu().numpy())

            val_loss /= len(val_loader)
            val_mae = mean_absolute_error(val_targets, val_preds)

            history['val_loss'].append(val_loss)
            history['val_mae'].append(val_mae)

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {train_loss:.4f}, "
                           f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")

        logger.info("Global model training completed")
        return history

    def predict(self, features_df: pd.DataFrame, feature_cols: List[str] = None) -> np.ndarray:
        """
        Generate predictions using the trained global model.

        Args:
            features_df: DataFrame with features
            feature_cols: List of feature columns to use

        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        if features_df.empty or len(features_df) < self.lookback:
            logger.warning("Insufficient data for prediction")
            return np.array([])

        # Use the exact features that were used during training
        feature_cols = ['open', 'high', 'low', 'volume', 'returns_1d', 'returns_5d', 'volatility_5d', 'sma_20', 'sma_50']

        # Create dataset for prediction
        dataset = TimeSeriesDataset(features_df, self.lookback, self.horizon, 'close', feature_cols)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                # Ensure outputs is at least 1D for extend()
                preds = outputs.squeeze().cpu().numpy()
                if preds.ndim == 0:  # scalar
                    preds = np.array([preds])
                predictions.extend(preds)

        return np.array(predictions)

    def save_model(self, path: str):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save")

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'input_size': self.model.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'lookback': self.lookback,
            'horizon': self.horizon
        }, path)

        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load a trained model from disk."""
        checkpoint = torch.load(path, map_location=self.device)

        input_size = checkpoint.get('input_size')
        if input_size is None:
            # Try to infer from saved state
            input_size = 10  # Default fallback

        if self.model_type == 'nhits':
            self.model = GlobalNHITSModel(
                input_size=input_size,
                hidden_size=checkpoint['hidden_size'],
                num_layers=checkpoint['num_layers'],
                output_size=1,
                dropout=self.dropout
            )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Restore parameters
        self.lookback = checkpoint['lookback']
        self.horizon = checkpoint['horizon']

        logger.info(f"Model loaded from {path}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            'model_type': self.model_type,
            'device': str(self.device),
            'lookback': self.lookback,
            'horizon': self.horizon,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'trained': self.model is not None
        }