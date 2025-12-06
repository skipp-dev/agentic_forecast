"""
LSTM Forecaster for Time Series Prediction

This module provides an LSTM-based forecaster for time series data.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

class LSTMModel(nn.Module):
    """LSTM Neural Network Model for Time Series Forecasting."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, output_size: int = 1, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # Take the last time step
        out = self.fc(out)
        return out

class LSTMForecaster:
    """
    LSTM-based Time Series Forecaster.

    Provides training and prediction capabilities for time series forecasting using LSTM networks.
    """

    def __init__(self, input_size: int = 10, hidden_size: int = 64, num_layers: int = 2,
                 output_size: int = 1, learning_rate: float = 0.001, epochs: int = 100,
                 batch_size: int = 32, device: str = 'auto'):
        """
        Initialize the LSTM Forecaster.

        Args:
            input_size: Number of input features
            hidden_size: Size of LSTM hidden layer
            num_layers: Number of LSTM layers
            output_size: Number of output predictions
            learning_rate: Learning rate for optimizer
            epochs: Number of training epochs
            batch_size: Batch size for training
            device: Device to run on ('cpu', 'cuda', or 'auto')
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        # Set device with robust CUDA checking
        if device == 'auto':
            self.device = self._get_safe_device()
        else:
            self.device = torch.device(device)

        # Initialize model
        self.model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        logger.info(f"LSTMForecaster initialized on device: {self.device}")

    def _get_safe_device(self) -> torch.device:
        """Get device with robust CUDA availability checking."""
        try:
            if not torch.cuda.is_available():
                return torch.device('cpu')

            # Try to access CUDA device
            torch.cuda.set_device(0)
            device = torch.device('cuda:0')

            # Test basic CUDA operation
            test_tensor = torch.tensor([1.0], device=device)
            test_result = test_tensor + 1
            del test_tensor, test_result
            torch.cuda.empty_cache()

            return device

        except Exception as e:
            logger.warning(f"CUDA device test failed: {e}, falling back to CPU")
            return torch.device('cpu')

    def prepare_data(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray, sequence_length: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare data for LSTM training by creating sequences.

        Args:
            X: Input features array or DataFrame
            y: Target values array
            sequence_length: Length of input sequences

        Returns:
            Tuple of (X_sequences, y_sequences) as tensors
        """
        # Convert DataFrame to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        X_sequences = []
        y_sequences = []

        for i in range(len(X) - sequence_length):
            X_sequences.append(X[i:i+sequence_length])
            y_sequences.append(y[i+sequence_length])

        X_sequences = torch.tensor(np.array(X_sequences), dtype=torch.float32).to(self.device)
        y_sequences = torch.tensor(np.array(y_sequences), dtype=torch.float32).to(self.device)

        return X_sequences, y_sequences

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None, sequence_length: int = 10) -> Dict[str, Any]:
        """
        Train the LSTM model.

        Args:
            X_train: Training input features
            y_train: Training target values
            X_val: Validation input features (optional)
            y_val: Validation target values (optional)
            sequence_length: Length of input sequences

        Returns:
            Dictionary with training results and metrics
        """
        # Prepare training data
        X_train_seq, y_train_seq = self.prepare_data(X_train, y_train, sequence_length)

        # Prepare validation data if provided
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self.prepare_data(X_val, y_val, sequence_length)
        else:
            X_val_seq, y_val_seq = None, None

        # Training loop
        best_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0

            # Mini-batch training
            for i in range(0, len(X_train_seq), self.batch_size):
                batch_X = X_train_seq[i:i+self.batch_size]
                batch_y = y_train_seq[i:i+self.batch_size]

                # Forward pass
                outputs = self.model(batch_X)
                loss = self.criterion(outputs.squeeze(), batch_y)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= (len(X_train_seq) // self.batch_size)

            # Validation
            if X_val_seq is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_seq)
                    val_loss = self.criterion(val_outputs.squeeze(), y_val_seq).item()

                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), 'best_lstm_model.pth')
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{self.epochs}, Loss: {epoch_loss:.4f}")

        # Load best model if validation was used
        if X_val_seq is not None:
            self.model.load_state_dict(torch.load('best_lstm_model.pth'))

        return {
            'final_loss': epoch_loss,
            'best_val_loss': best_loss if X_val_seq is not None else None,
            'epochs_trained': epoch
        }

    def predict(self, X: np.ndarray, sequence_length: int = 10) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            X: Input features for prediction
            sequence_length: Length of input sequences

        Returns:
            Array of predictions
        """
        self.model.eval()

        # Prepare data
        X_seq, _ = self.prepare_data(X, np.zeros(len(X)), sequence_length)  # Dummy y for preparation

        with torch.no_grad():
            predictions = self.model(X_seq).cpu().numpy()

        return predictions.flatten()

    def forecast(self, steps: int, recent_data: np.ndarray, sequence_length: int = 10) -> np.ndarray:
        """
        Generate multi-step forecast.

        Args:
            steps: Number of steps to forecast
            recent_data: Recent historical data to base forecast on
            sequence_length: Length of input sequences

        Returns:
            Array of forecasted values
        """
        forecasts = []

        current_sequence = recent_data[-sequence_length:].copy()

        for _ in range(steps):
            # Prepare current sequence
            X_seq = torch.tensor(current_sequence.reshape(1, sequence_length, -1), dtype=torch.float32).to(self.device)

            # Predict next value
            self.model.eval()
            with torch.no_grad():
                next_pred = self.model(X_seq).cpu().numpy()[0, 0]

            forecasts.append(next_pred)

            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = next_pred

        return np.array(forecasts)

    def save_model(self, path: str):
        """Save the trained model to disk."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'output_size': self.output_size
        }, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load a trained model from disk."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")

    @classmethod
    def load(cls, path: str, device: str = 'auto') -> 'LSTMForecaster':
        """Load a trained model from disk and return a new instance."""
        checkpoint = torch.load(path)
        
        # Create instance with saved params
        instance = cls(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers'],
            output_size=checkpoint['output_size'],
            device=device
        )
        
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"LSTMForecaster loaded from {path}")
        return instance

    def explain_prediction(self, X: np.ndarray, background_data: np.ndarray, sequence_length: int = 10) -> Dict[str, Any]:
        """
        Generate SHAP values for the LSTM model.
        
        Args:
            X: Input features to explain (raw 2D array, will be sequenced)
            background_data: Background data for DeepExplainer (raw 2D array)
            sequence_length: Length of input sequences
            
        Returns:
            Dictionary containing SHAP values and expected value
        """
        import shap
        
        self.model.eval()
        
        # Prepare sequences
        # We need a dummy y for prepare_data
        dummy_y_X = np.zeros(len(X))
        dummy_y_bg = np.zeros(len(background_data))
        
        # Ensure we have enough data for at least one sequence
        if len(X) <= sequence_length or len(background_data) <= sequence_length:
             logger.warning("Not enough data for SHAP explanation (need > sequence_length)")
             return {}

        X_seq, _ = self.prepare_data(X, dummy_y_X, sequence_length)
        bg_seq, _ = self.prepare_data(background_data, dummy_y_bg, sequence_length)
        
        # Use DeepExplainer
        # DeepExplainer expects tensors
        try:
            explainer = shap.DeepExplainer(self.model, bg_seq)
            shap_values = explainer.shap_values(X_seq)
            
            # shap_values might be a list (for multiple outputs) or array
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
                
            # Handle 4D output [n_samples, seq_len, n_features, 1]
            if len(shap_values.shape) == 4 and shap_values.shape[-1] == 1:
                shap_values = shap_values.squeeze(-1)
                
            # shap_values shape: [n_samples, sequence_length, input_size]
            
            # Aggregate over time dimension for simple feature importance
            # Sum absolute values or just sum? Usually sum of absolute values tells magnitude of impact
            # But for directional impact, simple sum might cancel out.
            # Let's provide mean absolute importance over time
            feature_importance = np.mean(np.abs(shap_values), axis=1) # [n_samples, input_size]
            
            return {
                "shap_values": shap_values, # 3D array
                "feature_importance": feature_importance, # 2D array
                "expected_value": explainer.expected_value
            }
            
        except Exception as e:
            logger.error(f"Failed to generate SHAP values: {e}")
            # Fallback or re-raise?
            # For now, return empty dict to avoid crashing the pipeline
            return {}
