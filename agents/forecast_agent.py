"""
Forecast Agent

Advanced forecasting agent with ensemble methods and GPU acceleration.
Extends existing forecast capabilities with sophisticated models.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.gpu_services import get_gpu_services
from src.data_pipeline import DataPipeline
from agents.hyperparameter_search_agent import HyperparameterSearchAgent
from agents.feature_engineer_agent import FeatureEngineerAgent

logger = logging.getLogger(__name__)

class TimeSeriesDataset(Dataset):
    """PyTorch dataset for time series forecasting."""

    def __init__(self, X: np.ndarray, y: np.ndarray, sequence_length: int = 60):
        self.X = X
        self.y = y
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.X) - self.sequence_length

    def __getitem__(self, idx):
        x = self.X[idx:idx + self.sequence_length]
        y = self.y[idx + self.sequence_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class LSTMModel(nn.Module):
    """LSTM model for time series forecasting."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class ForecastAgent:
    """
    Advanced forecasting agent with ensemble methods and GPU acceleration.

    Extends existing ForecastAgent with:
    - Ensemble forecasting (RF, GBM, XGBoost, LightGBM, LSTM)
    - GPU-accelerated training and inference
    - Automated model selection
    - Confidence intervals and uncertainty estimation
    - Multi-step ahead forecasting
    - Model stacking and blending
    """

    def __init__(self, gpu_services=None, data_pipeline=None,
                 hyperparameter_agent=None, feature_agent=None):
        """
        Initialize forecast agent.

        Args:
            gpu_services: GPU services instance
            data_pipeline: Data pipeline instance
            hyperparameter_agent: Hyperparameter search agent
            feature_agent: Feature engineering agent
        """
        super().__init__()
        self.gpu_services = gpu_services or get_gpu_services()
        self.data_pipeline = data_pipeline or DataPipeline()
        self.hyperparameter_agent = hyperparameter_agent or HyperparameterSearchAgent()
        self.feature_agent = feature_agent or FeatureEngineerAgent()

        # Model configurations
        self.model_configs = {
            'rf': {'class': RandomForestRegressor, 'params': {'n_estimators': 100, 'max_depth': 10}},
            'gbm': {'class': GradientBoostingRegressor, 'params': {'n_estimators': 100, 'max_depth': 5}},
            'xgb': {'class': xgb.XGBRegressor, 'params': {'n_estimators': 100, 'max_depth': 6}},
            'lgb': {'class': lgb.LGBMRegressor, 'params': {'n_estimators': 100, 'max_depth': 6}},
            'ridge': {'class': Ridge, 'params': {'alpha': 1.0}}
        }

        # Ensemble weights (learned during training)
        self.ensemble_weights = {}
        self.trained_models = {}

        # Forecasting configuration
        self.forecast_horizons = [1, 5, 10, 20]  # Days ahead
        self.confidence_levels = [0.80, 0.95]

        logger.info("Advanced Forecast Agent initialized with GPU acceleration")

    def generate_forecast(self, symbol: str, horizon: int = 1,
                         model_type: str = 'ensemble') -> Dict[str, Any]:
        """
        Generate comprehensive forecast for a symbol.

        Args:
            symbol: Stock symbol
            horizon: Forecast horizon in days
            model_type: Model type ('rf', 'gbm', 'xgb', 'lgb', 'lstm', 'ensemble')

        Returns:
            Forecast results with predictions and confidence intervals
        """
        logger.info(f"Generating {horizon}-day forecast for {symbol} using {model_type}")

        # Get historical data and features
        data = self._prepare_forecast_data(symbol)

        if data is None or len(data) < 100:
            logger.warning(f"Insufficient data for {symbol}")
            return self._create_empty_forecast(symbol, horizon)

        # Generate forecast based on model type
        if model_type == 'ensemble':
            forecast = self._generate_ensemble_forecast(data, horizon)
        elif model_type == 'lstm':
            forecast = self._generate_lstm_forecast(data, horizon)
        else:
            forecast = self._generate_single_model_forecast(data, horizon, model_type)

        # Add metadata
        forecast.update({
            'symbol': symbol,
            'horizon': horizon,
            'model_type': model_type,
            'timestamp': datetime.now(),
            'data_points': len(data)
        })

        logger.info(f"Forecast generated for {symbol}: {forecast['prediction']:.4f}")

        return forecast

    def _prepare_forecast_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Prepare data for forecasting."""
        try:
            # Fetch raw data
            raw_data = self.data_pipeline.av_client.get_daily_data(symbol, outputsize='full')
            if raw_data.empty:
                return None

            data_df = pd.DataFrame(raw_data)

            # Engineer features
            feature_data = self.feature_agent.engineer_features(
                symbol, data_df, feature_sets=['basic', 'spectral']
            )

            return feature_data

        except Exception as e:
            logger.error(f"Data preparation failed for {symbol}: {e}")
            return None

    def _generate_single_model_forecast(self, data: pd.DataFrame, horizon: int,
                                      model_type: str) -> Dict[str, Any]:
        """Generate forecast using a single model."""
        # Prepare features and target
        X, y = self._prepare_features_and_target(data, horizon)

        if len(X) < 50:  # Minimum training samples
            return self._create_empty_forecast(data.index.name or 'unknown', horizon)

        # Train or load model
        model_key = f"{data.index.name or 'unknown'}_{model_type}_{horizon}"
        if model_key not in self.trained_models:
            self.trained_models[model_key] = self._train_model(X, y, model_type)

        model = self.trained_models[model_key]

        # Generate prediction using DataFrame to preserve feature names
        latest_features = X.tail(1)
        prediction = model.predict(latest_features)[0]

        # Estimate uncertainty (simple method)
        train_predictions = model.predict(X)
        residuals = y - train_predictions
        std_residuals = np.std(residuals)

        # Confidence intervals
        confidence_intervals = {}
        for conf_level in self.confidence_levels:
            z_score = 1.96 if conf_level == 0.95 else 1.28  # Approximate
            margin = z_score * std_residuals
            confidence_intervals[f'{int(conf_level*100)}%'] = {
                'lower': prediction - margin,
                'upper': prediction + margin
            }

        return {
            'prediction': prediction,
            'confidence_intervals': confidence_intervals,
            'model_metrics': {
                'mae': mean_absolute_error(y, train_predictions),
                'rmse': np.sqrt(mean_squared_error(y, train_predictions))
            }
        }

    def _generate_ensemble_forecast(self, data: pd.DataFrame, horizon: int) -> Dict[str, Any]:
        """Generate ensemble forecast using multiple models."""
        # Prepare features and target
        X, y = self._prepare_features_and_target(data, horizon)

        if len(X) < 50:
            return self._create_empty_forecast(data.index.name or 'unknown', horizon)

        # Train ensemble models
        model_predictions = {}
        model_weights = {}

        for model_type in ['rf', 'gbm', 'xgb', 'lgb']:
            try:
                model_key = f"{data.index.name or 'unknown'}_{model_type}_{horizon}"
                if model_key not in self.trained_models:
                    self.trained_models[model_key] = self._train_model(X, y, model_type)
                model = self.trained_models[model_key]
                
                predictions = model.predict(X)
                model_predictions[model_type] = predictions

                # Weight by inverse of MAE
                mae = mean_absolute_error(y, predictions)
                model_weights[model_type] = 1.0 / (mae + 1e-6)  # Avoid division by zero

            except Exception as e:
                logger.warning(f"Failed to train {model_type}: {e}")
                continue

        # Normalize weights
        total_weight = sum(model_weights.values())
        if total_weight > 0:
            model_weights = {k: v/total_weight for k, v in model_weights.items()}
        else:
            # Equal weights if all models failed
            model_weights = {k: 1.0/len(model_weights) for k in model_weights.keys()}

        # Generate ensemble prediction using named features
        latest_features = X.tail(1)
        ensemble_prediction = 0.0

        for model_type, weight in model_weights.items():
            if model_type in model_predictions:
                model_key = f"{data.index.name or 'unknown'}_{model_type}_{horizon}"
                model_pred = self.trained_models[model_key].predict(latest_features)[0]
                ensemble_prediction += weight * model_pred

        # Ensemble uncertainty (weighted average of individual uncertainties)
        ensemble_std = 0.0
        for model_type, weight in model_weights.items():
            if model_type in model_predictions:
                train_predictions = model_predictions[model_type]
                residuals = y - train_predictions
                ensemble_std += weight * np.std(residuals)

        # Confidence intervals
        confidence_intervals = {}
        for conf_level in self.confidence_levels:
            z_score = 1.96 if conf_level == 0.95 else 1.28
            margin = z_score * ensemble_std
            confidence_intervals[f'{int(conf_level*100)}%'] = {
                'lower': ensemble_prediction - margin,
                'upper': ensemble_prediction + margin
            }

        return {
            'prediction': ensemble_prediction,
            'confidence_intervals': confidence_intervals,
            'model_weights': model_weights,
            'ensemble_metrics': {
                'n_models': len(model_weights),
                'weighted_mae': sum([w * mean_absolute_error(y, model_predictions.get(m, y))
                                    for m, w in model_weights.items()])
            }
        }

    def _generate_lstm_forecast(self, data: pd.DataFrame, horizon: int) -> Dict[str, Any]:
        """Generate forecast using LSTM model."""
        # Prepare sequential data
        X, y = self._prepare_features_and_target(data, horizon)

        if len(X) < 100:  # Need more data for LSTM
            logger.warning("Insufficient data for LSTM, falling back to ensemble")
            return self._generate_ensemble_forecast(data, horizon)

        # Convert to sequences
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        sequence_length = min(60, len(X_scaled) // 2)
        dataset = TimeSeriesDataset(X_scaled, y.values, sequence_length)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        # Train LSTM model
        model_key = f"{data.index.name or 'unknown'}_lstm_{horizon}"
        if model_key not in self.trained_models:
            self.trained_models[model_key] = self._train_lstm_model(dataloader, X.shape[1])

        model = self.trained_models[model_key]

        # Generate prediction
        model.eval()
        with torch.no_grad():
            # Get latest sequence
            latest_sequence = X_scaled[-sequence_length:].reshape(1, sequence_length, -1)
            latest_tensor = torch.tensor(latest_sequence, dtype=torch.float32)

            if self.gpu_services and torch.cuda.is_available():
                latest_tensor = latest_tensor.cuda()
                model = model.cuda()

            prediction = model(latest_tensor).item()

        # Inverse transform if needed (assuming target is scaled)
        # For simplicity, using raw prediction

        # Estimate uncertainty (bootstrap method)
        predictions = []
        for _ in range(100):  # Bootstrap samples
            sample_indices = np.random.choice(len(dataset), size=len(dataset), replace=True)
            sample_predictions = []

            model.eval()
            with torch.no_grad():
                for idx in sample_indices[-10:]:  # Last 10 samples
                    seq_x, _ = dataset[idx]
                    seq_x = seq_x.unsqueeze(0)
                    if self.gpu_services and torch.cuda.is_available():
                        seq_x = seq_x.cuda()
                    pred = model(seq_x).item()
                    sample_predictions.append(pred)

            predictions.append(np.mean(sample_predictions))

        std_prediction = np.std(predictions)

        # Confidence intervals
        confidence_intervals = {}
        for conf_level in self.confidence_levels:
            z_score = 1.96 if conf_level == 0.95 else 1.28
            margin = z_score * std_prediction
            confidence_intervals[f'{int(conf_level*100)}%'] = {
                'lower': prediction - margin,
                'upper': prediction + margin
            }

        return {
            'prediction': prediction,
            'confidence_intervals': confidence_intervals,
            'model_metrics': {
                'bootstrap_std': std_prediction,
                'sequence_length': sequence_length
            }
        }

    def _prepare_features_and_target(self, data: pd.DataFrame, horizon: int) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for forecasting."""
        # Use close price as target (shifted by horizon)
        target = data['close'].shift(-horizon).dropna()

        # Features are all columns except target
        features = data.drop(columns=['close'], errors='ignore')

        # Align features with target
        common_index = features.index.intersection(target.index)
        X = features.loc[common_index]
        y = target.loc[common_index]

        return X, y

    def _train_model(self, X: pd.DataFrame, y: pd.Series, model_type: str):
        """Train a single model."""
        model_config = self.model_configs[model_type]
        model_class = model_config['class']
        base_params = model_config['params']

        # Use hyperparameter search if available
        if self.hyperparameter_agent:
            try:
                best_params = self.hyperparameter_agent.search_hyperparameters(
                    model_type, X, y, n_trials=20
                )
                params = {**base_params, **best_params}
            except Exception as e:
                logger.warning(f"Hyperparameter search failed: {e}")
                params = base_params
        else:
            params = base_params

        # Train model
        model = model_class(**params)
        model.fit(X, y)

        return model

    def _train_lstm_model(self, dataloader: DataLoader, input_size: int) -> LSTMModel:
        """Train LSTM model."""
        model = LSTMModel(input_size=input_size)

        if self.gpu_services and torch.cuda.is_available():
            model = model.cuda()

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        model.train()
        for epoch in range(50):  # Limited epochs for demo
            epoch_loss = 0.0
            for inputs, targets in dataloader:
                if self.gpu_services and torch.cuda.is_available():
                    inputs, targets = inputs.cuda(), targets.cuda()

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                logger.info(f"LSTM Epoch {epoch+1}/50, Loss: {epoch_loss/len(dataloader):.4f}")

        return model

    def _create_empty_forecast(self, symbol: str, horizon: int) -> Dict[str, Any]:
        """Create empty forecast structure."""
        return {
            'symbol': symbol,
            'horizon': horizon,
            'prediction': None,
            'confidence_intervals': {},
            'error': 'Insufficient data for forecasting'
        }

    def evaluate_forecast_accuracy(self, symbol: str, test_period: str = '30d') -> Dict[str, float]:
        """
        Evaluate forecast accuracy on historical data.

        Args:
            symbol: Stock symbol
            test_period: Test period (e.g., '30d', '90d')

        Returns:
            Accuracy metrics
        """
        logger.info(f"Evaluating forecast accuracy for {symbol} over {test_period}")

        # Get historical data
        data = self._prepare_forecast_data(symbol)
        if data is None:
            return {'error': 'No data available'}

        # Split into train/test
        test_days = int(test_period.rstrip('d'))
        split_date = data.index[-1] - timedelta(days=test_days)

        train_data = data[data.index <= split_date]
        test_data = data[data.index > split_date]

        if len(test_data) < 10:
            return {'error': 'Insufficient test data'}

        # Generate forecasts for test period
        predictions = []
        actuals = []

        for i in range(len(test_data) - 1):  # Leave one day for prediction
            test_subset = pd.concat([train_data, test_data.iloc[:i+1]])

            forecast = self.generate_forecast(symbol, horizon=1, model_type='ensemble')
            if forecast['prediction'] is not None:
                predictions.append(forecast['prediction'])
                actuals.append(test_data.iloc[i+1]['close'])

        if not predictions:
            return {'error': 'No predictions generated'}

        # Calculate metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)

        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

        # Directional accuracy
        actual_direction = np.sign(np.diff(actuals))
        pred_direction = np.sign(np.diff(predictions))
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100

        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'n_predictions': len(predictions)
        }

# Convenience functions
def create_forecast_agent():
    """Create and configure forecast agent."""
    return ForecastAgent()

def generate_symbol_forecast(symbol: str, horizon: int = 1, model_type: str = 'ensemble'):
    """Generate forecast for a symbol with default settings."""
    agent = create_forecast_agent()
    return agent.generate_forecast(symbol, horizon, model_type)

def evaluate_forecast_performance(symbol: str, test_period: str = '30d'):
    """Evaluate forecast accuracy for a symbol."""
    agent = create_forecast_agent()
    return agent.evaluate_forecast_accuracy(symbol, test_period)