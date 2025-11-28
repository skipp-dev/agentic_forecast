"""
AutoML Service

Automated Machine Learning for IB Forecast system.
Provides automated model selection, hyperparameter tuning, and pipeline optimization.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import json
import asyncio
from concurrent.futures import ProcessPoolExecutor
import optuna
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.forecast_agent import ForecastAgent
from data.feature_store import FeatureStore

logger = logging.getLogger(__name__)

class AutoMLService:
    """
    Automated Machine Learning service for model selection and optimization.

    Features:
    - Automated model selection from multiple algorithms
    - Hyperparameter optimization using Optuna
    - Ensemble model creation
    - Model performance comparison
    - Automated feature selection
    - Pipeline optimization
    """

    def __init__(self, feature_store: Optional[FeatureStore] = None):
        """
        Initialize AutoML service.

        Args:
            feature_store: Feature store instance
        """
        self.feature_store = feature_store or FeatureStore()

        # Model configurations
        self.model_configs = {
            'linear': {
                'class': LinearRegression,
                'params': {}
            },
            'ridge': {
                'class': Ridge,
                'params': {'alpha': [0.1, 1.0, 10.0, 100.0]}
            },
            'lasso': {
                'class': Lasso,
                'params': {'alpha': [0.001, 0.01, 0.1, 1.0]}
            },
            'rf': {
                'class': RandomForestRegressor,
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'gb': {
                'class': GradientBoostingRegressor,
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            },
            'xgb': {
                'class': xgb.XGBRegressor,
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'lgb': {
                'class': lgb.LGBMRegressor,
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [5, 10, -1],
                    'num_leaves': [20, 31, 50]
                }
            },
            'catboost': {
                'class': CatBoostRegressor,
                'params': {
                    'iterations': [100, 200, 500],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'depth': [4, 6, 8],
                    'verbose': [False]
                }
            },
            'extra_trees': {
                'class': ExtraTreesRegressor,
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'svr': {
                'class': SVR,
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            },
            'mlp': {
                'class': MLPRegressor,
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                    'activation': ['relu', 'tanh'],
                    'learning_rate': ['constant', 'adaptive'],
                    'max_iter': [200, 500]
                }
            }
        }

        # AutoML configuration
        self.automl_config = {
            'max_trials': 50,
            'cv_folds': 5,
            'optimization_timeout': 3600,  # 1 hour
            'ensemble_size': 5,
            'feature_selection_ratio': 0.8
        }

        logger.info("AutoML Service initialized")

    async def run_automl_pipeline(self, symbol: str, target_horizon: int = 1,
                                start_date: Optional[datetime] = None,
                                end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Run complete AutoML pipeline for a given symbol and horizon.

        Args:
            symbol: Trading symbol
            target_horizon: Forecast horizon in periods
            start_date: Start date for training data
            end_date: End date for training data

        Returns:
            Dictionary with AutoML results
        """
        logger.info(f"Starting AutoML pipeline for {symbol}, horizon {target_horizon}")

        # Get training data
        train_data = await self._get_training_data(symbol, start_date, end_date)
        if train_data.empty:
            return {'status': 'error', 'message': 'No training data available'}

        # Prepare features and target
        X, y = self._prepare_features_target(train_data, target_horizon)
        if X.empty or len(y) == 0:
            return {'status': 'error', 'message': 'Insufficient data for training'}

        # Feature selection
        selected_features = await self._select_features(X, y)

        # Model selection and optimization
        model_results = await self._optimize_models(X[selected_features], y)

        # Create ensemble
        ensemble_model = await self._create_ensemble(model_results, X[selected_features], y)

        # Evaluate final model
        evaluation = await self._evaluate_model(ensemble_model, X[selected_features], y)

        result = {
            'status': 'success',
            'symbol': symbol,
            'horizon': target_horizon,
            'data_points': len(X),
            'selected_features': selected_features.tolist(),
            'model_results': model_results,
            'ensemble_model': ensemble_model,
            'evaluation': evaluation,
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"AutoML pipeline completed for {symbol}")
        return result

    async def optimize_single_model(self, model_name: str, X: pd.DataFrame, y: pd.Series,
                                  n_trials: int = 20) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a single model.

        Args:
            model_name: Name of the model to optimize
            X: Feature matrix
            y: Target vector
            n_trials: Number of optimization trials

        Returns:
            Dictionary with optimization results
        """
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")

        model_config = self.model_configs[model_name]

        def objective(trial):
            params = {}
            for param_name, param_values in model_config['params'].items():
                if isinstance(param_values[0], int):
                    params[param_name] = trial.suggest_int(param_name, min(param_values), max(param_values))
                elif isinstance(param_values[0], float):
                    params[param_name] = trial.suggest_float(param_name, min(param_values), max(param_values))
                else:
                    params[param_name] = trial.suggest_categorical(param_name, param_values)

            model = model_config['class'](**params)

            # Use time series cross-validation
            tscv = TimeSeriesSplit(n_splits=self.automl_config['cv_folds'])
            scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
            return -scores.mean()  # Minimize MSE

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_model = model_config['class'](**best_params)

        # Train final model
        best_model.fit(X, y)

        return {
            'model_name': model_name,
            'best_params': best_params,
            'best_score': study.best_value,
            'n_trials': n_trials,
            'model': best_model,
            'feature_importance': self._get_feature_importance(best_model, X.columns) if hasattr(best_model, 'feature_importances_') else None
        }

    async def _get_training_data(self, symbol: str, start_date: Optional[datetime],
                               end_date: Optional[datetime]) -> pd.DataFrame:
        """Get training data for AutoML."""
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365)  # 1 year of data

        # Query feature store for historical data
        features_df = self.feature_store.retrieve_features(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )

        return features_df

    def _prepare_features_target(self, data: pd.DataFrame, horizon: int) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for training."""
        # Assume the data has OHLC and technical indicators
        # Target is the future return after 'horizon' periods

        if 'close' not in data.columns:
            return pd.DataFrame(), pd.Series()

        # Calculate future returns as target
        data = data.copy()
        data['target'] = data['close'].shift(-horizon) / data['close'] - 1

        # Remove rows with NaN target
        data = data.dropna(subset=['target'])

        # Features are all columns except target and future data
        feature_cols = [col for col in data.columns if col != 'target' and not col.startswith('future_')]
        X = data[feature_cols]
        y = data['target']

        return X, y

    async def _select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.Index:
        """Select most important features."""
        # Use correlation and feature importance for selection
        correlations = X.corrwith(y).abs().sort_values(ascending=False)

        # Select top features based on correlation
        n_features = int(len(X.columns) * self.automl_config['feature_selection_ratio'])
        selected_features = correlations.head(n_features).index

        return selected_features

    async def _optimize_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Optimize multiple models in parallel."""
        model_names = list(self.model_configs.keys())
        results = {}

        # Run optimization for each model
        for model_name in model_names:
            try:
                logger.info(f"Optimizing {model_name}...")
                result = await self.optimize_single_model(model_name, X, y, n_trials=10)
                results[model_name] = result
                logger.info(f"{model_name} optimization completed: {result['best_score']:.4f}")
            except Exception as e:
                logger.error(f"Failed to optimize {model_name}: {e}")
                results[model_name] = {'error': str(e)}

        return results

    async def _create_ensemble(self, model_results: Dict[str, Any], X: pd.DataFrame,
                             y: pd.Series) -> Dict[str, Any]:
        """Create ensemble model from optimized individual models."""
        # Select top performing models
        valid_results = {k: v for k, v in model_results.items() if 'best_score' in v}
        sorted_models = sorted(valid_results.items(), key=lambda x: x[1]['best_score'])

        # Take top N models for ensemble
        top_models = sorted_models[:self.automl_config['ensemble_size']]
        ensemble_models = [result['model'] for _, result in top_models]

        # Create simple average ensemble
        class EnsembleModel:
            def __init__(self, models):
                self.models = models

            def predict(self, X):
                predictions = np.array([model.predict(X) for model in self.models])
                return np.mean(predictions, axis=0)

            def fit(self, X, y):
                # Ensemble doesn't need fitting
                pass

        ensemble = EnsembleModel(ensemble_models)

        return {
            'ensemble_type': 'simple_average',
            'models': [result['model_name'] for _, result in top_models],
            'weights': [1.0 / len(ensemble_models)] * len(ensemble_models),
            'model': ensemble
        }

    async def _evaluate_model(self, model_info: Dict[str, Any], X: pd.DataFrame,
                            y: pd.Series) -> Dict[str, Any]:
        """Evaluate model performance."""
        model = model_info['model']

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.automl_config['cv_folds'])

        predictions = []
        actuals = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            predictions.extend(pred)
            actuals.extend(y_test)

        # Calculate metrics
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': predictions,
            'actuals': actuals
        }

    def _get_feature_importance(self, model, feature_names) -> Dict[str, float]:
        """Get feature importance from model."""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        else:
            return {}

        return dict(zip(feature_names, importance))

    async def run_hyperparameter_search(self, model_name: str, X: pd.DataFrame, y: pd.Series,
                                      search_space: Dict[str, Any], n_trials: int = 50) -> Dict[str, Any]:
        """
        Run advanced hyperparameter search with custom search space.

        Args:
            model_name: Name of the model
            X: Feature matrix
            y: Target vector
            search_space: Custom hyperparameter search space
            n_trials: Number of trials

        Returns:
            Dictionary with search results
        """
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")

        model_class = self.model_configs[model_name]['class']

        def objective(trial):
            params = {}
            for param_name, param_config in search_space.items():
                param_type = param_config.get('type', 'float')

                if param_type == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['low'],
                        param_config['high']
                    )
                elif param_type == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config['choices']
                    )

            model = model_class(**params)

            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
            return -scores.mean()

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, timeout=self.automl_config['optimization_timeout'])

        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': n_trials,
            'study': study
        }

    async def create_neural_ensemble(self, X: pd.DataFrame, y: pd.Series,
                                   n_models: int = 5) -> Dict[str, Any]:
        """
        Create neural network ensemble with different architectures.

        Args:
            X: Feature matrix
            y: Target vector
            n_models: Number of models in ensemble

        Returns:
            Dictionary with neural ensemble
        """
        class SimpleNN(nn.Module):
            def __init__(self, input_size, hidden_size, output_size=1):
                super(SimpleNN, self).__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_size, hidden_size//2),
                    nn.ReLU(),
                    nn.Linear(hidden_size//2, output_size)
                )

            def forward(self, x):
                return self.layers(x)

        models = []
        for i in range(n_models):
            hidden_size = np.random.choice([32, 64, 128, 256])
            model = SimpleNN(X.shape[1], hidden_size)

            # Simple training (in practice, use proper training loop)
            # This is a simplified version for demonstration

            models.append(model)

        class NeuralEnsemble:
            def __init__(self, models):
                self.models = models

            def predict(self, X):
                if isinstance(X, pd.DataFrame):
                    X_tensor = torch.FloatTensor(X.values)
                else:
                    X_tensor = torch.FloatTensor(X)

                predictions = []
                for model in self.models:
                    model.eval()
                    with torch.no_grad():
                        pred = model(X_tensor).numpy().flatten()
                        predictions.append(pred)

                return np.mean(predictions, axis=0)

        ensemble = NeuralEnsemble(models)

        return {
            'ensemble_type': 'neural_ensemble',
            'n_models': n_models,
            'model': ensemble
        }