"""
Advanced Ensembles Service

Sophisticated ensemble learning for IB Forecast system.
Provides stacking, blending, boosting, and custom ensemble methods.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import json
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import StackingRegressor, VotingRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)

class AdvancedEnsemblesService:
    """
    Advanced ensemble learning service.

    Provides:
    - Stacking ensembles
    - Blending ensembles
    - Neural network ensembles
    - Dynamic weighting
    - Meta-learning approaches
    - Ensemble selection
    """

    def __init__(self):
        """
        Initialize advanced ensembles service.
        """
        self.ensemble_config = {
            'cv_folds': 5,
            'stacking_cv_folds': 3,
            'neural_ensemble_size': 5,
            'dynamic_weighting_window': 50
        }

        logger.info("Advanced Ensembles Service initialized")

    def create_stacking_ensemble(self, base_models: List[Tuple[str, Any]], X: pd.DataFrame,
                               y: pd.Series, meta_model: Optional[Any] = None) -> Dict[str, Any]:
        """
        Create stacking ensemble.

        Args:
            base_models: List of (name, model) tuples
            X: Feature matrix
            y: Target vector
            meta_model: Meta-learner model (default: LinearRegression)

        Returns:
            Dictionary with stacking ensemble
        """
        if meta_model is None:
            meta_model = LinearRegression()

        # Create stacking regressor
        stacking_ensemble = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_model,
            cv=self.ensemble_config['stacking_cv_folds'],
            n_jobs=-1
        )

        # Train the ensemble
        stacking_ensemble.fit(X, y)

        # Evaluate base models
        base_scores = {}
        for name, model in base_models:
            scores = cross_val_score(model, X, y, cv=self.ensemble_config['cv_folds'],
                                   scoring='neg_mean_squared_error')
            base_scores[name] = -scores.mean()

        # Evaluate ensemble
        ensemble_scores = cross_val_score(stacking_ensemble, X, y, cv=self.ensemble_config['cv_folds'],
                                        scoring='neg_mean_squared_error')
        ensemble_score = -ensemble_scores.mean()

        return {
            'ensemble_type': 'stacking',
            'base_models': [name for name, _ in base_models],
            'meta_model': type(meta_model).__name__,
            'base_model_scores': base_scores,
            'ensemble_score': ensemble_score,
            'improvement': ensemble_score - min(base_scores.values()),
            'model': stacking_ensemble
        }

    def create_blending_ensemble(self, base_models: List[Tuple[str, Any]], X: pd.DataFrame,
                               y: pd.Series, validation_size: float = 0.2) -> Dict[str, Any]:
        """
        Create blending ensemble.

        Args:
            base_models: List of (name, model) tuples
            X: Feature matrix
            y: Target vector
            validation_size: Size of validation set for blending

        Returns:
            Dictionary with blending ensemble
        """
        from sklearn.model_selection import train_test_split

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_size, shuffle=False
        )

        # Train base models on training set
        trained_models = []
        val_predictions = []

        for name, model in base_models:
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            val_predictions.append(val_pred)
            trained_models.append((name, model))

        # Create meta-features for validation set
        meta_features_val = np.column_stack(val_predictions)

        # Train meta-model
        meta_model = Ridge(alpha=1.0)
        meta_model.fit(meta_features_val, y_val)

        # Create custom blending ensemble
        class BlendingEnsemble(BaseEstimator, RegressorMixin):
            def __init__(self, base_models, meta_model):
                self.base_models = base_models
                self.meta_model = meta_model

            def fit(self, X, y):
                # Already trained
                return self

            def predict(self, X):
                # Get base model predictions
                base_predictions = []
                for _, model in self.base_models:
                    pred = model.predict(X)
                    base_predictions.append(pred)

                # Create meta-features
                meta_features = np.column_stack(base_predictions)

                # Final prediction
                return self.meta_model.predict(meta_features)

        blending_ensemble = BlendingEnsemble(trained_models, meta_model)

        # Evaluate ensemble
        ensemble_pred = blending_ensemble.predict(X_val)
        from sklearn.metrics import mean_squared_error
        ensemble_score = mean_squared_error(y_val, ensemble_pred)

        return {
            'ensemble_type': 'blending',
            'base_models': [name for name, _ in base_models],
            'meta_model': type(meta_model).__name__,
            'validation_size': validation_size,
            'ensemble_score': ensemble_score,
            'model': blending_ensemble
        }

    def create_neural_ensemble(self, X: pd.DataFrame, y: pd.Series,
                             n_models: int = 5, hidden_sizes: List[int] = None) -> Dict[str, Any]:
        """
        Create neural network ensemble.

        Args:
            X: Feature matrix
            y: Target vector
            n_models: Number of neural networks in ensemble
            hidden_sizes: List of hidden layer sizes to choose from

        Returns:
            Dictionary with neural ensemble
        """
        if hidden_sizes is None:
            hidden_sizes = [32, 64, 128, 256]

        class SimpleNN(nn.Module):
            def __init__(self, input_size, hidden_size, output_size=1):
                super(SimpleNN, self).__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_size, hidden_size//2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size//2, output_size)
                )

            def forward(self, x):
                return self.layers(x)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X.values)
        y_tensor = torch.FloatTensor(y.values).reshape(-1, 1)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        models = []
        losses = []

        for i in range(n_models):
            # Random architecture
            hidden_size = np.random.choice(hidden_sizes)

            model = SimpleNN(X.shape[1], hidden_size)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            # Simple training
            model.train()
            final_loss = 0

            for epoch in range(50):  # Reduced epochs for demo
                epoch_loss = 0
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                final_loss = epoch_loss / len(dataloader)

            models.append(model)
            losses.append(final_loss)

        # Create ensemble class
        class NeuralEnsemble(BaseEstimator, RegressorMixin):
            def __init__(self, models):
                self.models = models

            def fit(self, X, y):
                # Already trained
                return self

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

        neural_ensemble = NeuralEnsemble(models)

        return {
            'ensemble_type': 'neural_ensemble',
            'n_models': n_models,
            'hidden_sizes': hidden_sizes,
            'final_losses': losses,
            'model': neural_ensemble
        }

    def create_dynamic_weighting_ensemble(self, base_models: List[Tuple[str, Any]],
                                        X: pd.DataFrame, y: pd.Series,
                                        window_size: int = None) -> Dict[str, Any]:
        """
        Create dynamic weighting ensemble that adapts weights based on recent performance.

        Args:
            base_models: List of (name, model) tuples
            X: Feature matrix
            y: Target vector
            window_size: Window size for weight calculation

        Returns:
            Dictionary with dynamic weighting ensemble
        """
        if window_size is None:
            window_size = self.ensemble_config['dynamic_weighting_window']

        # Train base models
        trained_models = []
        for name, model in base_models:
            model.fit(X, y)
            trained_models.append((name, model))

        # Calculate weights based on recent performance
        weights = self._calculate_dynamic_weights(trained_models, X, y, window_size)

        # Create voting ensemble with calculated weights
        voting_ensemble = VotingRegressor(
            estimators=trained_models,
            weights=list(weights.values())
        )

        return {
            'ensemble_type': 'dynamic_weighting',
            'base_models': list(weights.keys()),
            'weights': weights,
            'window_size': window_size,
            'model': voting_ensemble
        }

    def create_boosting_ensemble(self, base_models: List[Tuple[str, Any]], X: pd.DataFrame,
                               y: pd.Series, n_estimators: int = 50) -> Dict[str, Any]:
        """
        Create boosting ensemble.

        Args:
            base_models: List of (name, model) tuples
            X: Feature matrix
            y: Target vector
            n_estimators: Number of boosting iterations

        Returns:
            Dictionary with boosting ensemble
        """
        # Use the first base model as the base estimator
        base_estimator = base_models[0][1]

        # Create AdaBoost ensemble
        boosting_ensemble = AdaBoostRegressor(
            estimator=base_estimator,
            n_estimators=n_estimators,
            random_state=42
        )

        boosting_ensemble.fit(X, y)

        # Evaluate performance
        scores = cross_val_score(boosting_ensemble, X, y, cv=self.ensemble_config['cv_folds'],
                               scoring='neg_mean_squared_error')
        ensemble_score = -scores.mean()

        return {
            'ensemble_type': 'boosting',
            'base_model': base_models[0][0],
            'n_estimators': n_estimators,
            'ensemble_score': ensemble_score,
            'model': boosting_ensemble
        }

    def ensemble_selection(self, base_models: List[Tuple[str, Any]], X: pd.DataFrame,
                         y: pd.Series, selection_method: str = 'greedy') -> Dict[str, Any]:
        """
        Perform ensemble selection to choose optimal subset of models.

        Args:
            base_models: List of (name, model) tuples
            X: Feature matrix
            y: Target vector
            selection_method: Selection method ('greedy', 'forward', 'backward')

        Returns:
            Dictionary with selected ensemble
        """
        if selection_method == 'greedy':
            return self._greedy_ensemble_selection(base_models, X, y)
        elif selection_method == 'forward':
            return self._forward_ensemble_selection(base_models, X, y)
        elif selection_method == 'backward':
            return self._backward_ensemble_selection(base_models, X, y)
        else:
            raise ValueError(f"Unknown selection method: {selection_method}")

    def create_meta_learning_ensemble(self, base_models: List[Tuple[str, Any]],
                                    X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Create meta-learning ensemble that learns how to combine predictions.

        Args:
            base_models: List of (name, model) tuples
            X: Feature matrix
            y: Target vector

        Returns:
            Dictionary with meta-learning ensemble
        """
        from sklearn.model_selection import KFold

        # Generate meta-features using cross-validation
        kf = KFold(n_splits=self.ensemble_config['cv_folds'], shuffle=False)

        meta_features = []
        meta_targets = []

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            fold_predictions = []

            for name, model in base_models:
                model.fit(X_train, y_train)
                val_pred = model.predict(X_val)
                fold_predictions.append(val_pred)

            # Store meta-features and targets
            meta_features.extend(np.column_stack(fold_predictions))
            meta_targets.extend(y_val)

        # Train meta-model
        meta_model = Ridge(alpha=1.0)
        meta_model.fit(meta_features, meta_targets)

        # Train base models on full data
        trained_models = []
        for name, model in base_models:
            model.fit(X, y)
            trained_models.append((name, model))

        # Create meta-learning ensemble
        class MetaLearningEnsemble(BaseEstimator, RegressorMixin):
            def __init__(self, base_models, meta_model):
                self.base_models = base_models
                self.meta_model = meta_model

            def fit(self, X, y):
                # Already trained
                return self

            def predict(self, X):
                # Get base model predictions
                base_predictions = []
                for _, model in self.base_models:
                    pred = model.predict(X)
                    base_predictions.append(pred)

                # Create meta-features
                meta_features = np.column_stack(base_predictions)

                # Final prediction
                return self.meta_model.predict(meta_features)

        meta_ensemble = MetaLearningEnsemble(trained_models, meta_model)

        return {
            'ensemble_type': 'meta_learning',
            'base_models': [name for name, _ in base_models],
            'meta_model': type(meta_model).__name__,
            'cv_folds': self.ensemble_config['cv_folds'],
            'model': meta_ensemble
        }

    def _calculate_dynamic_weights(self, trained_models: List[Tuple[str, Any]],
                                 X: pd.DataFrame, y: pd.Series, window_size: int) -> Dict[str, float]:
        """Calculate dynamic weights based on recent performance."""
        weights = {}

        for name, model in trained_models:
            # Use rolling window to calculate recent performance
            predictions = model.predict(X)

            # Calculate rolling MSE
            errors = (predictions - y.values) ** 2
            rolling_mse = pd.Series(errors).rolling(window=window_size).mean()

            # Weight is inverse of recent MSE (higher weight for lower error)
            recent_mse = rolling_mse.dropna().iloc[-1]
            weight = 1.0 / (1.0 + recent_mse)  # Add 1 to avoid division by zero

            weights[name] = weight

        # Normalize weights
        total_weight = sum(weights.values())
        weights = {name: weight / total_weight for name, weight in weights.items()}

        return weights

    def _greedy_ensemble_selection(self, base_models: List[Tuple[str, Any]],
                                 X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Greedy ensemble selection."""
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import cross_val_predict

        selected_models = []
        best_score = float('inf')

        remaining_models = base_models.copy()

        while remaining_models:
            best_improvement = 0
            best_model = None

            for name, model in remaining_models:
                # Test adding this model
                test_models = selected_models + [(name, model)]

                if len(test_models) == 1:
                    # Single model
                    predictions = cross_val_predict(model, X, y, cv=3)
                    score = mean_squared_error(y, predictions)
                else:
                    # Ensemble prediction
                    cv_predictions = []
                    for _, m in test_models:
                        pred = cross_val_predict(m, X, y, cv=3)
                        cv_predictions.append(pred)

                    ensemble_pred = np.mean(cv_predictions, axis=0)
                    score = mean_squared_error(y, ensemble_pred)

                improvement = best_score - score

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_model = (name, model)

            if best_improvement > 0:
                selected_models.append(best_model)
                remaining_models.remove(best_model)
                best_score -= best_improvement
            else:
                break

        # Create final ensemble
        voting_ensemble = VotingRegressor(estimators=selected_models)

        return {
            'ensemble_type': 'greedy_selection',
            'selected_models': [name for name, _ in selected_models],
            'final_score': best_score,
            'model': voting_ensemble
        }

    def _forward_ensemble_selection(self, base_models: List[Tuple[str, Any]],
                                  X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Forward ensemble selection."""
        # Simplified forward selection
        return self._greedy_ensemble_selection(base_models, X, y)

    def _backward_ensemble_selection(self, base_models: List[Tuple[str, Any]],
                                   X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Backward ensemble selection."""
        # Start with all models and remove worst performers
        selected_models = base_models.copy()
        min_improvement = 0.001  # Minimum improvement threshold

        while len(selected_models) > 1:
            worst_model = None
            best_score_after_removal = float('inf')

            for i, (name, model) in enumerate(selected_models):
                # Test removing this model
                test_models = selected_models.copy()
                test_models.pop(i)

                # Evaluate ensemble without this model
                from sklearn.model_selection import cross_val_predict
                cv_predictions = []
                for _, m in test_models:
                    pred = cross_val_predict(m, X, y, cv=3)
                    cv_predictions.append(pred)

                ensemble_pred = np.mean(cv_predictions, axis=0)
                score = mean_squared_error(y, ensemble_pred)

                if score < best_score_after_removal:
                    best_score_after_removal = score
                    worst_model = i

            # Check if removal improves performance significantly
            current_score = self._evaluate_ensemble_score(selected_models, X, y)
            if current_score - best_score_after_removal > min_improvement:
                selected_models.pop(worst_model)
            else:
                break

        voting_ensemble = VotingRegressor(estimators=selected_models)

        return {
            'ensemble_type': 'backward_selection',
            'selected_models': [name for name, _ in selected_models],
            'final_score': self._evaluate_ensemble_score(selected_models, X, y),
            'model': voting_ensemble
        }

    def _evaluate_ensemble_score(self, models: List[Tuple[str, Any]],
                               X: pd.DataFrame, y: pd.Series) -> float:
        """Evaluate ensemble score."""
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import cross_val_predict

        cv_predictions = []
        for _, model in models:
            pred = cross_val_predict(model, X, y, cv=3)
            cv_predictions.append(pred)

        ensemble_pred = np.mean(cv_predictions, axis=0)
        return mean_squared_error(y, ensemble_pred)