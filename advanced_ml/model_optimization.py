"""
Model Optimization Service

Automated model retraining, optimization, and performance monitoring.
Provides continuous learning and model improvement capabilities.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import json
import pickle
from pathlib import Path
import shutil
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Add paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)

class ModelOptimizationService:
    """
    Model optimization and continuous learning service.

    Provides:
    - Automated retraining pipelines
    - Performance monitoring and drift detection
    - Hyperparameter optimization
    - Model versioning and rollback
    - A/B testing framework
    - Automated feature engineering
    """

    def __init__(self, model_dir: str = "models", metrics_dir: str = "metrics"):
        """
        Initialize model optimization service.

        Args:
            model_dir: Directory to store trained models
            metrics_dir: Directory to store performance metrics
        """
        self.model_dir = Path(model_dir)
        self.metrics_dir = Path(metrics_dir)

        # Create directories
        self.model_dir.mkdir(exist_ok=True)
        self.metrics_dir.mkdir(exist_ok=True)

        self.optimization_config = {
            'retrain_threshold': 0.05,  # Retrain if performance drops by 5%
            'drift_threshold': 0.1,     # Drift detection threshold
            'min_samples_retrain': 1000, # Minimum samples for retraining
            'max_model_age_days': 30,   # Maximum model age before forced retrain
            'optuna_trials': 50,        # Number of optimization trials
            'cv_folds': 3,              # Cross-validation folds
            'test_size': 0.2            # Test set size
        }

        # Model registry
        self.model_registry = {}
        self.load_model_registry()

        logger.info("Model Optimization Service initialized")

    def automated_retraining_pipeline(self, model_name: str, X: pd.DataFrame,
                                    y: pd.Series, current_model: Any = None) -> Dict[str, Any]:
        """
        Automated retraining pipeline with performance monitoring.

        Args:
            model_name: Name of the model to retrain
            X: Feature matrix
            y: Target vector
            current_model: Current model instance (optional)

        Returns:
            Dictionary with retraining results
        """
        logger.info(f"Starting automated retraining for {model_name}")

        # Check if retraining is needed
        retrain_needed, reason = self._check_retrain_needed(model_name, X, y, current_model)

        if not retrain_needed:
            return {
                'retrained': False,
                'reason': reason,
                'model_name': model_name
            }

        # Perform hyperparameter optimization
        best_params = self.optimize_hyperparameters(model_name, X, y)

        # Train optimized model
        optimized_model = self._train_optimized_model(model_name, X, y, best_params)

        # Evaluate new model
        evaluation_results = self._evaluate_model(optimized_model, X, y)

        # Compare with current model
        comparison = {}
        if current_model:
            comparison = self._compare_models(current_model, optimized_model, X, y)

        # Save model if improved
        model_saved = False
        if not current_model or comparison.get('improvement', 0) > 0:
            self._save_model(model_name, optimized_model, evaluation_results, best_params)
            model_saved = True

        # Update metrics
        self._update_performance_metrics(model_name, evaluation_results)

        return {
            'retrained': True,
            'model_name': model_name,
            'best_params': best_params,
            'evaluation': evaluation_results,
            'comparison': comparison,
            'model_saved': model_saved,
            'reason': reason
        }

    def optimize_hyperparameters(self, model_name: str, X: pd.DataFrame,
                               y: pd.Series, n_trials: int = None) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.

        Args:
            model_name: Name of the model to optimize
            X: Feature matrix
            y: Target vector
            n_trials: Number of optimization trials

        Returns:
            Dictionary with best parameters
        """
        if n_trials is None:
            n_trials = self.optimization_config['optuna_trials']

        def objective(trial):
            if model_name.lower() == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
                }
                model = xgb.XGBRegressor(**params, random_state=42)

            elif model_name.lower() == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', -1, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
                }
                model = lgb.LGBMRegressor(**params, random_state=42)

            elif model_name.lower() == 'catboost':
                params = {
                    'iterations': trial.suggest_int('iterations', 50, 500),
                    'depth': trial.suggest_int('depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                    'border_count': trial.suggest_int('border_count', 32, 255),
                    'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1)
                }
                model = CatBoostRegressor(**params, random_state=42, verbose=False)

            else:
                raise ValueError(f"Unsupported model: {model_name}")

            # Cross-validation score
            tscv = TimeSeriesSplit(n_splits=self.optimization_config['cv_folds'])
            scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
            return -scores.mean()  # Minimize MSE

        # Create study
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner()
        )

        # Optimize
        study.optimize(objective, n_trials=n_trials)

        logger.info(f"Hyperparameter optimization completed for {model_name}")
        logger.info(f"Best score: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        return study.best_params

    def detect_model_drift(self, model_name: str, X_new: pd.DataFrame,
                         y_new: pd.Series, threshold: float = None) -> Dict[str, Any]:
        """
        Detect model drift using statistical tests.

        Args:
            model_name: Name of the model to check
            X_new: New feature data
            y_new: New target data
            threshold: Drift detection threshold

        Returns:
            Dictionary with drift detection results
        """
        if threshold is None:
            threshold = self.optimization_config['drift_threshold']

        # Load current model
        model_info = self.model_registry.get(model_name)
        if not model_info:
            return {'drift_detected': False, 'reason': 'Model not found in registry'}

        model = self._load_model_from_registry(model_name)
        if model is None:
            return {'drift_detected': False, 'reason': 'Could not load model'}

        # Get baseline performance
        baseline_metrics = model_info.get('metrics', {})

        # Calculate current performance
        y_pred = model.predict(X_new)
        current_mse = mean_squared_error(y_new, y_pred)
        current_mae = mean_absolute_error(y_new, y_pred)
        current_r2 = r2_score(y_new, y_pred)

        # Compare with baseline
        baseline_mse = baseline_metrics.get('mse', current_mse)

        # Calculate drift score (relative change)
        drift_score = abs(current_mse - baseline_mse) / baseline_mse

        drift_detected = drift_score > threshold

        return {
            'drift_detected': drift_detected,
            'drift_score': drift_score,
            'threshold': threshold,
            'current_metrics': {
                'mse': current_mse,
                'mae': current_mae,
                'r2': current_r2
            },
            'baseline_metrics': baseline_metrics,
            'model_name': model_name
        }

    def ab_testing_framework(self, model_a: Any, model_b: Any, X_test: pd.DataFrame,
                           y_test: pd.Series, confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        A/B testing framework for model comparison.

        Args:
            model_a: First model
            model_b: Second model
            X_test: Test features
            y_test: Test targets
            confidence_level: Statistical confidence level

        Returns:
            Dictionary with A/B test results
        """
        from scipy import stats

        # Get predictions
        pred_a = model_a.predict(X_test)
        pred_b = model_b.predict(X_test)

        # Calculate metrics
        mse_a = mean_squared_error(y_test, pred_a)
        mse_b = mean_squared_error(y_test, pred_b)

        mae_a = mean_absolute_error(y_test, pred_a)
        mae_b = mean_absolute_error(y_test, pred_b)

        r2_a = r2_score(y_test, pred_a)
        r2_b = r2_score(y_test, pred_b)

        # Calculate prediction differences
        diff_mse = mse_a - mse_b
        diff_mae = mae_a - mae_b
        diff_r2 = r2_b - r2_a  # Higher R2 is better

        # Statistical significance test (paired t-test on absolute errors)
        errors_a = np.abs(pred_a - y_test.values)
        errors_b = np.abs(pred_b - y_test.values)

        t_stat, p_value = stats.ttest_rel(errors_a, errors_b)

        # Determine winner
        if p_value < (1 - confidence_level):
            if mse_b < mse_a:  # Lower MSE is better
                winner = 'model_b'
            else:
                winner = 'model_a'
        else:
            winner = 'no_significant_difference'

        return {
            'model_a_metrics': {'mse': mse_a, 'mae': mae_a, 'r2': r2_a},
            'model_b_metrics': {'mse': mse_b, 'mae': mae_b, 'r2': r2_b},
            'differences': {'mse': diff_mse, 'mae': diff_mae, 'r2': diff_r2},
            'statistical_test': {
                't_statistic': t_stat,
                'p_value': p_value,
                'confidence_level': confidence_level,
                'significant': p_value < (1 - confidence_level)
            },
            'winner': winner,
            'recommendation': 'model_b' if winner == 'model_b' else 'model_a' if winner == 'model_a' else 'further_testing'
        }

    def automated_feature_engineering(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Automated feature engineering pipeline.

        Args:
            X: Input features
            y: Target variable (optional, for supervised feature selection)

        Returns:
            Enhanced feature matrix
        """
        X_engineered = X.copy()

        # Basic feature engineering
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        # Polynomial features
        for col in numeric_cols:
            X_engineered[f'{col}_squared'] = X[col] ** 2
            X_engineered[f'{col}_cubed'] = X[col] ** 3
            X_engineered[f'{col}_sqrt'] = np.sqrt(np.abs(X[col]))
            X_engineered[f'{col}_log'] = np.log1p(np.abs(X[col]))

        # Interaction features
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                X_engineered[f'{col1}_{col2}_ratio'] = X[col1] / (X[col2] + 1e-8)
                X_engineered[f'{col1}_{col2}_product'] = X[col1] * X[col2]

        # Rolling statistics (if time series)
        if isinstance(X.index, pd.DatetimeIndex):
            for col in numeric_cols:
                X_engineered[f'{col}_rolling_mean_7'] = X[col].rolling(window=7, min_periods=1).mean()
                X_engineered[f'{col}_rolling_std_7'] = X[col].rolling(window=7, min_periods=1).std()
                X_engineered[f'{col}_rolling_mean_30'] = X[col].rolling(window=30, min_periods=1).mean()
                X_engineered[f'{col}_rolling_std_30'] = X[col].rolling(window=30, min_periods=1).std()

        # Remove infinite and NaN values
        X_engineered = X_engineered.replace([np.inf, -np.inf], np.nan)
        X_engineered = X_engineered.fillna(X_engineered.mean())

        return X_engineered

    def model_versioning_system(self, model_name: str, model: Any,
                              metadata: Dict[str, Any]) -> str:
        """
        Save model with versioning.

        Args:
            model_name: Name of the model
            model: Trained model object
            metadata: Model metadata

        Returns:
            Version identifier
        """
        # Generate version
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"{model_name}_v_{timestamp}"

        # Create version directory
        version_dir = self.model_dir / version
        version_dir.mkdir(exist_ok=True)

        # Save model
        model_path = version_dir / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Save metadata
        metadata_path = version_dir / "metadata.json"
        metadata['version'] = version
        metadata['created_at'] = timestamp
        metadata['model_name'] = model_name

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        # Update registry
        self.model_registry[model_name] = {
            'latest_version': version,
            'versions': self.model_registry.get(model_name, {}).get('versions', []) + [version],
            'metadata': metadata
        }
        self._save_model_registry()

        logger.info(f"Model {model_name} version {version} saved")

        return version

    def rollback_model(self, model_name: str, version: str) -> bool:
        """
        Rollback to a specific model version.

        Args:
            model_name: Name of the model
            version: Version to rollback to

        Returns:
            Success status
        """
        if model_name not in self.model_registry:
            logger.error(f"Model {model_name} not found in registry")
            return False

        versions = self.model_registry[model_name]['versions']
        if version not in versions:
            logger.error(f"Version {version} not found for model {model_name}")
            return False

        # Update latest version
        self.model_registry[model_name]['latest_version'] = version
        self._save_model_registry()

        logger.info(f"Model {model_name} rolled back to version {version}")
        return True

    def _check_retrain_needed(self, model_name: str, X: pd.DataFrame,
                            y: pd.Series, current_model: Any) -> Tuple[bool, str]:
        """Check if model retraining is needed."""
        # Check minimum samples
        if len(X) < self.optimization_config['min_samples_retrain']:
            return False, f"Insufficient samples: {len(X)} < {self.optimization_config['min_samples_retrain']}"

        # Check model age
        model_info = self.model_registry.get(model_name, {})
        if model_info:
            created_at = model_info.get('metadata', {}).get('created_at')
            if created_at:
                try:
                    created_date = datetime.strptime(created_at, "%Y%m%d_%H%M%S")
                    age_days = (datetime.now() - created_date).days
                    if age_days > self.optimization_config['max_model_age_days']:
                        return True, f"Model age exceeded: {age_days} > {self.optimization_config['max_model_age_days']} days"
                except:
                    pass

        # Check performance degradation
        if current_model:
            y_pred = current_model.predict(X)
            current_mse = mean_squared_error(y, y_pred)

            baseline_mse = model_info.get('metrics', {}).get('mse')
            if baseline_mse:
                degradation = (current_mse - baseline_mse) / baseline_mse
                if degradation > self.optimization_config['retrain_threshold']:
                    return True, f"Performance degraded by {degradation:.1%}"

        return True, "Scheduled retraining"

    def _train_optimized_model(self, model_name: str, X: pd.DataFrame,
                             y: pd.Series, params: Dict[str, Any]) -> Any:
        """Train model with optimized parameters."""
        if model_name.lower() == 'xgboost':
            model = xgb.XGBRegressor(**params, random_state=42)
        elif model_name.lower() == 'lightgbm':
            model = lgb.LGBMRegressor(**params, random_state=42)
        elif model_name.lower() == 'catboost':
            model = CatBoostRegressor(**params, random_state=42, verbose=False)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        model.fit(X, y)
        return model

    def _evaluate_model(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        y_pred = model.predict(X)

        return {
            'mse': mean_squared_error(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred))
        }

    def _compare_models(self, model_a: Any, model_b: Any, X: pd.DataFrame,
                       y: pd.Series) -> Dict[str, Any]:
        """Compare two models."""
        metrics_a = self._evaluate_model(model_a, X, y)
        metrics_b = self._evaluate_model(model_b, X, y)

        improvement = metrics_a['mse'] - metrics_b['mse']  # Positive means B is better

        return {
            'model_a_metrics': metrics_a,
            'model_b_metrics': metrics_b,
            'improvement': improvement,
            'improvement_pct': improvement / metrics_a['mse'] if metrics_a['mse'] != 0 else 0,
            'better_model': 'model_b' if improvement > 0 else 'model_a'
        }

    def _save_model(self, model_name: str, model: Any, metrics: Dict[str, float],
                   params: Dict[str, Any]) -> None:
        """Save model to registry."""
        metadata = {
            'metrics': metrics,
            'params': params,
            'saved_at': datetime.now().isoformat()
        }

        version = self.model_versioning_system(model_name, model, metadata)

    def _update_performance_metrics(self, model_name: str, metrics: Dict[str, float]) -> None:
        """Update performance metrics history."""
        metrics_file = self.metrics_dir / f"{model_name}_metrics.json"

        # Load existing metrics
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                history = json.load(f)
        else:
            history = []

        # Add new metrics
        entry = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        history.append(entry)

        # Keep only last 100 entries
        history = history[-100:]

        # Save
        with open(metrics_file, 'w') as f:
            json.dump(history, f, indent=2, default=str)

    def _load_model_from_registry(self, model_name: str) -> Any:
        """Load model from registry."""
        model_info = self.model_registry.get(model_name)
        if not model_info:
            return None

        version = model_info.get('latest_version')
        if not version:
            return None

        model_path = self.model_dir / version / "model.pkl"
        if model_path.exists():
            with open(model_path, 'rb') as f:
                return pickle.load(f)

        return None

    def load_model_registry(self) -> None:
        """Load model registry from disk."""
        registry_file = self.model_dir / "model_registry.json"
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                self.model_registry = json.load(f)

    def _save_model_registry(self) -> None:
        """Save model registry to disk."""
        registry_file = self.model_dir / "model_registry.json"
        with open(registry_file, 'w') as f:
            json.dump(self.model_registry, f, indent=2, default=str)