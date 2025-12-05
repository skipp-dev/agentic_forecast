import logging
from typing import Dict, Any, Optional, List, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)

class EnsembleForecaster:
    """
    Ensemble forecaster combining multiple models.
    Supports simple averaging and weighted averaging.
    """
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize ensemble forecaster.
        
        Args:
            weights: Dictionary of weights for each model name. If None, equal weights are used.
        """
        self.models: Dict[str, Any] = {}
        self.weights = weights
        self.meta_model = None
        logger.info("EnsembleForecaster initialized")

    def add_base_model(self, name: str, model: Any):
        """
        Add a base model to the ensemble.
        
        Args:
            name: Name of the model
            model: Trained model instance (must have predict method)
        """
        self.models[name] = model
        logger.info(f"Added model {name} to ensemble")

    def train_ensemble(self, X, y):
        """
        Train the ensemble. 
        For simple averaging, this might be a no-op or used to train a meta-learner.
        Currently implemented as a no-op for simple averaging.
        """
        logger.info("Training ensemble (No-op for simple averaging)")
        # Future: Implement stacking/meta-learning here
        pass

    def predict(self, X) -> np.ndarray:
        """
        Generate predictions using the ensemble.
        
        Args:
            X: Input features
            
        Returns:
            Array of predictions
        """
        if not self.models:
            raise ValueError("No models in ensemble")

        predictions = []
        model_names = []
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                # Handle different output shapes
                if len(pred.shape) > 1:
                    pred = pred.flatten()
                predictions.append(pred)
                model_names.append(name)
            except Exception as e:
                logger.error(f"Prediction failed for model {name}: {e}")

        if not predictions:
            raise ValueError("All models failed to predict")

        predictions = np.array(predictions)
        
        # Apply weights if provided
        if self.weights:
            # Normalize weights for available models
            active_weights = [self.weights.get(name, 0.0) for name in model_names]
            total_weight = sum(active_weights)
            if total_weight > 0:
                normalized_weights = [w / total_weight for w in active_weights]
                weighted_avg = np.average(predictions, axis=0, weights=normalized_weights)
                return weighted_avg
            else:
                logger.warning("Weights sum to zero or missing, falling back to simple average")
        
        # Simple average
        return np.mean(predictions, axis=0)

    def explain_prediction(self, X, background_data=None) -> Dict[str, Any]:
        """
        Generate SHAP values for the ensemble predictions.
        
        Args:
            X: Input features to explain
            background_data: Optional background data for explainers (e.g. KernelExplainer)
            
        Returns:
            Dictionary containing:
            - shap_values: The aggregated SHAP values
            - expected_value: The aggregated expected value
            - feature_names: List of feature names (if X is DataFrame)
        """
        import shap
        import pandas as pd
        
        if not self.models:
            raise ValueError("No models in ensemble")
            
        all_shap_values = []
        all_expected_values = []
        model_names = []
        
        # Convert X to numpy if needed for consistency, but keep DataFrame for column names
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else [f"feature_{i}" for i in range(X_array.shape[1])]
        
        # Handle background data
        if background_data is not None:
            bg_array = background_data.values if isinstance(background_data, pd.DataFrame) else background_data
        else:
            # Use a small sample of X as background if not provided, or None
            bg_array = X_array[:10] if X_array.shape[0] > 10 else X_array

        for name, model in self.models.items():
            try:
                # Determine appropriate explainer
                explainer = None
                
                # Check for Tree-based models (sklearn, xgboost, lightgbm, catboost)
                is_tree = False
                model_type = str(type(model)).lower()
                if 'ensemble' in model_type or 'tree' in model_type or 'xgb' in model_type or 'lgb' in model_type or 'catboost' in model_type:
                    try:
                        explainer = shap.TreeExplainer(model)
                        is_tree = True
                    except Exception:
                        pass # Fallback to other explainers
                
                # Check for Linear models
                if not explainer and ('linear' in model_type or 'ridge' in model_type or 'lasso' in model_type):
                    try:
                        explainer = shap.LinearExplainer(model, bg_array)
                    except Exception:
                        pass

                # Fallback to KernelExplainer
                if not explainer:
                    # KernelExplainer needs a predict function
                    # We wrap the predict method to ensure it returns a 1D array
                    def predict_wrapper(data):
                        res = model.predict(data)
                        if len(res.shape) > 1:
                            return res.flatten()
                        return res
                        
                    explainer = shap.KernelExplainer(predict_wrapper, bg_array)

                # Calculate SHAP values
                # TreeExplainer returns different shapes depending on interaction values etc.
                # We want just the shap values matrix [n_samples, n_features]
                shap_vals = explainer.shap_values(X)
                
                # Handle different return types from shap_values
                if isinstance(shap_vals, list):
                    # For some classifiers/regressors it returns a list. For regression usually just one array.
                    # If it's a list (e.g. Random Forest Regressor sometimes?), take the first one if it matches shape
                    shap_vals = shap_vals[0]
                
                # Ensure shape is [n_samples, n_features]
                if len(shap_vals.shape) > 2:
                     # Sometimes returns [n_samples, n_features, 1]
                    shap_vals = shap_vals.squeeze()

                all_shap_values.append(shap_vals)
                
                # Get expected value
                ev = explainer.expected_value
                if isinstance(ev, list) or isinstance(ev, np.ndarray):
                     if np.size(ev) == 1:
                         ev = float(ev)
                     else:
                         ev = ev[0] # Take first if multiple
                all_expected_values.append(ev)
                
                model_names.append(name)
                
            except Exception as e:
                logger.warning(f"Could not explain model {name}: {e}")
                # If one fails, we might have to skip it or fail the whole thing. 
                # For now, let's skip and re-normalize weights later.
                continue

        if not all_shap_values:
            raise ValueError("Could not generate SHAP values for any model")

        # Aggregate
        all_shap_values = np.array(all_shap_values) # [n_models, n_samples, n_features]
        all_expected_values = np.array(all_expected_values) # [n_models]
        
        # Calculate weights
        active_weights = [self.weights.get(name, 0.0) if self.weights else 1.0 for name in model_names]
        total_weight = sum(active_weights)
        if total_weight == 0:
            normalized_weights = [1.0 / len(model_names)] * len(model_names)
        else:
            normalized_weights = [w / total_weight for w in active_weights]
            
        # Weighted average
        avg_shap_values = np.average(all_shap_values, axis=0, weights=normalized_weights)
        avg_expected_value = np.average(all_expected_values, axis=0, weights=normalized_weights)
        
        return {
            "shap_values": avg_shap_values,
            "expected_value": avg_expected_value,
            "feature_names": feature_names
        }

    def save(self, path: str):
        """Save ensemble to disk."""
        import joblib
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Ensemble saved to {path}")

    @classmethod
    def load(cls, path: str):
        """Load ensemble from disk."""
        import joblib
        return joblib.load(path)
