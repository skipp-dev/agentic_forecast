import shap
import pandas as pd
import numpy as np
from typing import Dict, Any, Union, List, Optional
import logging

logger = logging.getLogger(__name__)

class ExplainabilityAgent:
    """
    Agent for explaining model predictions using SHAP.
    Supports NeuralForecast, LinearRegression, EnsembleForecaster, and LSTMForecaster.
    """
    def __init__(self, model: Any, model_family: str, feature_names: list):
        self.model = model
        self.model_family = model_family
        self.feature_names = feature_names

    def explain(self, data: pd.DataFrame, sample_size: int = 100) -> Dict[str, Any]:
        """
        Generates SHAP values to explain model predictions.
        """
        # Sample data for explanation
        sample_data = data.head(sample_size).copy()
        
        # Check if model has built-in explanation capability (EnsembleForecaster, LSTMForecaster)
        if hasattr(self.model, 'explain_prediction'):
            try:
                logger.info(f"Using built-in explain_prediction for {self.model_family}")
                # Pass background data if needed (using the same sample data)
                explanation = self.model.explain_prediction(sample_data, background_data=sample_data)
                
                shap_values = explanation.get('shap_values')
                
                # If LSTM returns 3D shap values, we might need to flatten or use the feature_importance directly
                if len(np.array(shap_values).shape) == 3:
                    # [n_samples, sequence_length, n_features]
                    # Use the pre-calculated feature importance if available, or calculate mean abs
                    if 'feature_importance' in explanation:
                        # This is [n_samples, n_features] (aggregated over time)
                        # But shap_values usually expects [n_samples, n_features] for the summary plot logic below
                        # We will use the aggregated importance as "shap values" for the summary plot 
                        # (approximation for 2D visualization of 3D data)
                        shap_values = explanation['feature_importance'] 
                    else:
                        shap_values = np.mean(np.abs(shap_values), axis=1)

                # Ensure shap_values matches feature_names length
                if shap_values.shape[1] != len(self.feature_names):
                    logger.warning(f"Shape mismatch: SHAP {shap_values.shape[1]} vs Features {len(self.feature_names)}")
                    # Try to align if possible, or just truncate/pad? 
                    # Better to just proceed and let _generate_summary handle it or fail gracefully
                
            except Exception as e:
                logger.error(f"Built-in explanation failed: {e}. Falling back to KernelExplainer.")
                shap_values = self._explain_generic(sample_data)
        
        # Check if sklearn model
        elif self._is_sklearn_linear(self.model):
            # For sklearn models, use LinearExplainer
            background_data = sample_data[self.feature_names].values
            try:
                explainer = shap.LinearExplainer(self.model, background_data)
                shap_values = explainer.shap_values(sample_data[self.feature_names].values)
            except Exception as e:
                logger.warning(f"LinearExplainer failed: {e}. Falling back to KernelExplainer.")
                shap_values = self._explain_generic(sample_data)
        else:
            # Fallback to generic KernelExplainer (NeuralForecast etc.)
            shap_values = self._explain_generic(sample_data)
        
        # Generate summary statistics
        try:
            summary = self._generate_summary(shap_values, sample_data)
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            logger.error(f"shap_values shape: {np.array(shap_values).shape}")
            raise
        
        return {
            'shap_values': shap_values,
            'feature_importance': summary['feature_importance'],
            'summary_plot_data': summary['summary_plot_data'],
            'model_family': self.model_family,
            'sample_size': sample_size
        }

    def _is_sklearn_linear(self, model):
        try:
            from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
            return isinstance(model, (LinearRegression, Ridge, Lasso, ElasticNet))
        except ImportError:
            return False

    def _explain_generic(self, sample_data: pd.DataFrame) -> np.ndarray:
        """
        Generic explanation using KernelExplainer.
        """
        def predict_fn(X):
            # Convert to DataFrame if needed
            if isinstance(X, np.ndarray):
                df = pd.DataFrame(X, columns=self.feature_names)
            else:
                df = X.copy()
                
            # Handle NeuralForecast specific requirements
            if 'NeuralForecast' in str(type(self.model)):
                df['ds'] = pd.date_range(start='2024-01-01', periods=len(df), freq='D')
                df['unique_id'] = 0  # Match the training unique_id
                df['y'] = np.nan  # NaN for prediction
                df = df[['unique_id', 'ds', 'y'] + self.feature_names]
                
                # Make predictions
                preds = self.model.predict(df)
                
                # Extract predictions based on model family
                if self.model_family == 'CNNLSTM':
                    pred_col = 'BiTCN' # Common default
                elif self.model_family == 'Ensemble':
                    pred_col = 'ensemble'
                elif self.model_family in preds.columns:
                    pred_col = self.model_family
                else:
                    # Fallback to last column
                    pred_col = preds.columns[-1]
                    
                return preds[pred_col].values.reshape(-1, 1)
            
            # Generic predict
            if hasattr(self.model, 'predict'):
                preds = self.model.predict(df)
                if len(preds.shape) > 1:
                    return preds.flatten()
                return preds
            
            raise ValueError("Model does not have a predict method")

        # Use KernelExplainer
        background_data = sample_data.values
        # Limit background data for speed
        bg_summary = shap.kmeans(background_data, 10) if len(background_data) > 10 else background_data
        
        explainer = shap.KernelExplainer(predict_fn, bg_summary)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(sample_data.values)
        
        # Handle list return (for some models)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
            
        return shap_values

    def _generate_summary(self, shap_values: np.ndarray, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics from SHAP values.
        """
        # Calculate mean absolute SHAP values for feature importance
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_shap
        }).sort_values('importance', ascending=False)
        
        # Prepare data for summary plot
        summary_plot_data = {
            'features': self.feature_names,
            'shap_values': shap_values.tolist(),
            'feature_values': data.values.tolist()
        }
        
        return {
            'feature_importance': feature_importance,
            'summary_plot_data': summary_plot_data
        }
