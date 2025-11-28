import shap
import pandas as pd
import numpy as np
from typing import Dict, Any
from neuralforecast import NeuralForecast

class ExplainabilityAgent:
    """
    Agent for explaining model predictions using SHAP.
    """
    def __init__(self, model: NeuralForecast, model_family: str, feature_names: list):
        self.model = model
        self.model_family = model_family
        self.feature_names = feature_names

    def explain(self, data: pd.DataFrame, sample_size: int = 100) -> Dict[str, Any]:
        """
        Generates SHAP values to explain model predictions.
        """
        # Sample data for explanation
        sample_data = data.head(sample_size).copy()
        
        # Check if sklearn model
        from sklearn.linear_model import LinearRegression
        if isinstance(self.model, LinearRegression):
            # For sklearn models, use LinearExplainer
            background_data = sample_data[self.feature_names].values
            explainer = shap.LinearExplainer(self.model, background_data)
            shap_values = explainer.shap_values(sample_data[self.feature_names].values)
        else:
            # For NeuralForecast models, use KernelExplainer
            def predict_fn(X):
                # Convert to NeuralForecast format
                df = pd.DataFrame(X, columns=self.feature_names)
                df['ds'] = pd.date_range(start='2024-01-01', periods=len(df), freq='D')
                df['unique_id'] = 0  # Match the training unique_id
                df['y'] = np.nan  # NaN for prediction
                df = df[['unique_id', 'ds', 'y'] + self.feature_names]
                
                # Make predictions
                preds = self.model.predict(df)
                
                # Extract predictions based on model family
                if self.model_family == 'CNNLSTM':
                    pred_col = 'BiTCN'
                elif self.model_family == 'Ensemble':
                    pred_col = 'ensemble'
                else:
                    pred_col = self.model_family
                    
                return preds[pred_col].values.reshape(-1, 1)

            # Use KernelExplainer for NeuralForecast models
            background_data = sample_data.values
            explainer = shap.KernelExplainer(predict_fn, background_data)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(sample_data.values)
        
        # Generate summary statistics
        summary = self._generate_summary(shap_values, sample_data)
        
        return {
            'shap_values': shap_values,
            'feature_importance': summary['feature_importance'],
            'summary_plot_data': summary['summary_plot_data'],
            'model_family': self.model_family,
            'sample_size': sample_size
        }

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
