"""
Model Interpretability Service

Advanced model interpretability for IB Forecast system.
Provides SHAP values, feature importance, partial dependence plots, and model explanations.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
from sklearn.inspection import permutation_importance
import shap
import lime
import lime.lime_tabular
from pdpbox import pdp
import eli5
from eli5.sklearn import PermutationImportance
import dalex as dx

# Add paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)

class ModelInterpretabilityService:
    """
    Model interpretability service for explaining ML model predictions.

    Provides:
    - SHAP (SHapley Additive exPlanations) values
    - Feature importance analysis
    - Partial dependence plots
    - LIME (Local Interpretable Model-agnostic Explanations)
    - Permutation importance
    - Model performance diagnostics
    """

    def __init__(self):
        """
        Initialize model interpretability service.
        """
        self.interpretability_config = {
            'shap_max_evals': 1000,
            'lime_samples': 1000,
            'pdp_grid_resolution': 20,
            'permutation_repeats': 10
        }

        # Initialize SHAP and LIME explainers cache
        self.shap_explainers = {}
        self.lime_explainers = {}

        logger.info("Model Interpretability Service initialized")

    def explain_prediction(self, model, X: pd.DataFrame, y: Optional[pd.Series] = None,
                          method: str = 'shap', instance_idx: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate explanation for a model prediction.

        Args:
            model: Trained ML model
            X: Feature matrix
            y: Target vector (optional)
            method: Explanation method ('shap', 'lime', 'permutation')
            instance_idx: Index of instance to explain (for local explanations)

        Returns:
            Dictionary with explanation results
        """
        if method == 'shap':
            return self._explain_with_shap(model, X, instance_idx)
        elif method == 'lime':
            return self._explain_with_lime(model, X, instance_idx)
        elif method == 'permutation':
            return self._explain_with_permutation(model, X, y)
        else:
            raise ValueError(f"Unknown explanation method: {method}")

    def analyze_feature_importance(self, model, X: pd.DataFrame, y: Optional[pd.Series] = None,
                                 method: str = 'shap') -> Dict[str, Any]:
        """
        Analyze global feature importance.

        Args:
            model: Trained ML model
            X: Feature matrix
            y: Target vector (optional)
            method: Importance method ('shap', 'permutation', 'built_in')

        Returns:
            Dictionary with feature importance results
        """
        if method == 'shap':
            return self._shap_feature_importance(model, X)
        elif method == 'permutation':
            return self._permutation_importance(model, X, y)
        elif method == 'built_in':
            return self._built_in_importance(model, X)
        else:
            raise ValueError(f"Unknown importance method: {method}")

    def generate_partial_dependence(self, model, X: pd.DataFrame,
                                  features: List[Union[str, int]],
                                  kind: str = 'average') -> Dict[str, Any]:
        """
        Generate partial dependence plots.

        Args:
            model: Trained ML model
            X: Feature matrix
            features: Features to analyze
            kind: Type of PD plot ('average', 'individual', 'both')

        Returns:
            Dictionary with partial dependence results
        """
        try:
            # Calculate partial dependence
            pd_results = partial_dependence(
                model, X, features,
                kind=kind,
                grid_resolution=self.interpretability_config['pdp_grid_resolution']
            )

            results = {
                'feature_names': [X.columns[f] if isinstance(f, int) else f for f in features],
                'partial_dependence': pd_results['average'].tolist() if 'average' in pd_results else None,
                'individual': pd_results.get('individual', []),
                'grid_values': pd_results['grid_values']
            }

            return results

        except Exception as e:
            logger.error(f"Error generating partial dependence: {e}")
            return {'error': str(e)}

    def create_model_profile(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Create comprehensive model profile using DALEX.

        Args:
            model: Trained ML model
            X: Feature matrix
            y: Target vector

        Returns:
            Dictionary with model profile
        """
        try:
            # Create DALEX explainer
            explainer = dx.Explainer(model, X, y, verbose=False)

            # Generate model performance
            model_performance = explainer.model_performance()

            # Generate feature importance
            feature_importance = explainer.model_parts()

            # Generate partial dependence for top features
            top_features = feature_importance.result.variable[:5].tolist()
            pd_profiles = {}
            for feature in top_features:
                try:
                    pd_profile = explainer.model_profile(variables=[feature])
                    pd_profiles[feature] = {
                        'cp': pd_profile.result._cp_profile.tolist(),
                        'variable': pd_profile.result.variable_name
                    }
                except:
                    continue

            return {
                'model_performance': {
                    'mse': model_performance.result.mse.values[0],
                    'rmse': model_performance.result.rmse.values[0],
                    'r2': model_performance.result.r2.values[0],
                    'mae': model_performance.result.mae.values[0]
                },
                'feature_importance': {
                    'variables': feature_importance.result.variable.tolist(),
                    'dropout_loss': feature_importance.result.dropout_loss.tolist(),
                    'label': feature_importance.result.label.tolist()
                },
                'partial_dependence': pd_profiles
            }

        except Exception as e:
            logger.error(f"Error creating model profile: {e}")
            return {'error': str(e)}

    def detect_model_drift(self, model, X_reference: pd.DataFrame, X_current: pd.DataFrame,
                          y_reference: Optional[pd.Series] = None,
                          y_current: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Detect model drift by comparing reference and current data.

        Args:
            model: Trained ML model
            X_reference: Reference feature data
            X_current: Current feature data
            y_reference: Reference target data
            y_current: Current target data

        Returns:
            Dictionary with drift detection results
        """
        drift_results = {}

        # Feature distribution drift
        drift_results['feature_drift'] = self._detect_feature_drift(X_reference, X_current)

        # Prediction drift
        if y_reference is not None and y_current is not None:
            predictions_ref = model.predict(X_reference)
            predictions_cur = model.predict(X_current)

            drift_results['prediction_drift'] = self._detect_prediction_drift(
                predictions_ref, predictions_cur, y_reference, y_current
            )

        # Performance drift
        if y_reference is not None and y_current is not None:
            from sklearn.metrics import mean_squared_error, r2_score

            mse_ref = mean_squared_error(y_reference, predictions_ref)
            mse_cur = mean_squared_error(y_current, predictions_cur)
            r2_ref = r2_score(y_reference, predictions_ref)
            r2_cur = r2_score(y_current, predictions_cur)

            drift_results['performance_drift'] = {
                'mse_change': mse_cur - mse_ref,
                'r2_change': r2_cur - r2_ref,
                'mse_reference': mse_ref,
                'mse_current': mse_cur,
                'r2_reference': r2_ref,
                'r2_current': r2_cur
            }

        return drift_results

    def _explain_with_shap(self, model, X: pd.DataFrame, instance_idx: Optional[int] = None) -> Dict[str, Any]:
        """Generate SHAP explanations."""
        try:
            # Determine explainer type based on model
            if hasattr(model, 'predict_proba'):
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.Explainer(model)

            # Cache explainer
            model_id = id(model)
            if model_id not in self.shap_explainers:
                self.shap_explainers[model_id] = explainer

            explainer = self.shap_explainers[model_id]

            if instance_idx is not None:
                # Local explanation
                instance = X.iloc[instance_idx:instance_idx+1]
                shap_values = explainer(instance)

                return {
                    'method': 'shap_local',
                    'instance_idx': instance_idx,
                    'shap_values': shap_values.values.tolist(),
                    'base_value': float(shap_values.base_values[0]),
                    'feature_names': X.columns.tolist(),
                    'feature_values': instance.values.tolist()[0]
                }
            else:
                # Global explanation
                shap_values = explainer(X)

                # Calculate mean absolute SHAP values for feature importance
                feature_importance = np.abs(shap_values.values).mean(axis=0)

                return {
                    'method': 'shap_global',
                    'shap_values': shap_values.values.tolist(),
                    'base_values': shap_values.base_values.tolist(),
                    'feature_names': X.columns.tolist(),
                    'feature_importance': feature_importance.tolist()
                }

        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {e}")
            return {'error': str(e)}

    def _explain_with_lime(self, model, X: pd.DataFrame, instance_idx: int) -> Dict[str, Any]:
        """Generate LIME explanations."""
        try:
            # Create LIME explainer
            model_id = id(model)
            if model_id not in self.lime_explainers:
                self.lime_explainers[model_id] = lime.lime_tabular.LimeTabularExplainer(
                    X.values,
                    feature_names=X.columns.tolist(),
                    class_names=['prediction'],
                    mode='regression'
                )

            explainer = self.lime_explainers[model_id]

            # Explain instance
            instance = X.iloc[instance_idx]
            explanation = explainer.explain_instance(
                instance.values,
                model.predict,
                num_features=len(X.columns),
                num_samples=self.interpretability_config['lime_samples']
            )

            # Extract feature contributions
            feature_contributions = {}
            for feature, weight in explanation.as_list():
                feature_contributions[feature] = weight

            return {
                'method': 'lime',
                'instance_idx': instance_idx,
                'feature_contributions': feature_contributions,
                'prediction': explanation.predicted_value,
                'intercept': explanation.intercept[0]
            }

        except Exception as e:
            logger.error(f"Error generating LIME explanation: {e}")
            return {'error': str(e)}

    def _explain_with_permutation(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Generate permutation importance explanations."""
        try:
            # Calculate permutation importance
            perm_importance = permutation_importance(
                model, X, y,
                n_repeats=self.interpretability_config['permutation_repeats'],
                random_state=42
            )

            return {
                'method': 'permutation',
                'feature_names': X.columns.tolist(),
                'importances': perm_importance.importances.tolist(),
                'importances_mean': perm_importance.importances_mean.tolist(),
                'importances_std': perm_importance.importances_std.tolist()
            }

        except Exception as e:
            logger.error(f"Error generating permutation explanation: {e}")
            return {'error': str(e)}

    def _shap_feature_importance(self, model, X: pd.DataFrame) -> Dict[str, Any]:
        """Calculate SHAP-based feature importance."""
        shap_result = self._explain_with_shap(model, X)

        if 'error' in shap_result:
            return shap_result

        return {
            'method': 'shap_importance',
            'feature_names': shap_result['feature_names'],
            'importance_values': shap_result['feature_importance'],
            'importance_dict': dict(zip(shap_result['feature_names'], shap_result['feature_importance']))
        }

    def _permutation_importance(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Calculate permutation feature importance."""
        perm_result = self._explain_with_permutation(model, X, y)

        if 'error' in perm_result:
            return perm_result

        return {
            'method': 'permutation_importance',
            'feature_names': perm_result['feature_names'],
            'importance_values': perm_result['importances_mean'],
            'importance_std': perm_result['importances_std'],
            'importance_dict': dict(zip(perm_result['feature_names'], perm_result['importances_mean']))
        }

    def _built_in_importance(self, model, X: pd.DataFrame) -> Dict[str, Any]:
        """Extract built-in feature importance from model."""
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_)
            else:
                return {'error': 'Model does not have built-in feature importance'}

            return {
                'method': 'built_in_importance',
                'feature_names': X.columns.tolist(),
                'importance_values': importance.tolist(),
                'importance_dict': dict(zip(X.columns, importance))
            }

        except Exception as e:
            return {'error': str(e)}

    def _detect_feature_drift(self, X_ref: pd.DataFrame, X_cur: pd.DataFrame) -> Dict[str, Any]:
        """Detect feature distribution drift."""
        drift_scores = {}

        for column in X_ref.columns:
            if column in X_cur.columns:
                # Kolmogorov-Smirnov test for distribution difference
                from scipy.stats import ks_2samp

                ref_values = X_ref[column].dropna()
                cur_values = X_cur[column].dropna()

                if len(ref_values) > 10 and len(cur_values) > 10:
                    ks_stat, p_value = ks_2samp(ref_values, cur_values)

                    # Calculate distribution statistics
                    ref_mean, ref_std = ref_values.mean(), ref_values.std()
                    cur_mean, cur_std = cur_values.mean(), cur_values.std()

                    drift_scores[column] = {
                        'ks_statistic': ks_stat,
                        'p_value': p_value,
                        'drift_detected': p_value < 0.05,
                        'mean_change': cur_mean - ref_mean,
                        'std_change': cur_std - ref_std,
                        'ref_mean': ref_mean,
                        'cur_mean': cur_mean
                    }

        return drift_scores

    def _detect_prediction_drift(self, pred_ref: np.ndarray, pred_cur: np.ndarray,
                               y_ref: pd.Series, y_cur: pd.Series) -> Dict[str, Any]:
        """Detect prediction distribution drift."""
        from scipy.stats import ks_2samp

        # KS test on predictions
        ks_stat, p_value = ks_2samp(pred_ref, pred_cur)

        # Calculate prediction statistics
        pred_ref_mean, pred_ref_std = np.mean(pred_ref), np.std(pred_ref)
        pred_cur_mean, pred_cur_std = np.mean(pred_cur), np.std(pred_cur)

        return {
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'drift_detected': p_value < 0.05,
            'mean_change': pred_cur_mean - pred_ref_mean,
            'std_change': pred_cur_std - pred_ref_std,
            'ref_mean': pred_ref_mean,
            'cur_mean': pred_cur_mean
        }

    def generate_interpretability_report(self, model, X: pd.DataFrame, y: pd.Series,
                                       sample_size: int = 100) -> Dict[str, Any]:
        """
        Generate comprehensive interpretability report.

        Args:
            model: Trained ML model
            X: Feature matrix
            y: Target vector
            sample_size: Number of samples for analysis

        Returns:
            Comprehensive interpretability report
        """
        # Sample data for efficiency
        if len(X) > sample_size:
            sample_indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X.iloc[sample_indices]
            y_sample = y.iloc[sample_indices]
        else:
            X_sample = X
            y_sample = y

        report = {
            'timestamp': datetime.now().isoformat(),
            'model_type': type(model).__name__,
            'dataset_info': {
                'total_samples': len(X),
                'sample_size': len(X_sample),
                'n_features': X.shape[1]
            }
        }

        # Feature importance analysis
        report['feature_importance'] = {
            'shap': self.analyze_feature_importance(model, X_sample, method='shap'),
            'permutation': self.analyze_feature_importance(model, X_sample, y_sample, method='permutation'),
            'built_in': self.analyze_feature_importance(model, X_sample, method='built_in')
        }

        # Model profile
        report['model_profile'] = self.create_model_profile(model, X_sample, y_sample)

        # Partial dependence for top features
        if 'shap' in report['feature_importance'] and 'importance_dict' in report['feature_importance']['shap']:
            top_features = sorted(
                report['feature_importance']['shap']['importance_dict'].items(),
                key=lambda x: x[1], reverse=True
            )[:3]

            report['partial_dependence'] = {}
            for feature_name, _ in top_features:
                feature_idx = X.columns.get_loc(feature_name)
                pd_result = self.generate_partial_dependence(model, X_sample, [feature_idx])
                report['partial_dependence'][feature_name] = pd_result

        # Sample explanations
        report['sample_explanations'] = {}
        sample_indices = np.random.choice(len(X_sample), min(5, len(X_sample)), replace=False)

        for idx in sample_indices:
            report['sample_explanations'][f'sample_{idx}'] = {
                'shap': self.explain_prediction(model, X_sample, method='shap', instance_idx=idx),
                'lime': self.explain_prediction(model, X_sample, method='lime', instance_idx=idx)
            }

        return report