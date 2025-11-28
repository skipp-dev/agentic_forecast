"""
Automated Feature Engineering Service

Advanced feature engineering and selection for ML models.
Provides automated feature creation, selection, and transformation.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import json
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import shap
from boruta import BorutaPy
import featuretools as ft

# Add paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)

class AutomatedFeatureEngineeringService:
    """
    Automated feature engineering service.

    Provides:
    - Feature creation and transformation
    - Feature selection methods
    - Dimensionality reduction
    - Feature importance analysis
    - Automated feature synthesis
    """

    def __init__(self):
        """
        Initialize automated feature engineering service.
        """
        self.feature_config = {
            'max_features': 100,        # Maximum number of features to select
            'variance_threshold': 0.01, # Variance threshold for feature selection
            'correlation_threshold': 0.95, # Correlation threshold for multicollinearity
            'polynomial_degree': 2,    # Degree for polynomial features
            'pca_components': 0.95,    # PCA explained variance ratio
            'boruta_runs': 100         # Number of Boruta runs
        }

        logger.info("Automated Feature Engineering Service initialized")

    def create_polynomial_features(self, X: pd.DataFrame, degree: int = None) -> pd.DataFrame:
        """
        Create polynomial features.

        Args:
            X: Input features
            degree: Polynomial degree

        Returns:
            DataFrame with polynomial features
        """
        if degree is None:
            degree = self.feature_config['polynomial_degree']

        numeric_cols = X.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return X.copy()

        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features = poly.fit_transform(X[numeric_cols])

        # Get feature names
        feature_names = poly.get_feature_names_out(numeric_cols)

        # Create DataFrame
        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=X.index)

        # Combine with original features
        result = pd.concat([X, poly_df], axis=1)

        # Remove duplicate columns (original features)
        result = result.loc[:, ~result.columns.duplicated()]

        return result

    def create_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between numeric columns.

        Args:
            X: Input features

        Returns:
            DataFrame with interaction features
        """
        X_result = X.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        # Create pairwise interactions
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                # Product
                X_result[f'{col1}_{col2}_product'] = X[col1] * X[col2]

                # Ratio (avoid division by zero)
                X_result[f'{col1}_{col2}_ratio'] = X[col1] / (X[col2] + 1e-8)

                # Sum and difference
                X_result[f'{col1}_{col2}_sum'] = X[col1] + X[col2]
                X_result[f'{col1}_{col2}_diff'] = X[col1] - X[col2]

        return X_result

    def create_time_series_features(self, X: pd.DataFrame, time_col: str = None) -> pd.DataFrame:
        """
        Create time series features.

        Args:
            X: Input features with datetime index or time column
            time_col: Name of time column (if not using index)

        Returns:
            DataFrame with time series features
        """
        X_result = X.copy()

        # Handle time column
        if time_col and time_col in X.columns:
            time_series = pd.to_datetime(X[time_col])
        elif isinstance(X.index, pd.DatetimeIndex):
            time_series = X.index
        else:
            logger.warning("No datetime column or index found for time series features")
            return X_result

        # Time-based features
        X_result['hour'] = time_series.hour
        X_result['day_of_week'] = time_series.dayofweek
        X_result['day_of_month'] = time_series.day
        X_result['month'] = time_series.month
        X_result['quarter'] = time_series.quarter
        X_result['year'] = time_series.year
        X_result['is_weekend'] = time_series.dayofweek.isin([5, 6]).astype(int)

        # Cyclic encoding for time features
        X_result['hour_sin'] = np.sin(2 * np.pi * time_series.hour / 24)
        X_result['hour_cos'] = np.cos(2 * np.pi * time_series.hour / 24)
        X_result['month_sin'] = np.sin(2 * np.pi * time_series.month / 12)
        X_result['month_cos'] = np.cos(2 * np.pi * time_series.month / 12)

        return X_result

    def create_rolling_statistics(self, X: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
        """
        Create rolling statistics features.

        Args:
            X: Input features
            windows: List of window sizes

        Returns:
            DataFrame with rolling statistics
        """
        if windows is None:
            windows = [3, 7, 14, 30]

        X_result = X.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            for window in windows:
                # Rolling statistics
                X_result[f'{col}_rolling_mean_{window}'] = X[col].rolling(window=window, min_periods=1).mean()
                X_result[f'{col}_rolling_std_{window}'] = X[col].rolling(window=window, min_periods=1).std()
                X_result[f'{col}_rolling_min_{window}'] = X[col].rolling(window=window, min_periods=1).min()
                X_result[f'{col}_rolling_max_{window}'] = X[col].rolling(window=window, min_periods=1).max()

                # Exponential moving averages
                X_result[f'{col}_ema_{window}'] = X[col].ewm(span=window, min_periods=1).mean()

        return X_result

    def select_features_univariate(self, X: pd.DataFrame, y: pd.Series,
                                 k: int = None, method: str = 'f_regression') -> Dict[str, Any]:
        """
        Univariate feature selection.

        Args:
            X: Feature matrix
            y: Target vector
            k: Number of features to select
            method: Selection method ('f_regression', 'mutual_info')

        Returns:
            Dictionary with selected features and scores
        """
        if k is None:
            k = min(self.feature_config['max_features'], X.shape[1])

        if method == 'f_regression':
            selector = SelectKBest(score_func=f_regression, k=k)
        elif method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=k)
        else:
            raise ValueError(f"Unsupported method: {method}")

        X_selected = selector.fit_transform(X, y)

        # Get selected feature names and scores
        selected_indices = selector.get_support(indices=True)
        selected_features = X.columns[selected_indices].tolist()
        feature_scores = selector.scores_[selected_indices]

        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': selected_features,
            'score': feature_scores
        }).sort_values('score', ascending=False)

        return {
            'selected_features': selected_features,
            'feature_scores': importance_df,
            'X_selected': pd.DataFrame(X_selected, columns=selected_features, index=X.index),
            'selector': selector
        }

    def select_features_rfe(self, X: pd.DataFrame, y: pd.Series,
                          estimator: Any = None, n_features: int = None) -> Dict[str, Any]:
        """
        Recursive Feature Elimination (RFE).

        Args:
            X: Feature matrix
            y: Target vector
            estimator: Base estimator for RFE
            n_features: Number of features to select

        Returns:
            Dictionary with RFE results
        """
        if estimator is None:
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)

        if n_features is None:
            n_features = min(self.feature_config['max_features'], X.shape[1] // 2)

        # Use RFECV for automatic feature selection
        selector = RFECV(estimator, step=1, cv=3, min_features_to_select=n_features)
        selector.fit(X, y)

        selected_features = X.columns[selector.support_].tolist()

        return {
            'selected_features': selected_features,
            'feature_ranking': selector.ranking_,
            'optimal_features': selector.n_features_,
            'cv_scores': selector.cv_results_['mean_test_score'],
            'X_selected': X[selected_features],
            'selector': selector
        }

    def select_features_boruta(self, X: pd.DataFrame, y: pd.Series,
                             estimator: Any = None) -> Dict[str, Any]:
        """
        Boruta feature selection.

        Args:
            X: Feature matrix
            y: Target vector
            estimator: Base estimator

        Returns:
            Dictionary with Boruta results
        """
        if estimator is None:
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)

        # Initialize Boruta
        boruta = BorutaPy(
            estimator=estimator,
            n_estimators='auto',
            max_iter=self.feature_config['boruta_runs'],
            random_state=42
        )

        # Fit Boruta
        boruta.fit(np.array(X), np.array(y))

        # Get results
        selected_features = X.columns[boruta.support_].tolist()
        tentative_features = X.columns[boruta.support_weak_].tolist()

        return {
            'selected_features': selected_features,
            'tentative_features': tentative_features,
            'rejected_features': X.columns[~boruta.support_ & ~boruta.support_weak_].tolist(),
            'X_selected': X[selected_features],
            'boruta_selector': boruta
        }

    def select_features_shap(self, X: pd.DataFrame, y: pd.Series,
                           model: Any = None, max_evals: int = 100) -> Dict[str, Any]:
        """
        SHAP-based feature selection.

        Args:
            X: Feature matrix
            y: Target vector
            model: Model for SHAP analysis
            max_evals: Maximum SHAP evaluations

        Returns:
            Dictionary with SHAP feature importance
        """
        if model is None:
            model = xgb.XGBRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)

        # Calculate SHAP values
        explainer = shap.Explainer(model)
        shap_values = explainer(X)

        # Calculate mean absolute SHAP values
        feature_importance = np.abs(shap_values.values).mean(axis=0)

        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'shap_importance': feature_importance
        }).sort_values('shap_importance', ascending=False)

        # Select top features
        n_select = min(self.feature_config['max_features'], len(X.columns))
        selected_features = importance_df.head(n_select)['feature'].tolist()

        return {
            'selected_features': selected_features,
            'feature_importance': importance_df,
            'shap_values': shap_values,
            'X_selected': X[selected_features],
            'model': model
        }

    def reduce_dimensionality_pca(self, X: pd.DataFrame, n_components: Union[int, float] = None) -> Dict[str, Any]:
        """
        Dimensionality reduction using PCA.

        Args:
            X: Feature matrix
            n_components: Number of components or explained variance ratio

        Returns:
            Dictionary with PCA results
        """
        if n_components is None:
            n_components = self.feature_config['pca_components']

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        # Create component names
        n_pcs = pca.components_.shape[0]
        component_names = [f'PC{i+1}' for i in range(n_pcs)]

        # Explained variance
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)

        return {
            'X_reduced': pd.DataFrame(X_pca, columns=component_names, index=X.index),
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance': cumulative_variance,
            'pca_components': pca.components_,
            'scaler': scaler,
            'pca_model': pca
        }

    def reduce_dimensionality_tsne(self, X: pd.DataFrame, n_components: int = 2,
                                 perplexity: float = 30.0) -> Dict[str, Any]:
        """
        Dimensionality reduction using t-SNE.

        Args:
            X: Feature matrix
            n_components: Number of components
            perplexity: t-SNE perplexity

        Returns:
            Dictionary with t-SNE results
        """
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply t-SNE
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        X_tsne = tsne.fit_transform(X_scaled)

        component_names = [f'tSNE{i+1}' for i in range(n_components)]

        return {
            'X_reduced': pd.DataFrame(X_tsne, columns=component_names, index=X.index),
            'scaler': scaler,
            'tsne_model': tsne
        }

    def remove_multicollinear_features(self, X: pd.DataFrame,
                                     threshold: float = None) -> Dict[str, Any]:
        """
        Remove multicollinear features based on correlation.

        Args:
            X: Feature matrix
            threshold: Correlation threshold

        Returns:
            Dictionary with selected features
        """
        if threshold is None:
            threshold = self.feature_config['correlation_threshold']

        # Calculate correlation matrix
        corr_matrix = X.corr().abs()

        # Find highly correlated features
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = []

        for column in upper.columns:
            if column in to_drop:
                continue
            correlated_features = upper[column][upper[column] > threshold].index.tolist()
            if correlated_features:
                # Keep the first feature, drop others
                to_drop.extend(correlated_features)

        # Remove duplicates
        to_drop = list(set(to_drop))
        selected_features = [col for col in X.columns if col not in to_drop]

        return {
            'selected_features': selected_features,
            'dropped_features': to_drop,
            'correlation_matrix': corr_matrix,
            'X_selected': X[selected_features]
        }

    def automated_feature_pipeline(self, X: pd.DataFrame, y: pd.Series = None,
                                 include_time_features: bool = True,
                                 include_polynomial: bool = True,
                                 selection_method: str = 'shap') -> Dict[str, Any]:
        """
        Complete automated feature engineering pipeline.

        Args:
            X: Input features
            y: Target variable (required for supervised methods)
            include_time_features: Whether to create time features
            include_polynomial: Whether to create polynomial features
            selection_method: Feature selection method

        Returns:
            Dictionary with engineered features
        """
        logger.info("Starting automated feature engineering pipeline")

        X_engineered = X.copy()

        # 1. Create time series features
        if include_time_features:
            X_engineered = self.create_time_series_features(X_engineered)

        # 2. Create polynomial features
        if include_polynomial:
            X_engineered = self.create_polynomial_features(X_engineered)

        # 3. Create interaction features
        X_engineered = self.create_interaction_features(X_engineered)

        # 4. Create rolling statistics
        X_engineered = self.create_rolling_statistics(X_engineered)

        # 5. Remove multicollinear features
        multicollinear_result = self.remove_multicollinear_features(X_engineered)
        X_engineered = multicollinear_result['X_selected']

        # 6. Feature selection (if target provided)
        selection_result = None
        if y is not None:
            if selection_method == 'shap':
                selection_result = self.select_features_shap(X_engineered, y)
            elif selection_method == 'rfe':
                selection_result = self.select_features_rfe(X_engineered, y)
            elif selection_method == 'boruta':
                selection_result = self.select_features_boruta(X_engineered, y)
            elif selection_method == 'univariate':
                selection_result = self.select_features_univariate(X_engineered, y)

            if selection_result:
                X_final = selection_result['X_selected']
            else:
                X_final = X_engineered
        else:
            X_final = X_engineered

        # 7. Dimensionality reduction (optional)
        pca_result = None
        if X_final.shape[1] > self.feature_config['max_features']:
            pca_result = self.reduce_dimensionality_pca(X_final)
            X_final = pca_result['X_reduced']

        return {
            'X_engineered': X_engineered,
            'X_final': X_final,
            'multicollinear_removal': multicollinear_result,
            'feature_selection': selection_result,
            'dimensionality_reduction': pca_result,
            'pipeline_steps': [
                'time_features', 'polynomial_features', 'interaction_features',
                'rolling_statistics', 'multicollinear_removal', 'feature_selection',
                'dimensionality_reduction'
            ]
        }