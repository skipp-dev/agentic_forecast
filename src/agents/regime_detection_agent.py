"""
Regime Detection Agent

Detects market regimes based on macro economic conditions:
- Rate regimes (easing, neutral, tightening)
- Labor regimes (expansion, contraction)
- Commodity regimes (bull, bear, sideways)
- Seasonal regimes
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Add paths for imports
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)

class RegimeDetectionAgent:
    """
    Agent responsible for detecting market regimes based on macro economic data.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Regime definitions
        self.regime_definitions = {
            'rate_regime': {
                'easing': lambda x: x['fed_funds_change_3m'] < -0.25,
                'neutral': lambda x: (x['fed_funds_change_3m'] >= -0.25) & (x['fed_funds_change_3m'] <= 0.25),
                'tightening': lambda x: x['fed_funds_change_3m'] > 0.25
            },
            'labor_regime': {
                'expansion': lambda x: x['payrolls_change_3m'] > 0.005,
                'contraction': lambda x: x['payrolls_change_3m'] < -0.005,
                'stagnation': lambda x: (x['payrolls_change_3m'] >= -0.005) & (x['payrolls_change_3m'] <= 0.005)
            },
            'commodity_regime': {
                'bull': lambda x: (x['oil_returns_1m'] > 0.1) | (x['gold_returns_1m'] > 0.05),
                'bear': lambda x: (x['oil_returns_1m'] < -0.1) | (x['gold_returns_1m'] < -0.05),
                'sideways': lambda x: ((x['oil_returns_1m'] >= -0.1) & (x['oil_returns_1m'] <= 0.1)) &
                               ((x['gold_returns_1m'] >= -0.05) & (x['gold_returns_1m'] <= 0.05))
            },
            'seasonal_regime': {
                'winter': lambda x: x['month'].isin([12, 1, 2]),
                'spring': lambda x: x['month'].isin([3, 4, 5]),
                'summer': lambda x: x['month'].isin([6, 7, 8]),
                'fall': lambda x: x['month'].isin([9, 10, 11])
            }
        }

    def detect_regimes(self, macro_features: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Detect market regimes from macro features.

        Args:
            macro_features: DataFrame with macro economic features

        Returns:
            Dictionary of regime classifications
        """
        logger.info("Detecting market regimes")

        if macro_features.empty:
            logger.warning("No macro features provided for regime detection")
            return {}

        regimes = {}

        # Add month column for seasonal analysis
        macro_features = macro_features.copy()
        macro_features['month'] = macro_features.index.month

        # Detect each regime type
        for regime_type, conditions in self.regime_definitions.items():
            try:
                regime_series = self._detect_single_regime(macro_features, regime_type, conditions)
                regimes[regime_type] = regime_series
                logger.info(f"Detected {regime_type}: {regime_series.value_counts().to_dict()}")
            except Exception as e:
                logger.error(f"Failed to detect {regime_type}: {e}")

        return regimes

    def _detect_single_regime(self, data: pd.DataFrame, regime_type: str,
                            conditions: Dict[str, callable]) -> pd.Series:
        """
        Detect a single regime type based on conditions.
        """
        regime_labels = pd.Series(index=data.index, dtype='object')

        for regime_label, condition_func in conditions.items():
            try:
                mask = condition_func(data)
                regime_labels[mask] = regime_label
            except KeyError as e:
                logger.warning(f"Missing column for {regime_type} {regime_label}: {e}")
                continue

        # Fill any unmapped values with 'unknown'
        regime_labels = regime_labels.fillna('unknown')

        return regime_labels

    def detect_clustered_regimes(self, macro_features: pd.DataFrame, n_clusters: int = 4) -> pd.Series:
        """
        Use unsupervised clustering to detect market regimes.

        Args:
            macro_features: Macro features DataFrame
            n_clusters: Number of regime clusters to identify

        Returns:
            Series with cluster labels as regime indicators
        """
        logger.info(f"Detecting clustered regimes with {n_clusters} clusters")

        if macro_features.empty:
            return pd.Series()

        # Select numeric columns for clustering
        numeric_cols = macro_features.select_dtypes(include=[np.number]).columns
        cluster_data = macro_features[numeric_cols].fillna(method='ffill').fillna(0)

        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_data)

        # Convert to series with datetime index
        regime_series = pd.Series(cluster_labels, index=macro_features.index, name='clustered_regime')

        # Map cluster numbers to more meaningful labels
        cluster_mapping = {i: f'regime_{i+1}' for i in range(n_clusters)}
        regime_series = regime_series.map(cluster_mapping)

        logger.info(f"Clustered regimes: {regime_series.value_counts().to_dict()}")

        return regime_series

    def get_regime_features(self, regimes: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Convert regime classifications into numerical features for modeling.

        Args:
            regimes: Dictionary of regime classifications

        Returns:
            DataFrame with regime-based features
        """
        logger.info("Generating regime-based features")

        if not regimes:
            return pd.DataFrame()

        # Combine all regime series
        regime_df = pd.DataFrame(index=next(iter(regimes.values())).index)

        for regime_type, regime_series in regimes.items():
            # One-hot encode regime labels
            dummies = pd.get_dummies(regime_series, prefix=regime_type)
            regime_df = regime_df.join(dummies)

        # Add regime transition features
        for regime_type, regime_series in regimes.items():
            # Regime change indicator
            regime_change = (regime_series != regime_series.shift(1)).astype(int)
            regime_df[f'{regime_type}_change'] = regime_change

            # Regime duration
            regime_duration = regime_series.groupby((regime_series != regime_series.shift()).cumsum()).cumcount() + 1
            regime_df[f'{regime_type}_duration'] = regime_duration

        logger.info(f"Generated {regime_df.shape[1]} regime features")
        return regime_df

    def analyze_regime_performance(self, regimes: Dict[str, pd.Series],
                                 asset_returns: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Analyze how different assets perform under various regimes.

        Args:
            regimes: Dictionary of regime classifications
            asset_returns: DataFrame with asset returns

        Returns:
            Dictionary with regime performance analysis
        """
        logger.info("Analyzing regime performance")

        performance_analysis = {}

        for regime_type, regime_series in regimes.items():
            regime_performance = {}

            for regime_label in regime_series.unique():
                if regime_label == 'unknown':
                    continue

                # Get returns during this regime
                regime_mask = regime_series == regime_label
                regime_returns = asset_returns[regime_mask]

                if not regime_returns.empty:
                    # Calculate performance metrics
                    mean_returns = regime_returns.mean()
                    volatility = regime_returns.std()
                    sharpe_ratio = mean_returns / volatility * np.sqrt(252)  # Annualized

                    regime_performance[regime_label] = {
                        'mean_return': mean_returns,
                        'volatility': volatility,
                        'sharpe_ratio': sharpe_ratio,
                        'periods': len(regime_returns)
                    }

            performance_analysis[regime_type] = pd.DataFrame(regime_performance).T

        return performance_analysis

    def get_regime_summary(self, macro_features: pd.DataFrame,
                          asset_returns: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Main method to perform complete regime analysis.

        Args:
            macro_features: Macro economic features
            asset_returns: Optional asset returns for performance analysis

        Returns:
            Complete regime analysis results
        """
        # Detect regimes
        regimes = self.detect_regimes(macro_features)

        # Add clustered regimes
        clustered_regimes = self.detect_clustered_regimes(macro_features)
        if not clustered_regimes.empty:
            regimes['clustered_regime'] = clustered_regimes

        # Generate regime features
        regime_features = self.get_regime_features(regimes)

        # Analyze performance if asset returns provided
        performance_analysis = {}
        if asset_returns is not None:
            performance_analysis = self.analyze_regime_performance(regimes, asset_returns)

        return {
            'regimes': regimes,
            'regime_features': regime_features,
            'performance_analysis': performance_analysis,
            'detection_timestamp': datetime.now(),
            'regime_types': list(regimes.keys())
        }</content>
<parameter name="filePath">c:\Users\spreu\Documents\agentic_forecast\agents\regime_detection_agent.py