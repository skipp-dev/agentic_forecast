"""
FX Data Agent

Handles collection and processing of foreign exchange data from Frankfurter.
Provides EUR↔USD, EUR↔GBP, EUR↔JPY rates for cross-asset analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Optional, List, Dict, Union

from ..frankfurter_client import FrankfurterClient

logger = logging.getLogger(__name__)

class FXDataAgent:
    """
    Agent for collecting FX data from Frankfurter.

    Features:
    - EUR-based exchange rates (USD, GBP, JPY)
    - Historical time series
    - Cross-asset FX feature calculation
    - Stress detection
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the FX Data Agent.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.client = FrankfurterClient()

        # Data storage paths
        self.raw_data_path = Path('data/raw/fx/frankfurter')
        self.processed_data_path = Path('data/processed/fx')
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)

        # Target currency pairs (EUR as base)
        self.target_pairs = ['EURUSD', 'EURGBP', 'EURJPY']
        self.target_currencies = ['USD', 'GBP', 'JPY']

        logger.info("Initialized FX Data Agent")

    def fetch_fx_data(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch FX time series data.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with FX rates
        """
        try:
            logger.info(f"Fetching FX data from {start_date} to {end_date}")

            raw_data = self.client.get_time_series(
                start_date=start_date,
                end_date=end_date,
                base='EUR',
                symbols=self.target_currencies
            )

            if raw_data:
                df = self.client.process_time_series_data(raw_data)
                if not df.empty:
                    logger.info(f"Successfully fetched {len(df)} FX rate observations")
                    return df

        except Exception as e:
            logger.error(f"Failed to fetch FX data: {e}")

        return None

    def save_fx_data(self, df: pd.DataFrame):
        """
        Save FX data to parquet file.

        Args:
            df: DataFrame to save
        """
        if df is None or df.empty:
            return

        filename = "fx_rates.parquet"
        filepath = self.raw_data_path / filename

        try:
            df.to_parquet(filepath, index=False)
            logger.info(f"Saved FX data to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save FX data: {e}")

    def load_fx_data(self) -> Optional[pd.DataFrame]:
        """
        Load FX data from parquet file.

        Returns:
            DataFrame or None if file doesn't exist
        """
        filename = "fx_rates.parquet"
        filepath = self.raw_data_path / filename

        if not filepath.exists():
            return None

        try:
            df = pd.read_parquet(filepath)
            logger.info(f"Loaded FX data from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Failed to load FX data: {e}")
            return None

    def update_fx_data(self, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Update FX data.

        Args:
            force_refresh: If True, ignore cache

        Returns:
            DataFrame with FX data
        """
        # Check cache first
        if not force_refresh:
            cached_data = self.load_fx_data()
            if cached_data is not None:
                # Check if data is recent enough (within last day)
                latest_date = cached_data['date'].max()
                days_old = (datetime.now() - latest_date).days

                if days_old <= 1:  # Data is current
                    logger.info(f"Using cached FX data (last updated: {latest_date.date()})")
                    return cached_data

        # Fetch fresh data (last 2 years)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)

        df = self.fetch_fx_data(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )

        if df is not None:
            self.save_fx_data(df)
            return df

        return None

    def calculate_fx_features(self, fx_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate FX-specific features for cross-asset analysis.

        Args:
            fx_data: Raw FX DataFrame

        Returns:
            DataFrame with FX features (daily)
        """
        if fx_data is None or fx_data.empty:
            return pd.DataFrame()

        # Pivot to wide format for easier calculations
        fx_pivot = fx_data.pivot(
            index='date',
            columns='pair',
            values='rate'
        ).sort_index()

        features = pd.DataFrame(index=fx_pivot.index)

        # Calculate returns for each pair
        for pair in self.target_pairs:
            if pair in fx_pivot.columns:
                features[f'{pair.lower()}_ret_1d'] = fx_pivot[pair].pct_change()
                features[f'{pair.lower()}_ret_3d'] = fx_pivot[pair].pct_change(3)
                features[f'{pair.lower()}_ret_5d'] = fx_pivot[pair].pct_change(5)

                # Volatility
                features[f'{pair.lower()}_vol_20d'] = features[f'{pair.lower()}_ret_1d'].rolling(20).std()

        # Focus on EURUSD for main features (as mentioned in requirements)
        if 'EURUSD' in fx_pivot.columns:
            eurusd_rate = fx_pivot['EURUSD']

            # EURUSD specific features
            features['eurusd_rate'] = eurusd_rate
            features['eurusd_ret_1d'] = eurusd_rate.pct_change()
            features['eurusd_ret_5d'] = eurusd_rate.pct_change(5)
            features['eurusd_vol_20d'] = features['eurusd_ret_1d'].rolling(20).std()

            # FX stress flag (significant moves)
            features['fx_stress_flag'] = (abs(features['eurusd_ret_1d']) > 0.005).astype(int)  # 0.5% move

            # FX volatility spike detection
            vol_ma = features['eurusd_vol_20d'].rolling(50).mean()
            features['fx_vol_spike_flag'] = (features['eurusd_vol_20d'] > vol_ma * 1.5).astype(int)

        return features

    def get_fx_regime_signals(self, fx_features: pd.DataFrame) -> pd.DataFrame:
        """
        Generate FX regime signals from features.

        Args:
            fx_features: DataFrame with FX features

        Returns:
            DataFrame with regime classifications
        """
        regimes = pd.DataFrame(index=fx_features.index)

        # FX regime based on volatility and stress
        vol = fx_features.get('eurusd_vol_20d', 0)
        stress_flag = fx_features.get('fx_stress_flag', 0)
        vol_spike = fx_features.get('fx_vol_spike_flag', 0)

        # Simple regime logic
        conditions = [
            (stress_flag == 1) | (vol_spike == 1),  # Stress regime
            (vol > vol.quantile(0.8)),  # High vol regime
        ]
        choices = ['stress', 'high_vol']
        default = 'normal'

        regimes['fx_regime'] = np.select(conditions, choices, default=default)

        return regimes