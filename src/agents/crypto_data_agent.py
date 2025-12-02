"""
Crypto Data Agent

Handles collection and processing of cryptocurrency data from CoinGecko.
Provides BTC and ETH price data for cross-asset analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Optional, List, Dict, Union

from ..coingecko_client import CoinGeckoClient

logger = logging.getLogger(__name__)

class CryptoDataAgent:
    """
    Agent for collecting cryptocurrency data from CoinGecko.

    Features:
    - BTC and ETH historical price data
    - Market cap and volume information
    - Cross-asset feature calculation
    - Crash/boom detection
    """

    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize the Crypto Data Agent.

        Args:
            api_key: CoinGecko API key (optional)
            config: Configuration dictionary
        """
        self.api_key = api_key
        self.config = config or {}
        self.client = CoinGeckoClient(api_key=api_key)

        # Data storage paths
        self.raw_data_path = Path('data/raw/crypto/coingecko')
        self.processed_data_path = Path('data/processed/crypto')
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)

        # Target cryptocurrencies
        self.target_coins = ['bitcoin', 'ethereum']
        self.coin_symbols = {'bitcoin': 'BTC', 'ethereum': 'ETH'}

        logger.info("Initialized Crypto Data Agent")

    def fetch_crypto_data(self, coin_id: str, days: Union[int, str] = 'max') -> Optional[pd.DataFrame]:
        """
        Fetch historical data for a cryptocurrency.

        Args:
            coin_id: CoinGecko coin ID ('bitcoin', 'ethereum')
            days: Number of days or 'max' for all data

        Returns:
            DataFrame with price data
        """
        try:
            logger.info(f"Fetching {coin_id} data ({days} days)")

            raw_data = self.client.get_coin_market_chart(coin_id, days=days)
            if raw_data:
                df = self.client.process_market_chart_data(raw_data)
                if not df.empty:
                    df['coin_id'] = coin_id
                    df['symbol'] = self.coin_symbols.get(coin_id, coin_id.upper())
                    logger.info(f"Successfully fetched {len(df)} days of data for {coin_id}")
                    return df

        except Exception as e:
            logger.error(f"Failed to fetch data for {coin_id}: {e}")

        return None

    def save_crypto_data(self, coin_id: str, df: pd.DataFrame):
        """
        Save crypto data to parquet file.

        Args:
            coin_id: CoinGecko coin ID
            df: DataFrame to save
        """
        if df is None or df.empty:
            return

        filename = f"{coin_id}.parquet"
        filepath = self.raw_data_path / filename

        try:
            df.to_parquet(filepath, index=True)
            logger.info(f"Saved crypto data for {coin_id} to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save crypto data for {coin_id}: {e}")

    def load_crypto_data(self, coin_id: str) -> Optional[pd.DataFrame]:
        """
        Load crypto data from parquet file.

        Args:
            coin_id: CoinGecko coin ID

        Returns:
            DataFrame or None if file doesn't exist
        """
        filename = f"{coin_id}.parquet"
        filepath = self.raw_data_path / filename

        if not filepath.exists():
            return None

        try:
            df = pd.read_parquet(filepath)
            logger.info(f"Loaded crypto data for {coin_id} from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Failed to load crypto data for {coin_id}: {e}")
            return None

    def update_crypto_data(self, coin_id: str, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Update data for a cryptocurrency.

        Args:
            coin_id: CoinGecko coin ID
            force_refresh: If True, ignore cache

        Returns:
            DataFrame with crypto data
        """
        # Check cache first
        if not force_refresh:
            cached_data = self.load_crypto_data(coin_id)
            if cached_data is not None:
                # Check if data is recent enough (within last day)
                latest_date = cached_data.index.max()
                days_old = (datetime.now() - latest_date).days

                if days_old <= 1:  # Data is current
                    logger.info(f"Using cached data for {coin_id} (last updated: {latest_date.date()})")
                    return cached_data

        # Fetch fresh data
        df = self.fetch_crypto_data(coin_id, days='max')
        if df is not None:
            self.save_crypto_data(coin_id, df)
            return df

        return None

    def update_all_cryptos(self, force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Update data for all target cryptocurrencies.

        Args:
            force_refresh: If True, ignore cache for all coins

        Returns:
            Dictionary mapping coin IDs to their DataFrames
        """
        results = {}

        for coin_id in self.target_coins:
            try:
                df = self.update_crypto_data(coin_id, force_refresh=force_refresh)
                if df is not None:
                    results[coin_id] = df
                else:
                    logger.warning(f"Failed to update data for {coin_id}")
            except Exception as e:
                logger.error(f"Error updating {coin_id}: {e}")

        logger.info(f"Successfully updated {len(results)}/{len(self.target_coins)} cryptocurrencies")
        return results

    def calculate_crypto_features(self, crypto_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate crypto-specific features for cross-asset analysis.

        Args:
            crypto_data: Dictionary with crypto DataFrames

        Returns:
            DataFrame with crypto features (daily)
        """
        if not crypto_data:
            return pd.DataFrame()

        # Start with BTC data as base
        btc_data = crypto_data.get('bitcoin')
        if btc_data is None or btc_data.empty:
            return pd.DataFrame()

        # Calculate BTC features
        btc_features = self._calculate_single_crypto_features(btc_data, 'btc')

        # Add ETH features if available
        eth_data = crypto_data.get('ethereum')
        if eth_data is not None and not eth_data.empty:
            eth_features = self._calculate_single_crypto_features(eth_data, 'eth')
            # Merge on date
            btc_features = btc_features.merge(eth_features, left_index=True, right_index=True, how='left')

        return btc_features

    def _calculate_single_crypto_features(self, df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """
        Calculate features for a single cryptocurrency.

        Args:
            df: Crypto price DataFrame
            prefix: Column prefix ('btc' or 'eth')

        Returns:
            DataFrame with calculated features
        """
        features = pd.DataFrame(index=df.index)

        # Basic returns
        features[f'{prefix}_ret_1d'] = df['price'].pct_change()
        features[f'{prefix}_ret_3d'] = df['price'].pct_change(3)
        features[f'{prefix}_ret_5d'] = df['price'].pct_change(5)
        features[f'{prefix}_ret_10d'] = df['price'].pct_change(10)

        # Volatility
        features[f'{prefix}_vol_10d'] = features[f'{prefix}_ret_1d'].rolling(10).std()
        features[f'{prefix}_vol_20d'] = features[f'{prefix}_ret_1d'].rolling(20).std()
        features[f'{prefix}_vol_30d'] = features[f'{prefix}_ret_1d'].rolling(30).std()

        # Trend indicators
        features[f'{prefix}_trend_20d'] = df['price'].pct_change(20)
        features[f'{prefix}_drawdown_30d'] = (df['price'] - df['price'].rolling(30).max()) / df['price'].rolling(30).max()

        # Crash/Boom flags (using 7% threshold as mentioned)
        features[f'{prefix}_crash_flag'] = (features[f'{prefix}_ret_1d'] < -0.07).astype(int)
        features[f'{prefix}_boom_flag'] = (features[f'{prefix}_ret_1d'] > 0.07).astype(int)

        # Market cap and volume features (if available)
        if 'market_cap' in df.columns:
            features[f'{prefix}_market_cap'] = df['market_cap']
            features[f'{prefix}_mc_change_1d'] = df['market_cap'].pct_change()

        if 'total_volume' in df.columns:
            features[f'{prefix}_volume'] = df['total_volume']
            features[f'{prefix}_volume_ratio'] = df['total_volume'] / df['total_volume'].rolling(20).mean()

        return features

    def get_crypto_regime_signals(self, crypto_features: pd.DataFrame) -> pd.DataFrame:
        """
        Generate crypto regime signals from features.

        Args:
            crypto_features: DataFrame with crypto features

        Returns:
            DataFrame with regime classifications
        """
        regimes = pd.DataFrame(index=crypto_features.index)

        # BTC-based regime classification
        btc_trend = crypto_features.get('btc_trend_20d', 0)
        btc_vol = crypto_features.get('btc_vol_30d', 0)
        btc_drawdown = crypto_features.get('btc_drawdown_30d', 0)

        # Simple regime logic
        conditions = [
            (btc_drawdown < -0.2) & (btc_trend < -0.1),  # Crash regime
            (btc_trend > 0.1) & (btc_vol < btc_vol.quantile(0.7)),  # Bull regime
            (btc_vol > btc_vol.quantile(0.8)),  # High volatility regime
        ]
        choices = ['crash', 'bull', 'volatile']
        default = 'normal'

        regimes['crypto_regime'] = np.select(conditions, choices, default=default)

        return regimes</content>
<parameter name="filePath">c:\Users\spreu\Documents\agentic_forecast\agents\crypto_data_agent.py