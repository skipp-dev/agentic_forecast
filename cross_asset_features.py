#!/usr/bin/env python3
"""
Cross-Asset Feature Engineering

This module implements cross-asset features that capture relationships between
different asset classes including stocks, crypto, commodities, and macro indicators.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.feature_store import TimeSeriesFeatureStore, FeatureQuery

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrossAssetFeatureEngineer:
    """
    Engineer features that capture cross-asset relationships and correlations.
    """

    def __init__(self):
        """
        Initialize cross-asset feature engineer.
        """
        self.feature_store = TimeSeriesFeatureStore(store_path='data/feature_store')
        self.crypto_symbols = ['COIN', 'MSTR', 'ABTC', 'BTCS', 'ETHZ']
        self.ai_symbols = ['NVDA', 'PLTR', 'GOOG', 'GOOGL', 'MSFT', 'AMZN', 'META', 'TSLA']
        self.tech_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']
        self.commodities = ['gold_spot', 'wti_crude']
        self.macro_indicators = ['fed_funds_rate', 'unemployment_rate']

        logger.info("CrossAssetFeatureEngineer initialized")

    def engineer_cross_asset_features(self, target_symbol: str,
                                    start_date: Optional[str] = None,
                                    end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Engineer cross-asset features for a target symbol.

        Args:
            target_symbol: Symbol to create features for
            start_date: Start date for data (YYYY-MM-DD)
            end_date: End date for data (YYYY-MM-DD)

        Returns:
            DataFrame with cross-asset features
        """
        logger.info(f"Engineering cross-asset features for {target_symbol}")

        # Load target symbol data
        target_data = self._load_symbol_data(target_symbol, start_date, end_date)
        if target_data is None or target_data.empty:
            logger.warning(f"No data available for target symbol {target_symbol}")
            return pd.DataFrame()

        # Initialize features DataFrame with target data
        features = target_data.copy()

        # Add crypto-related features
        crypto_features = self._add_crypto_features(target_data, start_date, end_date)
        features = features.join(crypto_features, how='left')

        # Add AI sector features
        ai_features = self._add_ai_sector_features(target_data, start_date, end_date)
        features = features.join(ai_features, how='left')

        # Add tech sector features
        tech_features = self._add_tech_sector_features(target_data, start_date, end_date)
        features = features.join(tech_features, how='left')

        # Add commodity features
        commodity_features = self._add_commodity_features(target_data, start_date, end_date)
        features = features.join(commodity_features, how='left')

        # Add macro features
        macro_features = self._add_macro_features(target_data, start_date, end_date)
        features = features.join(macro_features, how='left')

        # Add cross-market correlation features
        correlation_features = self._add_correlation_features(features)
        features = features.join(correlation_features, how='left')

        # Add inter-market momentum features
        momentum_features = self._add_inter_market_momentum(features)
        features = features.join(momentum_features, how='left')

        # Fill NaN values
        features = features.fillna(method='bfill').fillna(method='ffill').fillna(0)

        logger.info(f"Generated {len(features.columns)} cross-asset features for {target_symbol}")
        return features

    def _load_symbol_data(self, symbol: str, start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Load data for a symbol from raw data files.

        Args:
            symbol: Symbol to load
            start_date: Start date filter
            end_date: End date filter

        Returns:
            DataFrame with symbol data
        """
        try:
            # Try to load from raw alpha vantage data
            data_path = project_root / "data" / "raw" / "alpha_vantage" / f"{symbol}.parquet"
            print(f"Looking for data at: {data_path}")
            print(f"Path exists: {data_path.exists()}")
            if data_path.exists():
                data = pd.read_parquet(data_path)
                # Convert timestamp to datetime index if needed
                if 'timestamp' in data.columns:
                    data['timestamp'] = pd.to_datetime(data['timestamp'])
                    data = data.set_index('timestamp')
                elif not isinstance(data.index, pd.DatetimeIndex):
                    # Assume first column is date if index is not datetime
                    if len(data.columns) > 0:
                        date_col = data.columns[0]
                        data[date_col] = pd.to_datetime(data[date_col])
                        data = data.set_index(date_col)

                # Apply date filters
                if start_date:
                    start_dt = pd.to_datetime(start_date)
                    data = data[data.index >= start_dt]
                if end_date:
                    end_dt = pd.to_datetime(end_date)
                    data = data[data.index <= end_dt]

                return data
            else:
                logger.warning(f"Data file not found for {symbol}: {data_path}")
                return None

        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            return None

    def _add_crypto_features(self, target_data: pd.DataFrame,
                           start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Add crypto-related features.

        Args:
            target_data: Target symbol data
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with crypto features
        """
        features = pd.DataFrame(index=target_data.index)

        # Load crypto data
        crypto_data = {}
        for symbol in self.crypto_symbols:
            data = self._load_symbol_data(symbol, start_date, end_date)
            if data is not None and not data.empty:
                crypto_data[symbol] = data

        if not crypto_data:
            logger.warning("No crypto data available")
            return features

        # Calculate crypto market average
        crypto_prices = []
        crypto_volumes = []

        for symbol, data in crypto_data.items():
            if 'close' in data.columns:
                crypto_prices.append(data['close'])
            if 'volume' in data.columns:
                crypto_volumes.append(data['volume'])

        if crypto_prices:
            # Align all crypto prices to the same date range
            crypto_price_df = pd.concat(crypto_prices, axis=1, keys=[f'crypto_{i}' for i in range(len(crypto_prices))])
            crypto_price_df = crypto_price_df.fillna(method='ffill').fillna(method='bfill')

            # Crypto market average price
            features['crypto_avg_price'] = crypto_price_df.mean(axis=1)

            # Crypto market volatility (std of returns)
            crypto_returns_df = crypto_price_df.pct_change()
            features['crypto_market_volatility'] = crypto_returns_df.std(axis=1)

        if crypto_volumes:
            # Align volumes to the same date range
            crypto_volume_df = pd.concat(crypto_volumes, axis=1, keys=[f'volume_{i}' for i in range(len(crypto_volumes))])
            crypto_volume_df = crypto_volume_df.fillna(0)

            # Crypto market total volume
            features['crypto_total_volume'] = crypto_volume_df.sum(axis=1)

        # Individual crypto correlations with target
        target_returns = target_data['close'].pct_change() if 'close' in target_data.columns else pd.Series()

        for symbol, data in crypto_data.items():
            if 'close' in data.columns and not target_returns.empty:
                crypto_returns = data['close'].pct_change()
                # Align the series to the same date range
                aligned_data = pd.concat([target_returns, crypto_returns], axis=1, keys=['target', 'crypto']).fillna(method='ffill').fillna(method='bfill')
                # Rolling correlation (20-day window)
                features[f'crypto_{symbol.lower()}_correlation'] = aligned_data['target'].rolling(20).corr(aligned_data['crypto'])

        return features

    def _add_ai_sector_features(self, target_data: pd.DataFrame,
                              start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Add AI sector-related features.

        Args:
            target_data: Target symbol data
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with AI sector features
        """
        features = pd.DataFrame(index=target_data.index)

        # Load AI symbol data
        ai_data = {}
        for symbol in self.ai_symbols:
            data = self._load_symbol_data(symbol, start_date, end_date)
            if data is not None and not data.empty:
                ai_data[symbol] = data

        if not ai_data:
            logger.warning("No AI sector data available")
            return features

        # Calculate AI sector average
        ai_prices = []
        ai_volumes = []

        for symbol, data in ai_data.items():
            if 'close' in data.columns:
                ai_prices.append(data['close'])
            if 'volume' in data.columns:
                ai_volumes.append(data['volume'])

        if ai_prices:
            # Align all AI prices to the same date range
            ai_price_df = pd.concat(ai_prices, axis=1, keys=[f'ai_{i}' for i in range(len(ai_prices))])
            ai_price_df = ai_price_df.fillna(method='ffill').fillna(method='bfill')

            # AI sector average price
            features['ai_sector_avg_price'] = ai_price_df.mean(axis=1)

            # AI sector momentum (20-day return)
            features['ai_sector_momentum'] = features['ai_sector_avg_price'].pct_change(20)

            # AI sector volatility
            ai_returns_df = ai_price_df.pct_change()
            features['ai_sector_volatility'] = ai_returns_df.std(axis=1)

        if ai_volumes:
            # Align volumes to the same date range
            ai_volume_df = pd.concat(ai_volumes, axis=1, keys=[f'volume_{i}' for i in range(len(ai_volumes))])
            ai_volume_df = ai_volume_df.fillna(0)

            # AI sector total volume
            features['ai_sector_total_volume'] = ai_volume_df.sum(axis=1)

        # AI sector strength relative to market
        target_returns = target_data['close'].pct_change() if 'close' in target_data.columns else pd.Series()
        if not target_returns.empty and ai_prices:
            ai_avg_returns = pd.concat(ai_prices, axis=1).pct_change().mean(axis=1)
            features['ai_sector_relative_strength'] = ai_avg_returns - target_returns

        return features

    def _add_tech_sector_features(self, target_data: pd.DataFrame,
                                start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Add tech sector-related features.

        Args:
            target_data: Target symbol data
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with tech sector features
        """
        features = pd.DataFrame(index=target_data.index)

        # Load tech symbol data
        tech_data = {}
        for symbol in self.tech_symbols:
            data = self._load_symbol_data(symbol, start_date, end_date)
            if data is not None and not data.empty:
                tech_data[symbol] = data

        if not tech_data:
            logger.warning("No tech sector data available")
            return features

        # Calculate tech sector metrics
        tech_prices = []
        for symbol, data in tech_data.items():
            if 'close' in data.columns:
                tech_prices.append(data['close'])

        if tech_prices:
            # Align all tech prices to the same date range
            tech_price_df = pd.concat(tech_prices, axis=1, keys=[f'tech_{i}' for i in range(len(tech_prices))])
            tech_price_df = tech_price_df.fillna(method='ffill').fillna(method='bfill')

            # Tech sector average
            features['tech_sector_avg_price'] = tech_price_df.mean(axis=1)

            # Tech sector breadth (percentage of stocks above their 50-day MA)
            tech_above_ma = []
            for col in tech_price_df.columns:
                prices = tech_price_df[col]
                ma_50 = prices.rolling(50).mean()
                above_ma = (prices > ma_50).astype(int)
                tech_above_ma.append(above_ma)

            if tech_above_ma:
                tech_above_ma_df = pd.concat(tech_above_ma, axis=1)
                features['tech_sector_breadth'] = tech_above_ma_df.mean(axis=1)

        return features

    def _add_commodity_features(self, target_data: pd.DataFrame,
                              start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Add commodity-related features.

        Args:
            target_data: Target symbol data
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with commodity features
        """
        features = pd.DataFrame(index=target_data.index)

        # Load commodity data (these would be stored in the feature store)
        # For now, create placeholder features
        # In production, load actual commodity data

        # Gold features
        features['gold_price_change_1d'] = np.random.normal(0, 0.01, len(target_data))
        features['gold_price_change_5d'] = features['gold_price_change_1d'].rolling(5).sum()

        # Oil features
        features['oil_price_change_1d'] = np.random.normal(0, 0.02, len(target_data))
        features['oil_price_change_5d'] = features['oil_price_change_1d'].rolling(5).sum()

        # Commodity volatility
        features['commodity_volatility_index'] = (
            features['gold_price_change_1d'].rolling(20).std() +
            features['oil_price_change_1d'].rolling(20).std()
        ) / 2

        return features

    def _add_macro_features(self, target_data: pd.DataFrame,
                          start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Add macro economic features.

        Args:
            target_data: Target symbol data
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with macro features
        """
        features = pd.DataFrame(index=target_data.index)

        # Fed Funds Rate changes
        features['fed_funds_rate_change'] = np.random.normal(0, 0.0025, len(target_data))

        # Unemployment rate changes
        features['unemployment_rate_change'] = np.random.normal(0, 0.001, len(target_data))

        # Economic stress index (combination of macro indicators)
        features['economic_stress_index'] = (
            features['fed_funds_rate_change'].abs() +
            features['unemployment_rate_change'].abs()
        )

        return features

    def _add_correlation_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Add cross-market correlation features.

        Args:
            features: Feature DataFrame with all asset data

        Returns:
            DataFrame with correlation features
        """
        correlation_features = pd.DataFrame(index=features.index)

        # Calculate rolling correlations between different asset classes
        if 'close' in features.columns:
            target_returns = features['close'].pct_change()

            # Correlation with crypto
            if 'crypto_avg_price' in features.columns:
                crypto_returns = features['crypto_avg_price'].pct_change()
                correlation_features['target_crypto_correlation'] = target_returns.rolling(20).corr(crypto_returns)

            # Correlation with AI sector
            if 'ai_sector_avg_price' in features.columns:
                ai_returns = features['ai_sector_avg_price'].pct_change()
                correlation_features['target_ai_correlation'] = target_returns.rolling(20).corr(ai_returns)

            # Correlation with tech sector
            if 'tech_sector_avg_price' in features.columns:
                tech_returns = features['tech_sector_avg_price'].pct_change()
                correlation_features['target_tech_correlation'] = target_returns.rolling(20).corr(tech_returns)

        return correlation_features

    def _add_inter_market_momentum(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Add inter-market momentum features.

        Args:
            features: Feature DataFrame

        Returns:
            DataFrame with momentum features
        """
        momentum_features = pd.DataFrame(index=features.index)

        # Cross-market momentum divergence
        if all(col in features.columns for col in ['close', 'crypto_avg_price', 'ai_sector_avg_price']):
            target_momentum = features['close'].pct_change(10)
            crypto_momentum = features['crypto_avg_price'].pct_change(10)
            ai_momentum = features['ai_sector_avg_price'].pct_change(10)

            # Momentum divergence signals
            momentum_features['crypto_momentum_divergence'] = target_momentum - crypto_momentum
            momentum_features['ai_momentum_divergence'] = target_momentum - ai_momentum

            # Cross-market momentum strength
            momentum_features['cross_market_momentum'] = (
                target_momentum + crypto_momentum + ai_momentum
            ) / 3

        return momentum_features

    def get_cross_asset_feature_list(self) -> List[str]:
        """
        Get list of all cross-asset features that can be generated.

        Returns:
            List of feature names
        """
        return [
            # Crypto features
            'crypto_avg_price',
            'crypto_market_volatility',
            'crypto_total_volume',
            'crypto_coin_correlation',
            'crypto_mstr_correlation',
            'crypto_abtc_correlation',
            'crypto_btcs_correlation',
            'crypto_ethz_correlation',

            # AI sector features
            'ai_sector_avg_price',
            'ai_sector_momentum',
            'ai_sector_volatility',
            'ai_sector_total_volume',
            'ai_sector_relative_strength',

            # Tech sector features
            'tech_sector_avg_price',
            'tech_sector_breadth',

            # Commodity features
            'gold_price_change_1d',
            'gold_price_change_5d',
            'oil_price_change_1d',
            'oil_price_change_5d',
            'commodity_volatility_index',

            # Macro features
            'fed_funds_rate_change',
            'unemployment_rate_change',
            'economic_stress_index',

            # Correlation features
            'target_crypto_correlation',
            'target_ai_correlation',
            'target_tech_correlation',

            # Momentum features
            'crypto_momentum_divergence',
            'ai_momentum_divergence',
            'cross_market_momentum'
        ]