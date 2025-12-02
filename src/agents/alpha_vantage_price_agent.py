"""
Alpha Vantage Price Agent

Handles daily EOD price data collection for stocks, ETFs, and indices.
Provides OHLCV data with adjustments for splits and dividends.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
from typing import Optional, List, Dict, Union
import time
import requests

from src.alpha_vantage_client import AlphaVantageClient

logger = logging.getLogger(__name__)

class AlphaVantagePriceAgent:
    """
    Agent for collecting daily price data from Alpha Vantage.

    Features:
    - Daily OHLCV data for stocks/ETFs
    - Automatic split/dividend adjustments
    - Rate limiting and error handling
    - Data validation and quality checks
    """

    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize the Alpha Vantage Price Agent.

        Args:
            api_key: Alpha Vantage API key (optional, will use env var if not provided)
            config: Configuration dictionary
        """
        self.api_key = api_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        if not self.api_key:
            raise ValueError("Alpha Vantage API key required. Set ALPHA_VANTAGE_API_KEY env var or pass api_key parameter.")

        self.config = config or {}
        self.client = AlphaVantageClient(api_key=self.api_key)

        # Data storage paths
        self.raw_data_path = Path('data/raw/prices/alpha_vantage')
        self.processed_data_path = Path('data/processed/prices')
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)

        # Rate limiting
        self.rate_limit = self.config.get('alpha_vantage', {}).get('rate_limit', 75)  # Premium tier
        self.requests_made = 0
        self.last_request_time = datetime.now()

        logger.info(f"Initialized Alpha Vantage Price Agent (rate limit: {self.rate_limit}/min)")

    def _rate_limit_wait(self):
        """Enforce rate limiting."""
        if self.requests_made >= self.rate_limit:
            # Calculate wait time until next minute
            time_since_last = (datetime.now() - self.last_request_time).total_seconds()
            if time_since_last < 60:
                wait_time = 60 - time_since_last
                logger.info(f"Rate limit reached, waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)

            # Reset counter
            self.requests_made = 0
            self.last_request_time = datetime.now()

    def fetch_daily_data(self, symbol: str, outputsize: str = 'full') -> Optional[pd.DataFrame]:
        """
        Fetch daily price data for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'SPY')
            outputsize: 'compact' (last 100 days) or 'full' (20+ years)

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            self._rate_limit_wait()

            logger.info(f"Fetching Alpha Vantage data for {symbol} (outputsize: {outputsize})")

            # Use TIME_SERIES_DAILY_ADJUSTED for stocks/ETFs
            df = self.client.get_daily_adjusted_data(symbol, outputsize=outputsize)

            if df is not None and not df.empty:
                # Add symbol column
                df['symbol'] = symbol

                # Ensure proper column order
                columns = ['symbol', 'open', 'high', 'low', 'close', 'adjusted_close',
                          'volume', 'dividend_amount', 'split_coefficient']
                df = df.reindex(columns=columns)

                logger.info(f"Successfully fetched {len(df)} days of data for {symbol}")
                self.requests_made += 1
                return df
            else:
                logger.warning(f"No data received for {symbol}")
                return None

        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return None

    def save_raw_data(self, df: pd.DataFrame, symbol: str):
        """
        Save raw data to parquet file.

        Args:
            df: DataFrame to save
            symbol: Symbol name for filename
        """
        if df is None or df.empty:
            return

        filename = f"{symbol}.parquet"
        filepath = self.raw_data_path / filename

        try:
            df.to_parquet(filepath, index=True)
            logger.info(f"Saved raw data for {symbol} to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save raw data for {symbol}: {e}")

    def load_raw_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load raw data from parquet file.

        Args:
            symbol: Symbol name

        Returns:
            DataFrame or None if file doesn't exist
        """
        filename = f"{symbol}.parquet"
        filepath = self.raw_data_path / filename

        if not filepath.exists():
            return None

        try:
            df = pd.read_parquet(filepath)
            logger.info(f"Loaded raw data for {symbol} from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Failed to load raw data for {symbol}: {e}")
            return None

    def update_symbol_data(self, symbol: str, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Update data for a symbol, using cache if available.

        Args:
            symbol: Trading symbol
            force_refresh: If True, ignore cache and fetch fresh data

        Returns:
            DataFrame with price data or None if failed
        """
        # Check cache first
        if not force_refresh:
            cached_data = self.load_raw_data(symbol)
            if cached_data is not None:
                # Check if data is recent enough (within last trading day)
                latest_date = cached_data.index.max()
                days_old = (datetime.now() - latest_date).days

                if days_old <= 1:  # Data is current
                    logger.info(f"Using cached data for {symbol} (last updated: {latest_date.date()})")
                    return cached_data

        # Fetch fresh data
        df = self.fetch_daily_data(symbol, outputsize='full')
        if df is not None:
            self.save_raw_data(df, symbol)
            return df

        return None

    def batch_update_symbols(self, symbols: List[str], force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Update data for multiple symbols.

        Args:
            symbols: List of trading symbols
            force_refresh: If True, ignore cache for all symbols

        Returns:
            Dictionary mapping symbols to their DataFrames
        """
        results = {}

        for i, symbol in enumerate(symbols):
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i+1}/{len(symbols)} symbols")

            try:
                df = self.update_symbol_data(symbol, force_refresh=force_refresh)
                if df is not None:
                    results[symbol] = df
                else:
                    logger.warning(f"Failed to update data for {symbol}")
            except Exception as e:
                logger.error(f"Error updating {symbol}: {e}")

        logger.info(f"Successfully updated {len(results)}/{len(symbols)} symbols")
        return results

    def get_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate basic price-based features from raw data.

        Args:
            df: Raw price DataFrame

        Returns:
            DataFrame with additional features
        """
        if df is None or df.empty:
            return df

        # Work on a copy
        features_df = df.copy()

        # Basic returns
        features_df['ret_1d'] = features_df['adjusted_close'].pct_change()
        features_df['ret_3d'] = features_df['adjusted_close'].pct_change(3)
        features_df['ret_5d'] = features_df['adjusted_close'].pct_change(5)
        features_df['ret_10d'] = features_df['adjusted_close'].pct_change(10)

        # Volatility (rolling std of returns)
        features_df['vol_10d'] = features_df['ret_1d'].rolling(10).std()
        features_df['vol_20d'] = features_df['ret_1d'].rolling(20).std()

        # Gap risk (overnight gap)
        features_df['gap_open_close'] = (features_df['open'] - features_df['close'].shift(1)) / features_df['close'].shift(1)

        # Dollar volume
        features_df['dollar_volume'] = features_df['adjusted_close'] * features_df['volume']

        # Intraday range
        features_df['intraday_range'] = (features_df['high'] - features_df['low']) / features_df['close']

        return features_df</content>
<parameter name="filePath">c:\Users\spreu\Documents\agentic_forecast\agents\alpha_vantage_price_agent.py