"""
Alpha Vantage Data Client for IB Forecast System

This module provides integration with Alpha Vantage APIs for fetching:
- Historical stock data (intraday, daily, weekly, monthly)
- Technical indicators
- News and sentiment data
- Fundamental data

Supports premium account features with 300 API calls per minute.
"""

import os
import time
import pandas as pd
from typing import Optional, Dict, Any, List
import requests
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlphaVantageClient:
    """
    Client for Alpha Vantage API integration.

    Handles rate limiting, error handling, and data formatting.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Alpha Vantage client.

        Args:
            api_key: Alpha Vantage API key. If None, loads from ALPHA_VANTAGE_API_KEY env var.
        """
        self.api_key = api_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        if not self.api_key:
            raise ValueError("Alpha Vantage API key not found. Set ALPHA_VANTAGE_API_KEY in .env file.")

        # Get entitlement parameter for premium accounts
        self.entitlement = os.getenv('ALPHA_VANTAGE_ENTITLEMENT', 'realtime')

        # Base URL for Alpha Vantage API
        self.base_url = "https://www.alphavantage.co/query"

        # Initialize requests session
        self.session = requests.Session()

        # Rate limiting: 300 calls per minute for premium
        self.rate_limit = 300
        self.time_window = 60  # seconds
        self.call_times = []

        logger.info(f"Alpha Vantage client initialized with premium rate limits (entitlement: {self.entitlement}).")

    def _check_rate_limit(self):
        """Check and enforce rate limiting."""
        current_time = time.time()

        # Remove calls outside the time window
        self.call_times = [t for t in self.call_times if current_time - t < self.time_window]

        if len(self.call_times) >= self.rate_limit:
            # Calculate sleep time based on the oldest call in the window
            # We need to wait until the oldest call expires
            oldest_call_time = self.call_times[0]
            sleep_time = self.time_window - (current_time - oldest_call_time) + 0.1 # Add buffer
            
            if sleep_time > 0:
                logger.warning(f"Rate limit reached ({len(self.call_times)} calls). Sleeping for {sleep_time:.2f} seconds.")
                time.sleep(sleep_time)
                
                # After sleeping, update current_time and filter again to ensure we are compliant
                current_time = time.time()
                self.call_times = [t for t in self.call_times if current_time - t < self.time_window]

        self.call_times.append(current_time)

    def _make_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a request to Alpha Vantage API with entitlement parameter.

        Args:
            params: Query parameters for the API call

        Returns:
            JSON response from the API
        """
        self._check_rate_limit()

        # Add entitlement parameter for premium accounts
        params['entitlement'] = self.entitlement
        params['apikey'] = self.api_key

        try:
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Check for API errors
            if 'Error Message' in data:
                raise Exception(f"Alpha Vantage API Error: {data['Error Message']}")
            if 'Note' in data and 'rate limit' in data['Note'].lower():
                raise Exception(f"Rate limit exceeded: {data['Note']}")

            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise

    def get_daily_data(self, symbol: str, outputsize: str = 'compact') -> pd.DataFrame:
        """
        Fetch daily time series data.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            outputsize: 'compact' (100 data points) or 'full' (20+ years)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': symbol,
                'outputsize': outputsize
            }

            data = self._make_request(params)

            # Extract time series data
            time_series_key = 'Time Series (Daily)'
            if time_series_key not in data:
                logger.error(f"No time series data found in response for {symbol}")
                logger.error(f"Available keys: {list(data.keys())}")
                if 'Error Message' in data:
                    logger.error(f"API Error: {data['Error Message']}")
                elif 'Information' in data:
                    logger.error(f"API Info: {data['Information']}")
                return pd.DataFrame()

            time_series = data[time_series_key]

            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            # Rename columns to match expected format
            column_mapping = {
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. adjusted close': 'adjusted_close',
                '6. volume': 'volume',
                '7. dividend amount': 'dividend_amount',
                '8. split coefficient': 'split_coefficient'
            }
            df = df.rename(columns=column_mapping)

            # Convert numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'adjusted_close', 'volume', 'dividend_amount', 'split_coefficient']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            logger.info(f"Fetched {len(df)} daily data points for {symbol}")
            return df
        except Exception as e:
            logger.error(f"Error fetching daily data for {symbol}: {e}")
            return pd.DataFrame()

    def get_intraday_data(self, symbol: str, interval: str = '5min', outputsize: str = 'compact') -> pd.DataFrame:
        """
        Fetch intraday time series data.

        Args:
            symbol: Stock symbol
            interval: '1min', '5min', '15min', '30min', '60min'
            outputsize: 'compact' or 'full'

        Returns:
            DataFrame with OHLCV data
        """
        try:
            params = {
                'function': 'TIME_SERIES_INTRADAY',
                'symbol': symbol,
                'interval': interval,
                'outputsize': outputsize
            }

            data = self._make_request(params)

            # Extract time series data
            time_series_key = f'Time Series ({interval})'
            if time_series_key not in data:
                logger.error(f"No time series data found in response for {symbol}")
                return pd.DataFrame()

            time_series = data[time_series_key]

            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            # Rename columns to match expected format
            column_mapping = {
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume'
            }
            df = df.rename(columns=column_mapping)

            # Convert numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            logger.info(f"Fetched {len(df)} intraday data points for {symbol} ({interval})")
            return df
        except Exception as e:
            logger.error(f"Error fetching intraday data for {symbol}: {e}")
            return pd.DataFrame()

    def get_technical_indicator(self, symbol: str, indicator: str, interval: str = 'daily', **kwargs) -> pd.DataFrame:
        """
        Fetch technical indicator data.

        Args:
            symbol: Stock symbol
            indicator: Indicator name (e.g., 'SMA', 'RSI', 'MACD')
            interval: 'daily', 'weekly', 'monthly', '1min', etc.
            **kwargs: Additional parameters for the indicator

        Returns:
            DataFrame with indicator values
        """
        self._check_rate_limit()

        try:
            # Map indicator names to method calls
            indicator_methods = {
                'SMA': self.ti.get_sma,
                'EMA': self.ti.get_ema,
                'RSI': self.ti.get_rsi,
                'MACD': self.ti.get_macd,
                'BBANDS': self.ti.get_bbands,
                'STOCH': self.ti.get_stoch,
                'ADX': self.ti.get_adx,
                'CCI': self.ti.get_cci,
                'AROON': self.ti.get_aroon,
                'MFI': self.ti.get_mfi,
                'ROC': self.ti.get_roc,
                'WILLR': self.ti.get_willr,
                'OBV': self.ti.get_obv,
                'ATR': self.ti.get_atr
            }

            if indicator not in indicator_methods:
                raise ValueError(f"Unsupported indicator: {indicator}")

            method = indicator_methods[indicator]
            data, meta_data = method(symbol=symbol, interval=interval, **kwargs)
            data.index = pd.to_datetime(data.index)
            data = data.sort_index()
            logger.info(f"Fetched {indicator} data for {symbol}")
            return data
        except Exception as e:
            logger.error(f"Error fetching {indicator} for {symbol}: {e}")
            return pd.DataFrame()

    def get_news_sentiment(self, tickers: List[str], time_from: Optional[str] = None,
                          limit: int = 50) -> pd.DataFrame:
        """
        Fetch news and sentiment data.

        Args:
            tickers: List of stock symbols
            time_from: Start time in YYYYMMDDTHHMM format
            limit: Maximum number of articles (max 1000)

        Returns:
            DataFrame with news articles and sentiment scores
        """
        self._check_rate_limit()

        try:
            tickers_str = ','.join(tickers)
            data, meta_data = self.ai.get_news_sentiment(
                tickers=tickers_str,
                time_from=time_from,
                limit=min(limit, 1000)
            )
            data.index = pd.to_datetime(data.index)
            data = data.sort_index()
            logger.info(f"Fetched {len(data)} news articles for {tickers}")
            return data
        except Exception as e:
            logger.error(f"Error fetching news sentiment: {e}")
            return pd.DataFrame()

    def get_company_overview(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch company overview/fundamentals.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with company information
        """
        self._check_rate_limit()

        try:
            data, meta_data = self.fd.get_company_overview(symbol=symbol)
            logger.info(f"Fetched company overview for {symbol}")
            return data.to_dict('records')[0] if not data.empty else {}
        except Exception as e:
            logger.error(f"Error fetching company overview for {symbol}: {e}")
            return {}

    def get_earnings_history(self, symbol: str) -> pd.DataFrame:
        """
        Fetch earnings history.

        Args:
            symbol: Stock symbol

        Returns:
            DataFrame with earnings data
        """
        self._check_rate_limit()

        try:
            data, meta_data = self.fd.get_earnings(symbol=symbol)
            data.index = pd.to_datetime(data.index)
            data = data.sort_index()
            logger.info(f"Fetched earnings history for {symbol}")
            return data
        except Exception as e:
            logger.error(f"Error fetching earnings for {symbol}: {e}")
            return pd.DataFrame()

    def search_symbol(self, keywords: str) -> pd.DataFrame:
        """
        Search for symbols by keywords.

        Args:
            keywords: Search keywords

        Returns:
            DataFrame with matching symbols
        """
        self._check_rate_limit()

        try:
            data, meta_data = self.ts.get_symbol_search(keywords=keywords)
            logger.info(f"Symbol search for '{keywords}' returned {len(data)} results")
            return data
        except Exception as e:
            logger.error(f"Error searching symbols: {e}")
            return pd.DataFrame()


# Convenience functions for common use cases
def get_stock_data(symbol: str, period: str = '2y', interval: str = 'daily') -> pd.DataFrame:
    """
    High-level function to get stock data for a given period.

    Args:
        symbol: Stock symbol
        period: '1y', '2y', '5y', 'max'
        interval: 'daily', 'weekly', 'monthly', or intraday like '5min'

    Returns:
        DataFrame with OHLCV data
    """
    client = AlphaVantageClient()

    if interval in ['daily', 'weekly', 'monthly']:
        if period == 'max':
            outputsize = 'full'
        else:
            outputsize = 'compact'  # Will be filtered later

        if interval == 'daily':
            data = client.get_daily_data(symbol, outputsize)
        elif interval == 'weekly':
            data, _ = client.ts.get_weekly(symbol)
            data.columns = ['open', 'high', 'low', 'close', 'volume']
        elif interval == 'monthly':
            data, _ = client.ts.get_monthly(symbol)
            data.columns = ['open', 'high', 'low', 'close', 'volume']

        data.index = pd.to_datetime(data.index)
        data = data.sort_index()

        # Filter by period if needed
        if period != 'max':
            years = int(period[:-1])
            cutoff_date = pd.Timestamp.now() - pd.DateOffset(years=years)
            data = data[data.index >= cutoff_date]

    else:
        # Intraday
        data = client.get_intraday_data(symbol, interval, 'full')

    return data


def get_technical_features(symbol: str, indicators: List[str] = None) -> pd.DataFrame:
    """
    Get multiple technical indicators as features.

    Args:
        indicators: List of indicator names. If None, uses common ones.

    Returns:
        DataFrame with technical indicators
    """
    if indicators is None:
        indicators = ['SMA', 'EMA', 'RSI', 'MACD', 'BBANDS']

    client = AlphaVantageClient()
    features = {}

    for indicator in indicators:
        if indicator == 'SMA':
            data = client.get_technical_indicator(symbol, 'SMA', time_period=20)
            features['SMA_20'] = data['SMA'] if 'SMA' in data.columns else data.iloc[:, 0]
        elif indicator == 'EMA':
            data = client.get_technical_indicator(symbol, 'EMA', time_period=20)
            features['EMA_20'] = data['EMA'] if 'EMA' in data.columns else data.iloc[:, 0]
        elif indicator == 'RSI':
            data = client.get_technical_indicator(symbol, 'RSI', time_period=14)
            features['RSI_14'] = data['RSI'] if 'RSI' in data.columns else data.iloc[:, 0]
        elif indicator == 'MACD':
            data = client.get_technical_indicator(symbol, 'MACD')
            for col in data.columns:
                features[f'MACD_{col}'] = data[col]
        elif indicator == 'BBANDS':
            data = client.get_technical_indicator(symbol, 'BBANDS', time_period=20)
            for col in data.columns:
                features[f'BBANDS_{col}'] = data[col]

    features_df = pd.DataFrame(features)
    features_df.index = pd.to_datetime(features_df.index)
    return features_df.sort_index()


if __name__ == "__main__":
    # Example usage
    client = AlphaVantageClient()

    # Test daily data
    print("Fetching AAPL daily data...")
    data = client.get_daily_data('AAPL', 'compact')
    print(f"Got {len(data)} data points")
    print(data.head())

    # Test technical indicator
    print("\nFetching AAPL RSI...")
    rsi = client.get_technical_indicator('AAPL', 'RSI', time_period=14)
    print(rsi.head())

    # Test news sentiment
    print("\nFetching news for AAPL...")
    news = client.get_news_sentiment(['AAPL'], limit=5)
    print(f"Got {len(news)} news articles")
    if not news.empty:
        print(news[['title', 'sentiment_score']].head())