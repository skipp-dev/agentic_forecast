"""
CoinGecko Client

Handles API calls to CoinGecko for cryptocurrency data.
"""

import requests
import pandas as pd
import logging
from datetime import datetime
from typing import Optional, Dict, List, Union
import time
import os

logger = logging.getLogger(__name__)

class CoinGeckoClient:
    """
    Client for CoinGecko API.

    Provides access to:
    - Cryptocurrency market data
    - Historical price data
    - Market cap and volume information
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize CoinGecko client.

        Args:
            api_key: CoinGecko API key (optional, free tier works without key)
        """
        self.api_key = api_key or os.getenv('COINGECKO_API_KEY')
        self.base_url = "https://api.coingecko.com/api/v3"
        self.session = requests.Session()

        # Rate limiting (CoinGecko free tier: 30 requests/minute)
        self.requests_per_minute = 0
        self.last_reset_time = time.time()

        logger.info("Initialized CoinGecko client")

    def _check_rate_limit(self):
        """Check and enforce rate limiting."""
        current_time = time.time()
        if current_time - self.last_reset_time >= 60:
            self.requests_per_minute = 0
            self.last_reset_time = current_time

        if self.requests_per_minute >= 25:  # Leave some buffer
            wait_time = 60 - (current_time - self.last_reset_time)
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                self.requests_per_minute = 0
                self.last_reset_time = time.time()

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Make API request to CoinGecko.

        Args:
            endpoint: API endpoint (without base URL)
            params: Query parameters

        Returns:
            JSON response or None if failed
        """
        self._check_rate_limit()

        url = f"{self.base_url}/{endpoint}"
        params = params or {}

        # Add API key if available
        if self.api_key:
            params['x_cg_demo_api_key'] = self.api_key

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            self.requests_per_minute += 1
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"CoinGecko API request failed: {e}")
            return None

    def get_coin_market_chart(self, coin_id: str, vs_currency: str = 'usd',
                            days: Union[int, str] = 'max') -> Optional[Dict]:
        """
        Get historical market data for a cryptocurrency.

        Args:
            coin_id: CoinGecko coin ID (e.g., 'bitcoin', 'ethereum')
            vs_currency: Target currency (default: 'usd')
            days: Number of days or 'max' for all available data

        Returns:
            Dictionary with prices, market_caps, and total_volumes
        """
        endpoint = f"coins/{coin_id}/market_chart"
        params = {
            'vs_currency': vs_currency,
            'days': days,
            'interval': 'daily'
        }

        return self._make_request(endpoint, params)

    def get_coin_list(self) -> Optional[List[Dict]]:
        """
        Get list of all supported coins.

        Returns:
            List of coin dictionaries
        """
        endpoint = "coins/list"
        return self._make_request(endpoint)

    def get_simple_price(self, coin_ids: List[str], vs_currencies: List[str]) -> Optional[Dict]:
        """
        Get simple price data for multiple coins.

        Args:
            coin_ids: List of coin IDs
            vs_currencies: List of target currencies

        Returns:
            Dictionary with price data
        """
        endpoint = "simple/price"
        params = {
            'ids': ','.join(coin_ids),
            'vs_currencies': ','.join(vs_currencies),
            'include_market_cap': 'true',
            'include_24hr_vol': 'true',
            'include_24hr_change': 'true'
        }

        return self._make_request(endpoint, params)

    def process_market_chart_data(self, data: Dict) -> pd.DataFrame:
        """
        Process raw market chart data into a clean DataFrame.

        Args:
            data: Raw data from get_coin_market_chart

        Returns:
            DataFrame with processed data
        """
        if not data:
            return pd.DataFrame()

        # Extract data arrays
        prices = data.get('prices', [])
        market_caps = data.get('market_caps', [])
        total_volumes = data.get('total_volumes', [])

        if not prices:
            return pd.DataFrame()

        # Convert to DataFrame
        df_data = []

        for i, (timestamp, price) in enumerate(prices):
            row = {
                'timestamp': datetime.fromtimestamp(timestamp / 1000),
                'price': price,
            }

            # Add market cap if available
            if i < len(market_caps):
                row['market_cap'] = market_caps[i][1]

            # Add volume if available
            if i < len(total_volumes):
                row['total_volume'] = total_volumes[i][1]

            df_data.append(row)

        df = pd.DataFrame(df_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        return df</content>
<parameter name="filePath">c:\Users\spreu\Documents\agentic_forecast\src\coingecko_client.py