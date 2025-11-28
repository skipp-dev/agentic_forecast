"""
Frankfurter FX Client

Handles API calls to Frankfurter for foreign exchange data.
"""

import requests
import pandas as pd
import logging
from datetime import datetime
from typing import Optional, Dict, List, Union
import time
import os

logger = logging.getLogger(__name__)

class FrankfurterClient:
    """
    Client for Frankfurter FX API.

    Provides access to:
    - Current exchange rates
    - Historical exchange rates
    - Currency conversion
    """

    def __init__(self):
        """
        Initialize Frankfurter client.

        Frankfurter is free and doesn't require an API key.
        """
        self.base_url = "https://api.frankfurter.app"
        self.session = requests.Session()

        # Rate limiting (Frankfurter: no strict limits, but be respectful)
        self.requests_per_minute = 0
        self.last_reset_time = time.time()

        logger.info("Initialized Frankfurter FX client")

    def _check_rate_limit(self):
        """Check and enforce rate limiting."""
        current_time = time.time()
        if current_time - self.last_reset_time >= 60:
            self.requests_per_minute = 0
            self.last_reset_time = current_time

        if self.requests_per_minute >= 50:  # Be conservative
            wait_time = 60 - (current_time - self.last_reset_time)
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                self.requests_per_minute = 0
                self.last_reset_time = time.time()

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Make API request to Frankfurter.

        Args:
            endpoint: API endpoint (without base URL)
            params: Query parameters

        Returns:
            JSON response or None if failed
        """
        self._check_rate_limit()

        url = f"{self.base_url}/{endpoint}"

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            self.requests_per_minute += 1
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"Frankfurter API request failed: {e}")
            return None

    def get_latest_rates(self, base: str = 'EUR', symbols: Optional[List[str]] = None) -> Optional[Dict]:
        """
        Get latest exchange rates.

        Args:
            base: Base currency (default: EUR)
            symbols: List of target currencies

        Returns:
            Dictionary with rates data
        """
        endpoint = "latest"
        params = {'from': base}
        if symbols:
            params['to'] = ','.join(symbols)

        return self._make_request(endpoint, params)

    def get_historical_rates(self, date: str, base: str = 'EUR',
                           symbols: Optional[List[str]] = None) -> Optional[Dict]:
        """
        Get exchange rates for a specific date.

        Args:
            date: Date in YYYY-MM-DD format
            base: Base currency
            symbols: List of target currencies

        Returns:
            Dictionary with rates data
        """
        endpoint = date
        params = {'from': base}
        if symbols:
            params['to'] = ','.join(symbols)

        return self._make_request(endpoint, params)

    def get_time_series(self, start_date: str, end_date: str, base: str = 'EUR',
                       symbols: Optional[List[str]] = None) -> Optional[Dict]:
        """
        Get exchange rates for a date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            base: Base currency
            symbols: List of target currencies

        Returns:
            Dictionary with time series data
        """
        endpoint = f"{start_date}..{end_date}"
        params = {'from': base}
        if symbols:
            params['to'] = ','.join(symbols)

        return self._make_request(endpoint, params)

    def process_time_series_data(self, data: Dict) -> pd.DataFrame:
        """
        Process raw time series data into a clean DataFrame.

        Args:
            data: Raw data from get_time_series

        Returns:
            DataFrame with processed FX data
        """
        if not data or 'rates' not in data:
            return pd.DataFrame()

        rates_data = data['rates']
        base_currency = data.get('base', 'EUR')

        # Convert to long format
        records = []
        for date_str, rates in rates_data.items():
            for target_currency, rate in rates.items():
                records.append({
                    'date': pd.to_datetime(date_str),
                    'base': base_currency,
                    'quote': target_currency,
                    'rate': rate,
                    'pair': f"{base_currency}{target_currency}"
                })

        df = pd.DataFrame(records)
        return df</content>
<parameter name="filePath">c:\Users\spreu\Documents\agentic_forecast\src\frankfurter_client.py