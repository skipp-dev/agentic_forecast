"""
FMP (Financial Modeling Prep) Client

Handles API calls to Financial Modeling Prep for fundamentals data.
"""

import requests
import pandas as pd
import logging
from datetime import datetime
from typing import Optional, Dict, List, Union
import time
import os

logger = logging.getLogger(__name__)

class FMPClient:
    """
    Client for Financial Modeling Prep API.

    Provides access to:
    - Income statements
    - Balance sheets
    - Cash flow statements
    - Financial ratios
    - Company profiles
    - Earnings calendars
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FMP client.

        Args:
            api_key: FMP API key (optional, will use env var if not provided)
        """
        self.api_key = api_key or os.getenv('FMP_API_KEY')
        if not self.api_key:
            raise ValueError("FMP API key required. Set FMP_API_KEY env var or pass api_key parameter.")

        self.base_url = "https://financialmodelingprep.com/stable"
        self.session = requests.Session()

        # Rate limiting (FMP free tier: 250 requests/day, premium: higher)
        self.requests_today = 0
        self.last_reset_date = datetime.now().date()

        logger.info("Initialized FMP client")

    def _check_rate_limit(self):
        """Check and reset daily rate limit."""
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.requests_today = 0
            self.last_reset_date = today

        if self.requests_today >= 240:  # Leave some buffer
            logger.warning("Approaching FMP daily rate limit")
            time.sleep(1)  # Small delay

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Make API request to FMP.

        Args:
            endpoint: API endpoint (without base URL)
            params: Query parameters

        Returns:
            JSON response or None if failed
        """
        self._check_rate_limit()

        url = f"{self.base_url}/{endpoint}"
        params = params or {}
        params['apikey'] = self.api_key

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            self.requests_today += 1
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"FMP API request failed: {e}")
            return None

    def get_income_statement(self, symbol: str, period: str = 'annual', limit: int = 10) -> Optional[pd.DataFrame]:
        """
        Get income statement data.

        Args:
            symbol: Stock symbol
            period: 'annual' or 'quarterly'
            limit: Number of periods to retrieve

        Returns:
            DataFrame with income statement data
        """
        endpoint = "income-statement"
        params = {'symbol': symbol, 'period': period, 'limit': limit}

        data = self._make_request(endpoint, params)
        if data:
            df = pd.DataFrame(data)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df['symbol'] = symbol
            return df
        return None

    def get_balance_sheet(self, symbol: str, period: str = 'annual', limit: int = 10) -> Optional[pd.DataFrame]:
        """
        Get balance sheet data.

        Args:
            symbol: Stock symbol
            period: 'annual' or 'quarterly'
            limit: Number of periods to retrieve

        Returns:
            DataFrame with balance sheet data
        """
        endpoint = "balance-sheet-statement"
        params = {'symbol': symbol, 'period': period, 'limit': limit}

        data = self._make_request(endpoint, params)
        if data:
            df = pd.DataFrame(data)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df['symbol'] = symbol
            return df
        return None

    def get_cash_flow(self, symbol: str, period: str = 'annual', limit: int = 10) -> Optional[pd.DataFrame]:
        """
        Get cash flow statement data.

        Args:
            symbol: Stock symbol
            period: 'annual' or 'quarterly'
            limit: Number of periods to retrieve

        Returns:
            DataFrame with cash flow data
        """
        endpoint = "cash-flow-statement"
        params = {'symbol': symbol, 'period': period, 'limit': limit}

        data = self._make_request(endpoint, params)
        if data:
            df = pd.DataFrame(data)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df['symbol'] = symbol
            return df
        return None

    def get_ratios(self, symbol: str, period: str = 'annual', limit: int = 10) -> Optional[pd.DataFrame]:
        """
        Get financial ratios.

        Args:
            symbol: Stock symbol
            period: 'annual' or 'quarterly'
            limit: Number of periods to retrieve

        Returns:
            DataFrame with financial ratios
        """
        endpoint = "ratios"
        params = {'symbol': symbol, 'period': period, 'limit': limit}

        data = self._make_request(endpoint, params)
        if data:
            df = pd.DataFrame(data)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df['symbol'] = symbol
            return df
        return None

    def get_company_profile(self, symbol: str) -> Optional[Dict]:
        """
        Get company profile information.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with company profile data
        """
        endpoint = "profile"
        params = {'symbol': symbol}
        return self._make_request(endpoint, params)

    def get_earnings_calendar(self, symbol: Optional[str] = None, from_date: Optional[str] = None,
                            to_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Get earnings calendar data.

        Args:
            symbol: Specific symbol or None for all
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with earnings data
        """
        endpoint = "earnings-calendar"
        params = {}
        if symbol:
            params['symbol'] = symbol
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date

        data = self._make_request(endpoint, params)
        if data:
            df = pd.DataFrame(data)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            return df
        return None