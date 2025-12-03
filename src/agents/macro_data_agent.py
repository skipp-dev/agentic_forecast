"""
Macro Economic Data Agent

Handles collection and processing of macro economic indicators including:
- Interest rates (Fed Funds, Treasury yields)
- Labor market data (unemployment, payrolls)
- Commodity prices (oil, gold, copper)
- Economic indicators (GDP, inflation)
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import requests
import yfinance as yf

# Add paths for imports
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)

class MacroDataAgent:
    """
    Agent responsible for collecting and processing macro economic data.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.macro_indicators = {
            # Interest Rates
            'fed_funds': {'ticker': '^IRX', 'name': 'Federal Funds Rate'},
            'treasury_10y': {'ticker': '^TNX', 'name': '10-Year Treasury Yield'},
            'treasury_2y': {'ticker': '^IRX', 'name': '2-Year Treasury Yield'},

            # Commodities
            'oil': {'ticker': 'CL=F', 'name': 'WTI Crude Oil'},
            'gold': {'ticker': 'GC=F', 'name': 'Gold Futures'},
            'copper': {'ticker': 'HG=F', 'name': 'Copper Futures'},

            # Economic Indicators (via FRED API if available)
            'unemployment': {'fred_id': 'UNRATE', 'name': 'Unemployment Rate'},
            'payrolls': {'fred_id': 'PAYEMS', 'name': 'Nonfarm Payrolls'},
            'gdp': {'fred_id': 'GDP', 'name': 'Gross Domestic Product'},
            'cpi': {'fred_id': 'CPIAUCSL', 'name': 'Consumer Price Index'},
        }

        # FRED API key from config
        self.fred_api_key = self.config.get('fred_api_key', os.getenv('FRED_API_KEY'))

    def collect_macro_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Collect macro economic data for the specified date range.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            Dictionary of DataFrames with macro data
        """
        logger.info(f"Collecting macro data from {start_date} to {end_date}")

        macro_data = {}

        # Collect market-based indicators (via Yahoo Finance)
        market_indicators = ['fed_funds', 'treasury_10y', 'treasury_2y', 'oil', 'gold', 'copper']

        for indicator in market_indicators:
            try:
                ticker = self.macro_indicators[indicator]['ticker']
                name = self.macro_indicators[indicator]['name']

                logger.info(f"Fetching {name} ({ticker})")
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)

                if not data.empty:
                    # Keep only Close price and rename
                    data = data[['Close']].rename(columns={'Close': indicator})
                    macro_data[indicator] = data
                    logger.info(f"Collected {len(data)} records for {indicator}")
                else:
                    logger.warning(f"No data available for {indicator}")

            except Exception as e:
                logger.error(f"Failed to collect {indicator}: {e}")

        # Collect economic indicators (via FRED API if available)
        if self.fred_api_key:
            economic_indicators = ['unemployment', 'payrolls', 'gdp', 'cpi']
            macro_data.update(self._collect_fred_data(economic_indicators, start_date, end_date))
        else:
            logger.warning("FRED API key not available, skipping economic indicators")

        return macro_data

    def _collect_fred_data(self, indicators: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Collect data from FRED (Federal Reserve Economic Data) API.
        """
        fred_data = {}

        base_url = "https://api.stlouisfed.org/fred/series/observations"

        for indicator in indicators:
            try:
                fred_id = self.macro_indicators[indicator]['fred_id']
                name = self.macro_indicators[indicator]['name']

                params = {
                    'series_id': fred_id,
                    'api_key': self.fred_api_key,
                    'file_type': 'json',
                    'observation_start': start_date,
                    'observation_end': end_date
                }

                response = requests.get(base_url, params=params)
                response.raise_for_status()

                data = response.json()

                if 'observations' in data:
                    # Convert to DataFrame
                    df = pd.DataFrame(data['observations'])
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    df = df[['value']].rename(columns={'value': indicator})
                    df[indicator] = pd.to_numeric(df[indicator], errors='coerce')

                    fred_data[indicator] = df
                    logger.info(f"Collected {len(df)} records for {indicator} from FRED")
                else:
                    logger.warning(f"No FRED data available for {indicator}")

            except Exception as e:
                logger.error(f"Failed to collect FRED data for {indicator}: {e}")

        return fred_data

    def process_macro_features(self, macro_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Process raw macro data into features suitable for modeling.

        Args:
            macro_data: Raw macro data dictionary

        Returns:
            Processed macro features DataFrame
        """
        logger.info("Processing macro features")

        if not macro_data:
            logger.warning("No macro data to process")
            return pd.DataFrame()

        # Combine all macro data into a single DataFrame
        combined_data = pd.DataFrame()

        for indicator, df in macro_data.items():
            if combined_data.empty:
                combined_data = df.copy()
            else:
                combined_data = combined_data.join(df, how='outer')

        # Forward fill missing values
        combined_data = combined_data.fillna(method='ffill')

        # Calculate additional features
        features_df = combined_data.copy()

        # Yield curve features
        if 'treasury_10y' in features_df.columns and 'treasury_2y' in features_df.columns:
            features_df['yield_curve_spread'] = features_df['treasury_10y'] - features_df['treasury_2y']
            features_df['yield_curve_slope'] = features_df['yield_curve_spread'] / features_df['treasury_10y']

        # Rate change features
        rate_columns = ['fed_funds', 'treasury_10y', 'treasury_2y']
        for col in rate_columns:
            if col in features_df.columns:
                features_df[f'{col}_change_1m'] = features_df[col].pct_change(30)
                features_df[f'{col}_change_3m'] = features_df[col].pct_change(90)

        # Commodity momentum
        commodity_columns = ['oil', 'gold', 'copper']
        for col in commodity_columns:
            if col in features_df.columns:
                features_df[f'{col}_returns_1m'] = features_df[col].pct_change(30)
                features_df[f'{col}_volatility_1m'] = features_df[col].rolling(30).std()

        # Economic indicators (monthly data)
        economic_indicators = ['unemployment', 'payrolls', 'gdp', 'cpi']
        for col in economic_indicators:
            if col in features_df.columns:
                features_df[f'{col}_change_1m'] = features_df[col].pct_change(30)
                features_df[f'{col}_change_3m'] = features_df[col].pct_change(90)

        logger.info(f"Generated {features_df.shape[1]} macro features")
        return features_df

    def get_macro_data(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Main method to collect and process macro data.

        Returns:
            Dictionary containing raw data and processed features
        """
        raw_data = self.collect_macro_data(start_date, end_date)
        processed_features = self.process_macro_features(raw_data)

        return {
            'raw_data': raw_data,
            'processed_features': processed_features,
            'collection_timestamp': datetime.now(),
            'indicators_collected': list(raw_data.keys())
        }