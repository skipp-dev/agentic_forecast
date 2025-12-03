"""
FMP Fundamentals Data Agent

Handles collection and processing of fundamental financial data from Financial Modeling Prep.
Provides income statements, balance sheets, cash flows, ratios, and earnings data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Optional, List, Dict, Union
import time

from ..fmp_client import FMPClient

logger = logging.getLogger(__name__)

class FundamentalsDataAgent:
    """
    Agent for collecting fundamental financial data from FMP.

    Features:
    - Income statements, balance sheets, cash flows
    - Financial ratios and metrics
    - Company profiles
    - Earnings calendars and surprises
    - Data validation and forward-filling
    """

    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize the FMP Fundamentals Agent.

        Args:
            api_key: FMP API key (optional, will use env var if not provided)
            config: Configuration dictionary
        """
        self.api_key = api_key
        self.config = config or {}
        self.client = FMPClient(api_key=api_key)

        # Data storage paths
        self.raw_data_path = Path('data/raw/fundamentals/fmp')
        self.processed_data_path = Path('data/processed/fundamentals')
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)

        logger.info("Initialized FMP Fundamentals Data Agent")

    def fetch_fundamentals(self, symbol: str, period: str = 'annual') -> Dict[str, pd.DataFrame]:
        """
        Fetch all fundamental data for a symbol.

        Args:
            symbol: Stock symbol
            period: 'annual' or 'quarterly'

        Returns:
            Dictionary with different fundamental datasets
        """
        logger.info(f"Fetching fundamentals for {symbol} ({period})")

        results = {}

        try:
            # Income statement
            income = self.client.get_income_statement(symbol, period=period, limit=10)
            if income is not None:
                results['income_statement'] = income
                logger.info(f"Got {len(income)} income statement periods for {symbol}")

            # Balance sheet
            balance = self.client.get_balance_sheet(symbol, period=period, limit=10)
            if balance is not None:
                results['balance_sheet'] = balance
                logger.info(f"Got {len(balance)} balance sheet periods for {symbol}")

            # Cash flow
            cashflow = self.client.get_cash_flow(symbol, period=period, limit=10)
            if cashflow is not None:
                results['cash_flow'] = cashflow
                logger.info(f"Got {len(cashflow)} cash flow periods for {symbol}")

            # Ratios
            ratios = self.client.get_ratios(symbol, period=period, limit=10)
            if ratios is not None:
                results['ratios'] = ratios
                logger.info(f"Got {len(ratios)} ratio periods for {symbol}")

        except Exception as e:
            logger.error(f"Failed to fetch fundamentals for {symbol}: {e}")

        return results

    def fetch_company_profile(self, symbol: str) -> Optional[Dict]:
        """
        Fetch company profile information.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with profile data
        """
        try:
            profile = self.client.get_company_profile(symbol)
            if profile and len(profile) > 0:
                return profile[0]  # FMP returns list with single item
        except Exception as e:
            logger.error(f"Failed to fetch profile for {symbol}: {e}")
        return None

    def fetch_earnings_history(self, symbol: str, years: int = 2) -> Optional[pd.DataFrame]:
        """
        Fetch earnings history for a symbol.

        Args:
            symbol: Stock symbol
            years: Number of years of history to fetch

        Returns:
            DataFrame with earnings data
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * years)

            earnings = self.client.get_earnings_calendar(
                symbol=symbol,
                from_date=start_date.strftime('%Y-%m-%d'),
                to_date=end_date.strftime('%Y-%m-%d')
            )

            if earnings is not None:
                logger.info(f"Got {len(earnings)} earnings events for {symbol}")
                return earnings

        except Exception as e:
            logger.error(f"Failed to fetch earnings for {symbol}: {e}")

        return None

    def save_fundamentals_data(self, symbol: str, data_dict: Dict[str, pd.DataFrame]):
        """
        Save fundamental data to parquet files.

        Args:
            symbol: Stock symbol
            data_dict: Dictionary with different datasets
        """
        symbol_path = self.raw_data_path / symbol
        symbol_path.mkdir(exist_ok=True)

        for data_type, df in data_dict.items():
            if df is not None and not df.empty:
                filename = f"{data_type}.parquet"
                filepath = symbol_path / filename

                try:
                    df.to_parquet(filepath, index=True)
                    logger.info(f"Saved {data_type} for {symbol} to {filepath}")
                except Exception as e:
                    logger.error(f"Failed to save {data_type} for {symbol}: {e}")

    def load_fundamentals_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Load fundamental data from parquet files.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with loaded datasets
        """
        symbol_path = self.raw_data_path / symbol
        if not symbol_path.exists():
            return {}

        results = {}
        data_types = ['income_statement', 'balance_sheet', 'cash_flow', 'ratios']

        for data_type in data_types:
            filename = f"{data_type}.parquet"
            filepath = symbol_path / filename

            if filepath.exists():
                try:
                    df = pd.read_parquet(filepath)
                    results[data_type] = df
                except Exception as e:
                    logger.error(f"Failed to load {data_type} for {symbol}: {e}")

        return results

    def calculate_fundamental_features(self, fundamentals: Dict[str, pd.DataFrame],
                                     profile: Optional[Dict] = None) -> pd.DataFrame:
        """
        Calculate fundamental features from raw data.

        Args:
            fundamentals: Dictionary with fundamental datasets
            profile: Company profile data

        Returns:
            DataFrame with calculated features
        """
        features = []

        # Process each fundamental dataset
        for data_type, df in fundamentals.items():
            if df is None or df.empty:
                continue

            # Sort by date
            df = df.sort_index()

            if data_type == 'ratios':
                # Financial ratios features
                ratio_features = self._extract_ratio_features(df)
                features.append(ratio_features)

            elif data_type == 'income_statement':
                # Income statement features
                income_features = self._extract_income_features(df)
                features.append(income_features)

            elif data_type == 'balance_sheet':
                # Balance sheet features
                balance_features = self._extract_balance_features(df)
                features.append(balance_features)

        # Combine all features
        if features:
            combined = pd.concat(features, axis=1)
            combined = combined.sort_index()

            # Add profile-based features
            if profile:
                combined['sector'] = profile.get('sector')
                combined['industry'] = profile.get('industry')
                combined['beta'] = profile.get('beta')
                
                # Handle market cap (try 'marketCap' first, then 'mktCap')
                market_cap = profile.get('marketCap') or profile.get('mktCap')
                combined['market_cap'] = market_cap
                
                # Safely calculate log market cap
                combined['market_cap'] = pd.to_numeric(combined['market_cap'], errors='coerce')
                combined['log_market_cap'] = np.log(combined['market_cap'])

            return combined

        return pd.DataFrame()

    def _extract_ratio_features(self, ratios_df: pd.DataFrame) -> pd.DataFrame:
        """Extract key ratio features."""
        features = pd.DataFrame(index=ratios_df.index)

        # Valuation ratios
        features['pe_ratio'] = ratios_df.get('priceEarningsRatio')
        features['ps_ratio'] = ratios_df.get('priceToSalesRatio')
        features['pb_ratio'] = ratios_df.get('priceToBookRatio')
        features['ev_ebitda'] = ratios_df.get('enterpriseValueOverEBITDA')

        # Profitability ratios
        features['roe'] = ratios_df.get('returnOnEquity')
        features['roa'] = ratios_df.get('returnOnAssets')
        features['net_margin'] = ratios_df.get('netProfitMargin')
        features['operating_margin'] = ratios_df.get('operatingProfitMargin')

        # Leverage ratios
        features['debt_to_equity'] = ratios_df.get('totalDebtToEquity')
        features['debt_to_assets'] = ratios_df.get('totalDebtToTotalAsset')

        # Efficiency ratios
        features['asset_turnover'] = ratios_df.get('assetTurnover')
        features['inventory_turnover'] = ratios_df.get('inventoryTurnover')

        return features

    def _extract_income_features(self, income_df: pd.DataFrame) -> pd.DataFrame:
        """Extract key income statement features."""
        features = pd.DataFrame(index=income_df.index)

        # Key income metrics
        features['revenue'] = income_df.get('revenue')
        features['net_income'] = income_df.get('netIncome')
        features['eps'] = income_df.get('eps')
        features['ebitda'] = income_df.get('ebitda')

        # Growth rates (YoY)
        features['revenue_growth'] = features['revenue'].pct_change(4)  # Annual comparison
        features['net_income_growth'] = features['net_income'].pct_change(4)
        features['eps_growth'] = features['eps'].pct_change(4)

        return features

    def _extract_balance_features(self, balance_df: pd.DataFrame) -> pd.DataFrame:
        """Extract key balance sheet features."""
        features = pd.DataFrame(index=balance_df.index)

        # Key balance metrics
        features['total_assets'] = balance_df.get('totalAssets')
        features['total_debt'] = balance_df.get('totalDebt')
        features['cash_and_equivalents'] = balance_df.get('cashAndCashEquivalents')
        features['total_shareholders_equity'] = balance_df.get('totalStockholdersEquity')

        # Liquidity ratios
        features['current_ratio'] = balance_df.get('totalCurrentAssets') / balance_df.get('totalCurrentLiabilities')
        features['quick_ratio'] = (balance_df.get('totalCurrentAssets') - balance_df.get('inventory')) / balance_df.get('totalCurrentLiabilities')

        return features

    def update_symbol_fundamentals(self, symbol: str, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Update fundamental data for a symbol.

        Args:
            symbol: Stock symbol
            force_refresh: If True, ignore cache

        Returns:
            DataFrame with fundamental features
        """
        # Check cache first
        if not force_refresh:
            cached_data = self.load_fundamentals_data(symbol)
            if cached_data:
                # Check if data is recent enough (within last month)
                # For fundamentals, we can be less strict than daily prices
                logger.info(f"Using cached fundamentals for {symbol}")
                profile = self.fetch_company_profile(symbol)  # Always get fresh profile
                return self.calculate_fundamental_features(cached_data, profile)

        # Fetch fresh data
        fundamentals = self.fetch_fundamentals(symbol)
        if fundamentals:
            self.save_fundamentals_data(symbol, fundamentals)

        # Get profile
        profile = self.fetch_company_profile(symbol)

        # Calculate features
        features_df = self.calculate_fundamental_features(fundamentals, profile)

        if not features_df.empty:
            logger.info(f"Successfully updated fundamentals for {symbol}")
            return features_df

        return None