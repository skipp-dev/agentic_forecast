#!/usr/bin/env python3
"""
Feature Engineering Pipeline for Agentic Forecasting System

This script generates features for all symbols based on feature_config.yaml:
- Price-based features (returns, volatility, rolling statistics)
- Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- Calendar/seasonality features (day-of-week, month-of-year, etc.)

Usage:
    python run_features.py [--symbols SYMBOL1 SYMBOL2] [--experiment EXPERIMENT]
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import logging
import yaml
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import with proper path handling
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# from src.data_loader import DataLoader  # Not needed for basic feature engineering
# from src.config.config_loader import get_feature_config  # Using direct YAML loading

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/feature_engineering.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Feature engineering pipeline based on configuration.
    """

    def __init__(self, config_path: str = "config/feature_config.yaml"):
        """
        Initialize feature engineer.

        Args:
            config_path: Path to feature configuration file (deprecated - now uses config_loader)
        """
        self.config_path = Path(config_path)  # Keep for backward compatibility

        # Load config directly from YAML
        config_file = Path(__file__).parent.parent / "config" / "feature_config.yaml"
        if config_file.exists():
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Fallback to basic config
            self.config = {
                'feature_groups': {
                    'price_basic': {
                        'features': {
                            'returns': {'type': 'returns', 'params': {'periods': [1, 5]}},
                            'volatility': {'type': 'rolling_std', 'params': {'window': [5, 10]}},
                            'sma': {'type': 'sma', 'params': {'window': [20, 50]}}
                        }
                    }
                },
                'activations': {'default': ['price_basic']}
            }

        # self.data_loader = DataLoader()  # Not needed for basic feature engineering

        logger.info(f"FeatureEngineer initialized with config: {config_path}")



    def engineer_features_for_symbol(self, symbol: str, data: pd.DataFrame,
                                   experiment: str = "baseline",
                                   dynamic_groups: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Engineer features for a single symbol.

        Args:
            symbol: Stock symbol
            data: Raw OHLCV data
            experiment: Feature experiment to use
            dynamic_groups: Optional list of feature groups to use instead of experiment config

        Returns:
            DataFrame with engineered features
        """
        logger.info(f"Engineering features for {symbol} using experiment: {experiment}")

        # Validate input data
        required_columns = ['close', 'high', 'low', 'open', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns for {symbol}: {missing_columns}")
        if data.empty:
            raise ValueError(f"Empty data provided for {symbol}")

        # Start with raw data
        features_df = data.copy()

        # Get feature groups - use dynamic_groups if provided, otherwise from experiment
        if dynamic_groups is not None:
            feature_groups = dynamic_groups
            logger.info(f"Using dynamic feature groups: {feature_groups}")
        else:
            experiment_config = self.config.get('experiments', {}).get(experiment, {})
            feature_groups = experiment_config.get('activations', [])

            if not feature_groups:
                logger.warning(f"No feature groups found for experiment {experiment}, using default")
                feature_groups = self.config.get('activations', {}).get('default', [])

        logger.info(f"Applying feature groups: {feature_groups}")

        # Apply each feature group
        for group_name in feature_groups:
            if group_name in self.config.get('feature_groups', {}):
                group_config = self.config['feature_groups'][group_name]
                features_df = self._apply_feature_group(features_df, group_config)
            else:
                logger.warning(f"Feature group '{group_name}' not found in config")

        # Clean up features
        features_df = self._clean_features(features_df)

        logger.info(f"Feature engineering complete for {symbol}: {features_df.shape[1]} features")
        return features_df

    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean up features and add target variable 'y'.

        Args:
            df: Features DataFrame

        Returns:
            Cleaned DataFrame with target variable 'y'
        """
        original_rows = len(df)
        
        # Add target variable 'y' - future 1-day return
        df['y'] = df['close'].pct_change(1).shift(-1)  # Next day's return

        # Remove rows with NaN values
        df = df.dropna()
        dropped_rows = original_rows - len(df)
        
        if dropped_rows > 0:
            logger.warning(f"Dropped {dropped_rows} rows with NaN values ({dropped_rows/original_rows*100:.1f}% of data)")
            if dropped_rows / original_rows > 0.5:
                logger.error(f"Excessive data loss: {dropped_rows/original_rows*100:.1f}% of rows dropped")

        # Ensure we have a clean index
        df = df.sort_index()

        return df

    def _apply_feature_group(self, df: pd.DataFrame, group_config: Dict[str, Any]) -> pd.DataFrame:
        """Apply a feature group to the dataframe."""
        features_config = group_config.get('features', {})

        for feature_name, feature_config in features_config.items():
            feature_type = feature_config.get('type')
            params = feature_config.get('params', {})

            try:
                if feature_type == 'returns':
                    df = self._add_returns(df, **params)
                elif feature_type == 'rolling_std':
                    df = self._add_rolling_std(df, **params)
                elif feature_type == 'rolling_mean':
                    df = self._add_rolling_mean(df, **params)
                elif feature_type == 'rolling_max':
                    df = self._add_rolling_max(df, **params)
                elif feature_type == 'rolling_min':
                    df = self._add_rolling_min(df, **params)
                elif feature_type == 'sma':
                    df = self._add_sma(df, **params)
                elif feature_type == 'ema':
                    df = self._add_ema(df, **params)
                elif feature_type == 'rsi':
                    df = self._add_rsi(df, **params)
                elif feature_type == 'macd':
                    df = self._add_macd(df, **params)
                elif feature_type == 'bollinger_bands':
                    df = self._add_bollinger_bands(df, **params)
                elif feature_type == 'day_of_week':
                    df = self._add_day_of_week(df)
                elif feature_type == 'month_of_year':
                    df = self._add_month_of_year(df)
                elif feature_type == 'quarter':
                    df = self._add_quarter(df)
                elif feature_type == 'is_month_end':
                    df = self._add_is_month_end(df)
                elif feature_type == 'is_month_start':
                    df = self._add_is_month_start(df)
                elif feature_type == 'is_quarter_end':
                    df = self._add_is_quarter_end(df)
                elif feature_type == 'is_quarter_start':
                    df = self._add_is_quarter_start(df)
                elif feature_type == 'day_of_month':
                    df = self._add_day_of_month(df)
                elif feature_type == 'week_of_year':
                    df = self._add_week_of_year(df)
                elif feature_type == 'macro_series':
                    df = self._add_macro_series(df, **params)
                elif feature_type == 'macro_series_change':
                    df = self._add_macro_series_change(df, **params)
                elif feature_type == 'macro_trend':
                    df = self._add_macro_trend(df, **params)
                elif feature_type == 'commodity_series':
                    df = self._add_commodity_series(df, **params)
                elif feature_type == 'commodity_returns':
                    df = self._add_commodity_returns(df, **params)
                elif feature_type == 'commodity_trend':
                    df = self._add_commodity_trend(df, **params)
                elif feature_type == 'commodity_spike':
                    df = self._add_commodity_spike(df, **params)
                elif feature_type == 'rate_sensitive':
                    df = self._add_rate_sensitive(df, **params)
                elif feature_type == 'defensive_correlation':
                    df = self._add_defensive_correlation(df, **params)
                elif feature_type == 'growth_indicators':
                    df = self._add_growth_indicators(df, **params)
                elif feature_type == 'risk_correlation':
                    df = self._add_risk_correlation(df, **params)
                elif feature_type == 'sector_correlation':
                    df = self._add_sector_correlation(df, **params)
                elif feature_type == 'commodity_correlation':
                    df = self._add_commodity_correlation(df, **params)
                elif feature_type == 'sector_short':
                    df = self._add_sector_short(df, **params)
                elif feature_type == 'deflation_indicators':
                    df = self._add_deflation_indicators(df, **params)
                elif feature_type == 'safe_haven_correlation':
                    df = self._add_safe_haven_correlation(df, **params)
                elif feature_type == 'inflation_hedges':
                    df = self._add_inflation_hedges(df, **params)
                elif feature_type == 'risk_on_indicators':
                    df = self._add_risk_on_indicators(df, **params)
                elif feature_type == 'growth_signals':
                    df = self._add_growth_signals(df, **params)
                elif feature_type == 'seasonal_pattern':
                    df = self._add_seasonal_pattern(df, **params)
                elif feature_type == 'tax_harvest_signals':
                    df = self._add_tax_harvest_signals(df, **params)
                elif feature_type == 'seasonal_rally':
                    df = self._add_seasonal_rally(df, **params)
                elif feature_type == 'earnings_indicators':
                    df = self._add_earnings_indicators(df, **params)
                elif feature_type == 'seasonal_doldrums':
                    df = self._add_seasonal_doldrums(df, **params)
                elif feature_type == 'seasonal_consumer':
                    df = self._add_seasonal_consumer(df, **params)
                elif feature_type == 'year_end_signals':
                    df = self._add_year_end_signals(df, **params)
                else:
                    logger.warning(f"Unknown feature type: {feature_type}")

            except Exception as e:
                logger.error(f"Failed to add feature {feature_name}: {e}")

        return df

    # Price-based features
    def _add_returns(self, df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        """Add return features."""
        if periods is None:
            periods = [1, 5, 10, 20]

        for period in periods:
            df[f'returns_{period}d'] = df['close'].pct_change(period)

        return df

    def _add_rolling_std(self, df: pd.DataFrame, window: List[int] = None) -> pd.DataFrame:
        """Add rolling standard deviation (volatility) features."""
        if window is None:
            window = [5, 10, 20, 30]

        for w in window:
            df[f'volatility_{w}d'] = df['close'].pct_change().rolling(w).std()

        return df

    def _add_rolling_mean(self, df: pd.DataFrame, window: List[int] = None) -> pd.DataFrame:
        """Add rolling mean features."""
        if window is None:
            window = [5, 10, 20, 30, 50]

        for w in window:
            df[f'price_mean_{w}d'] = df['close'].rolling(w).mean()

        return df

    def _add_rolling_max(self, df: pd.DataFrame, window: List[int] = None) -> pd.DataFrame:
        """Add rolling max features."""
        if window is None:
            window = [5, 10, 20]

        for w in window:
            df[f'price_max_{w}d'] = df['close'].rolling(w).max()

        return df

    def _add_rolling_min(self, df: pd.DataFrame, window: List[int] = None) -> pd.DataFrame:
        """Add rolling min features."""
        if window is None:
            window = [5, 10, 20]

        for w in window:
            df[f'price_min_{w}d'] = df['close'].rolling(w).min()

        return df

    # Technical indicators
    def _add_sma(self, df: pd.DataFrame, window: List[int] = None) -> pd.DataFrame:
        """Add Simple Moving Average features."""
        if window is None:
            window = [5, 10, 20, 50]

        for w in window:
            df[f'sma_{w}'] = df['close'].rolling(w).mean()

        return df

    def _add_ema(self, df: pd.DataFrame, window: List[int] = None) -> pd.DataFrame:
        """Add Exponential Moving Average features."""
        if window is None:
            window = [5, 10, 20, 50]

        for w in window:
            df[f'ema_{w}'] = df['close'].ewm(span=w).mean()

        return df

    def _add_rsi(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Add RSI indicator."""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / loss
        df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        return df

    def _add_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Add MACD indicator."""
        ema_fast = df['close'].ewm(span=fast).mean()
        ema_slow = df['close'].ewm(span=slow).mean()
        df[f'macd_{fast}_{slow}'] = ema_fast - ema_slow
        df[f'macd_signal_{fast}_{slow}_{signal}'] = df[f'macd_{fast}_{slow}'].ewm(span=signal).mean()
        df[f'macd_hist_{fast}_{slow}_{signal}'] = df[f'macd_{fast}_{slow}'] - df[f'macd_signal_{fast}_{slow}_{signal}']
        return df

    def _add_bollinger_bands(self, df: pd.DataFrame, window: int = 20, num_std: int = 2) -> pd.DataFrame:
        """Add Bollinger Bands."""
        sma = df['close'].rolling(window).mean()
        std = df['close'].rolling(window).std()
        df[f'bb_upper_{window}_{num_std}'] = sma + (std * num_std)
        df[f'bb_lower_{window}_{num_std}'] = sma - (std * num_std)
        df[f'bb_middle_{window}'] = sma
        df[f'bb_width_{window}_{num_std}'] = (df[f'bb_upper_{window}_{num_std}'] - df[f'bb_lower_{window}_{num_std}']) / df[f'bb_middle_{window}']
        df[f'bb_position_{window}_{num_std}'] = (df['close'] - df[f'bb_lower_{window}_{num_std}']) / (df[f'bb_upper_{window}_{num_std}'] - df[f'bb_lower_{window}_{num_std}'])
        return df

    # Calendar features
    def _add_day_of_week(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add day of week features."""
        df['day_of_week'] = df.index.dayofweek
        for i in range(7):
            df[f'dow_{i}'] = (df['day_of_week'] == i).astype(int)
        return df

    def _add_month_of_year(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add month of year features."""
        df['month_of_year'] = df.index.month
        for i in range(1, 13):
            df[f'month_{i}'] = (df['month_of_year'] == i).astype(int)
        return df

    def _add_quarter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add quarter features."""
        df['quarter'] = df.index.quarter
        for i in range(1, 5):
            df[f'quarter_{i}'] = (df['quarter'] == i).astype(int)
        return df

    def _add_is_month_end(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add month end indicator."""
        df['is_month_end'] = df.index.is_month_end.astype(int)
        return df

    def _add_is_month_start(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add month start indicator."""
        df['is_month_start'] = df.index.is_month_start.astype(int)
        return df

    def _add_is_quarter_end(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add quarter end indicator."""
        df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        return df

    def _add_is_quarter_start(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add quarter start indicator."""
        df['is_quarter_start'] = df.index.is_quarter_start.astype(int)
        return df

    def _add_day_of_month(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add day of month."""
        df['day_of_month'] = df.index.day
        return df

    def _add_week_of_year(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add week of year."""
        df['week_of_year'] = df.index.isocalendar().week
        return df

    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare features for modeling."""
        # Remove rows with all NaN values
        df = df.dropna(how='all')

        # Forward fill remaining NaN values (for indicators that need lookback)
        df = df.fillna(method='ffill')

        # Remove any remaining NaN rows
        df = df.dropna()

        # Ensure we have a clean index
        df = df.sort_index()

        return df

    def run_feature_engineering(self, symbols: Optional[List[str]] = None,
                              experiment: str = "baseline") -> Dict[str, pd.DataFrame]:
        """
        Run feature engineering for all symbols.

        Args:
            symbols: List of symbols to process (None for all)
            experiment: Feature experiment to use

        Returns:
            Dictionary of symbol -> feature DataFrame
        """
        logger.info(f"Starting feature engineering for experiment: {experiment}")

        # Get symbols to process
        if symbols is None:
            # Load all available raw data
            raw_data = self.data_loader.load_alpha_vantage_data()
            symbols = list(raw_data.keys())

        logger.info(f"Processing {len(symbols)} symbols")

        # Process each symbol
        feature_data = {}
        successful = 0
        failed = 0

        for symbol in symbols:
            try:
                # Load raw data for symbol
                raw_data = self.data_loader.load_alpha_vantage_data(symbols=[symbol])
                if symbol not in raw_data or raw_data[symbol].empty:
                    logger.warning(f"No raw data found for {symbol}")
                    failed += 1
                    continue

                # Engineer features
                features_df = self.engineer_features_for_symbol(
                    symbol, raw_data[symbol], experiment
                )

                if not features_df.empty:
                    feature_data[symbol] = features_df
                    successful += 1

                    # Save to disk
                    self._save_features(symbol, features_df, experiment)

                    logger.info(f"[SUCCESS] {symbol}: {features_df.shape[1]} features, {len(features_df)} rows")
                else:
                    logger.warning(f"No features generated for {symbol}")
                    failed += 1

            except Exception as e:
                logger.error(f"[FAILED] {symbol}: Failed to engineer features - {e}")
                failed += 1

        logger.info("\nFeature Engineering Summary:")
        logger.info("=" * 80)
        logger.info(f"Total symbols processed: {len(symbols)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success rate: {successful/len(symbols)*100:.1f}%")

        return feature_data

    # Macro and commodity features
    def _add_macro_series(self, df: pd.DataFrame, series_code: str,
                         transformations: Optional[List[str]] = None) -> pd.DataFrame:
        """Add macroeconomic series features."""
        if transformations is None:
            transformations = ["level"]

        # Import here to avoid circular imports
        from agents.macro_data_agent import MacroDataAgent

        agent = MacroDataAgent()
        macro_data = agent.load_macro_series(series_code)

        if macro_data.empty:
            logger.warning(f"No macro data available for {series_code}")
            return df

        # Align macro data with stock data dates
        macro_aligned = macro_data.reindex(df.index, method='ffill')

        for transform in transformations:
            if transform == "level":
                df[f'macro_{series_code.replace("/", "_").replace(".", "_")}_level'] = macro_aligned[series_code]
            elif transform.startswith("rolling_mean_"):
                window = int(transform.split("_")[-1])
                df[f'macro_{series_code.replace("/", "_").replace(".", "_")}_mean_{window}d'] = (
                    macro_aligned[series_code].rolling(window).mean()
                )
            elif transform.startswith("rolling_std_"):
                window = int(transform.split("_")[-1])
                df[f'macro_{series_code.replace("/", "_").replace(".", "_")}_std_{window}d'] = (
                    macro_aligned[series_code].rolling(window).std()
                )

        return df

    def _add_macro_series_change(self, df: pd.DataFrame, series_code: str,
                                periods: Optional[List[int]] = None) -> pd.DataFrame:
        """Add macroeconomic series change features."""
        if periods is None:
            periods = [1, 5]

        # Import here to avoid circular imports
        from agents.macro_data_agent import MacroDataAgent

        agent = MacroDataAgent()
        macro_data = agent.load_macro_series(series_code)

        if macro_data.empty:
            logger.warning(f"No macro data available for {series_code}")
            return df

        # Align macro data with stock data dates
        macro_aligned = macro_data.reindex(df.index, method='ffill')

        for period in periods:
            change_col = f'macro_{series_code.replace("/", "_").replace(".", "_")}_change_{period}d'
            df[change_col] = macro_aligned[series_code].diff(period)

        return df

    def _add_macro_trend(self, df: pd.DataFrame, series_code: str, window: int = 30) -> pd.DataFrame:
        """Add macroeconomic trend features."""
        # Import here to avoid circular imports
        from agents.macro_data_agent import MacroDataAgent

        agent = MacroDataAgent()
        macro_data = agent.load_macro_series(series_code)

        if macro_data.empty:
            logger.warning(f"No macro data available for {series_code}")
            return df

        # Align macro data with stock data dates
        macro_aligned = macro_data.reindex(df.index, method='ffill')

        # Calculate trend as slope of linear regression over window
        trend_col = f'macro_{series_code.replace("/", "_").replace(".", "_")}_trend_{window}d'

        def calculate_trend(series_window):
            if len(series_window) < window:
                return np.nan
            x = np.arange(len(series_window))
            y = series_window.values
            if len(y) == len(x) and len(y) > 1:
                slope = np.polyfit(x, y, 1)[0]
                return slope
            return np.nan

        df[trend_col] = macro_aligned[series_code].rolling(window).apply(calculate_trend, raw=False)

        return df

    def _add_commodity_series(self, df: pd.DataFrame, commodity_code: str,
                             transformations: Optional[List[str]] = None) -> pd.DataFrame:
        """Add commodity series features."""
        if transformations is None:
            transformations = ["returns"]

        # Import here to avoid circular imports
        from agents.commodity_data_agent import CommodityDataAgent

        agent = CommodityDataAgent()
        commodity_data = agent.load_commodity_series(commodity_code)

        if commodity_data.empty:
            logger.warning(f"No commodity data available for {commodity_code}")
            return df

        # Use close price for commodity
        price_col = f'{commodity_code}_close'
        if price_col not in commodity_data.columns:
            logger.warning(f"Price column {price_col} not found in commodity data")
            return df

        # Align commodity data with stock data dates
        commodity_aligned = commodity_data.reindex(df.index, method='ffill')

        for transform in transformations:
            if transform == "returns":
                df[f'commodity_{commodity_code}_returns_1d'] = commodity_aligned[price_col].pct_change(1)
            elif transform.startswith("rolling_mean_"):
                window = int(transform.split("_")[-1])
                df[f'commodity_{commodity_code}_mean_{window}d'] = (
                    commodity_aligned[price_col].rolling(window).mean()
                )
            elif transform.startswith("rolling_std_"):
                window = int(transform.split("_")[-1])
                df[f'commodity_{commodity_code}_std_{window}d'] = (
                    commodity_aligned[price_col].rolling(window).std()
                )
            elif transform.startswith("volatility_"):
                window = int(transform.split("_")[-1])
                df[f'commodity_{commodity_code}_volatility_{window}d'] = (
                    commodity_aligned[price_col].pct_change().rolling(window).std()
                )

        return df

    def _add_commodity_returns(self, df: pd.DataFrame, commodity_code: str,
                              periods: Optional[List[int]] = None) -> pd.DataFrame:
        """Add commodity return features."""
        if periods is None:
            periods = [1, 5, 10, 20]

        # Import here to avoid circular imports
        from agents.commodity_data_agent import CommodityDataAgent

        agent = CommodityDataAgent()
        commodity_data = agent.load_commodity_series(commodity_code)

        if commodity_data.empty:
            logger.warning(f"No commodity data available for {commodity_code}")
            return df

        # Use close price for commodity
        price_col = f'{commodity_code}_close'
        if price_col not in commodity_data.columns:
            logger.warning(f"Price column {price_col} not found in commodity data")
            return df

        # Align commodity data with stock data dates
        commodity_aligned = commodity_data.reindex(df.index, method='ffill')

        for period in periods:
            df[f'commodity_{commodity_code}_returns_{period}d'] = (
                commodity_aligned[price_col].pct_change(period)
            )

        return df

    def _add_commodity_trend(self, df: pd.DataFrame, commodity_code: str, window: int = 30) -> pd.DataFrame:
        """Add commodity trend features."""
        # Import here to avoid circular imports
        from agents.commodity_data_agent import CommodityDataAgent

        agent = CommodityDataAgent()
        commodity_data = agent.load_commodity_series(commodity_code)

        if commodity_data.empty:
            logger.warning(f"No commodity data available for {commodity_code}")
            return df

        # Use close price for commodity
        price_col = f'{commodity_code}_close'
        if price_col not in commodity_data.columns:
            logger.warning(f"Price column {price_col} not found in commodity data")
            return df

        # Align commodity data with stock data dates
        commodity_aligned = commodity_data.reindex(df.index, method='ffill')

        # Calculate trend as slope of linear regression over window
        trend_col = f'commodity_{commodity_code}_trend_{window}d'

        def calculate_trend(series_window):
            if len(series_window) < window:
                return np.nan
            x = np.arange(len(series_window))
            y = series_window.values
            if len(y) == len(x) and len(y) > 1:
                slope = np.polyfit(x, y, 1)[0]
                return slope
            return np.nan

        df[trend_col] = commodity_aligned[price_col].rolling(window).apply(calculate_trend, raw=False)

        return df

    def _add_commodity_spike(self, df: pd.DataFrame, commodity_code: str, threshold: float = 0.05) -> pd.DataFrame:
        """Add commodity spike detection features."""
        # Import here to avoid circular imports
        from agents.commodity_data_agent import CommodityDataAgent

        agent = CommodityDataAgent()
        commodity_data = agent.load_commodity_series(commodity_code)

        if commodity_data.empty:
            logger.warning(f"No commodity data available for {commodity_code}")
            return df

        # Use close price for commodity
        price_col = f'{commodity_code}_close'
        if price_col not in commodity_data.columns:
            logger.warning(f"Price column {price_col} not found in commodity data")
            return df

        # Align commodity data with stock data dates
        commodity_aligned = commodity_data.reindex(df.index, method='ffill')

        # Calculate daily returns
        daily_returns = commodity_aligned[price_col].pct_change(1)

        # Detect spikes (absolute return > threshold)
        spike_col = f'commodity_{commodity_code}_spike_{threshold:.0%}'
        df[spike_col] = (daily_returns.abs() > threshold).astype(int)

        return df

    def _save_features(self, symbol: str, features_df: pd.DataFrame, experiment: str):
        """Save engineered features to disk."""
        output_dir = Path(f"data/processed/{experiment}")
        output_dir.mkdir(parents=True, exist_ok=True)

        file_path = output_dir / f"{symbol}_features.parquet"
        features_df.to_parquet(file_path)
        logger.debug(f"Saved features for {symbol} to {file_path}")

    # Regime-specific feature methods (placeholders for Phase 2)
    def _add_rate_sensitive(self, df: pd.DataFrame, threshold: float = 0.04) -> pd.DataFrame:
        """Add rate-sensitive features (placeholder)."""
        # Placeholder: Add a simple feature based on volatility during high rates
        df['rate_sensitive_volatility'] = df['close'].rolling(20).std() * (1 if threshold > 0.03 else 0.5)
        return df

    def _add_defensive_correlation(self, df: pd.DataFrame, assets: Optional[List[str]] = None) -> pd.DataFrame:
        """Add defensive asset correlation features (placeholder)."""
        # Placeholder: Add correlation with defensive assets (would need asset data)
        df['defensive_correlation_proxy'] = df['close'].rolling(20).mean() / df['close'].rolling(50).mean()
        return df

    def _add_growth_indicators(self, df: pd.DataFrame, sectors: Optional[List[str]] = None) -> pd.DataFrame:
        """Add growth indicators (placeholder)."""
        # Placeholder: Momentum indicator as proxy for growth
        df['growth_momentum'] = df['close'].pct_change(20)
        return df

    def _add_risk_correlation(self, df: pd.DataFrame, assets: Optional[List[str]] = None) -> pd.DataFrame:
        """Add risk asset correlation features (placeholder)."""
        # Placeholder: Beta-like calculation
        market_returns = df['close'].pct_change(1)  # Using own returns as proxy
        df['risk_correlation'] = market_returns.rolling(20).corr(df['close'].pct_change(1))
        return df

    def _add_sector_correlation(self, df: pd.DataFrame, sector: str = "energy") -> pd.DataFrame:
        """Add sector correlation features (placeholder)."""
        # Placeholder: Sector-specific volatility
        df[f'{sector}_sector_volatility'] = df['close'].rolling(10).std()
        return df

    def _add_commodity_correlation(self, df: pd.DataFrame, commodities: Optional[List[str]] = None) -> pd.DataFrame:
        """Add commodity correlation features (placeholder)."""
        # Placeholder: Commodity price sensitivity
        df['commodity_sensitivity'] = df['close'].pct_change(5).rolling(10).std()
        return df

    def _add_sector_short(self, df: pd.DataFrame, sector: str = "energy") -> pd.DataFrame:
        """Add sector short features (placeholder)."""
        # Placeholder: Inverse sector performance
        df[f'{sector}_short_signal'] = -df['close'].pct_change(1)
        return df

    def _add_deflation_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add deflation indicators (placeholder)."""
        # Placeholder: Deflationary pressure signals
        df['deflation_pressure'] = df['volume'].rolling(20).mean() / df['volume'].rolling(5).mean()
        return df

    def _add_safe_haven_correlation(self, df: pd.DataFrame, assets: Optional[List[str]] = None) -> pd.DataFrame:
        """Add safe haven correlation features (placeholder)."""
        # Placeholder: Safe haven demand indicator
        df['safe_haven_demand'] = df['close'].rolling(20).std() / df['close'].rolling(5).std()
        return df

    def _add_inflation_hedges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add inflation hedge features (placeholder)."""
        # Placeholder: Real return approximation
        df['inflation_hedge_effectiveness'] = df['close'].pct_change(10) - df['close'].pct_change(1)
        return df

    def _add_risk_on_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add risk-on indicators (placeholder)."""
        # Placeholder: Risk appetite measure
        df['risk_appetite'] = df['close'].pct_change(20).rolling(10).mean()
        return df

    def _add_growth_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add growth signals (placeholder)."""
        # Placeholder: Growth trend strength
        df['growth_trend_strength'] = df['close'].rolling(50).mean() / df['close'].rolling(200).mean()
        return df

    def _add_seasonal_pattern(self, df: pd.DataFrame, quarter: int = 1) -> pd.DataFrame:
        """Add seasonal pattern features (placeholder)."""
        # Placeholder: Quarter-specific performance
        df[f'q{quarter}_seasonal_strength'] = df['close'].pct_change(63)  # ~quarter
        return df

    def _add_tax_harvest_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add tax harvest signals (placeholder)."""
        # Placeholder: Tax loss harvesting opportunities
        df['tax_harvest_opportunity'] = (df['close'].pct_change(252) < -0.1).astype(int)  # Annual loss
        return df

    def _add_seasonal_rally(self, df: pd.DataFrame, quarter: int = 2) -> pd.DataFrame:
        """Add seasonal rally features (placeholder)."""
        # Placeholder: Seasonal momentum
        df[f'q{quarter}_rally_momentum'] = df['close'].pct_change(21)  # ~month
        return df

    def _add_earnings_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add earnings season indicators (placeholder)."""
        # Placeholder: Earnings-related volatility
        df['earnings_volatility'] = df['close'].rolling(5).std() / df['close'].rolling(20).std()
        return df

    def _add_seasonal_doldrums(self, df: pd.DataFrame, quarter: int = 3) -> pd.DataFrame:
        """Add seasonal doldrums features (placeholder)."""
        # Placeholder: Summer slowdown indicator
        df[f'q{quarter}_doldrums'] = df['volume'].rolling(20).mean() / df['volume'].rolling(60).mean()
        return df

    def _add_seasonal_consumer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add seasonal consumer features (placeholder)."""
        # Placeholder: Consumer spending patterns
        df['seasonal_consumer_spending'] = df['close'].pct_change(30).rolling(10).mean()
        return df

    def _add_year_end_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add year-end positioning signals (placeholder)."""
        # Placeholder: Year-end portfolio adjustments
        df['year_end_positioning'] = df['close'].pct_change(10).rolling(5).mean()
        return df

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run feature engineering pipeline')
    parser.add_argument('--symbols', nargs='*', help='Specific symbols to process')
    parser.add_argument('--experiment', default='baseline',
                       help='Feature experiment to use (default: baseline)')
    parser.add_argument('--config', default='config/feature_config.yaml',
                       help='Path to feature config file')

    args = parser.parse_args()

    try:
        logger.info("Starting feature engineering pipeline...")

        # Initialize feature engineer
        engineer = FeatureEngineer(args.config)

        # Run feature engineering
        feature_data = engineer.run_feature_engineering(
            symbols=args.symbols,
            experiment=args.experiment
        )

        logger.info("\nFeature engineering completed successfully! [SUCCESS]")

        # Print summary
        if feature_data:
            sample_symbol = list(feature_data.keys())[0]
            sample_features = feature_data[sample_symbol]
            logger.info(f"\nSample features for {sample_symbol}:")
            logger.info(f"Shape: {sample_features.shape}")
            logger.info(f"Feature columns: {len(sample_features.columns)}")
            logger.info("Sample feature names:")
            for i, col in enumerate(sample_features.columns[:10]):
                logger.info(f"  {i+1}. {col}")
            if len(sample_features.columns) > 10:
                logger.info(f"  ... and {len(sample_features.columns) - 10} more")

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()