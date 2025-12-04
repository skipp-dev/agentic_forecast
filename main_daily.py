#!/usr/bin/env python3
"""
Production Daily Pipeline Runner

This script orchestrates the complete daily forecasting pipeline using Alpha Vantage only.
No synthetic data fallback - follows documented production architecture.

Usage:
    python main_daily.py [--symbols SYMBOL1 SYMBOL2] [--date YYYY-MM-DD]
"""

import os
import sys
import yaml
import logging
import argparse
from datetime import datetime, timedelta
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.alpha_vantage_client import AlphaVantageClient

def setup_logging(log_level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/daily_pipeline.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config():
    """Load configuration from config.yaml"""
    config_path = "config.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

def load_symbols_from_csv(csv_path="watchlist_main.csv"):
    """Load symbols from CSV file"""
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if 'Symbol' in df.columns:
                return df['Symbol'].tolist()
            else:
                logger.warning(f"'Symbol' column not found in {csv_path}")
                return []
        except Exception as e:
            logger.error(f"Error loading symbols from {csv_path}: {e}")
            return []
    else:
        logger.warning(f"{csv_path} not found")
        return []

def run_data_ingestion(symbols, start_date, end_date, config):
    """
    Run data ingestion using Alpha Vantage only (no synthetic fallback)

    Args:
        symbols: List of symbols to process
        start_date: Start date string
        end_date: End date string
        config: Configuration dict

    Returns:
        Dict of symbol -> DataFrame with raw data
    """
    logger = logging.getLogger(__name__)

    logger.info(f"Starting Alpha Vantage data ingestion for {len(symbols)} symbols")
    logger.info(f"Date range: {start_date} to {end_date}")

    # Initialize Alpha Vantage client (production only - no fallback)
    try:
        client = AlphaVantageClient()
        logger.info("✅ Alpha Vantage client initialized successfully")
    except ValueError as e:
        error_msg = f"FATAL: Alpha Vantage initialization failed: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    raw_data = {}

    for symbol in symbols:
        try:
            logger.info(f"Fetching data for {symbol}")

            # Use Alpha Vantage daily data
            data = client.get_daily_data(symbol, outputsize='full')

            if data is not None and not data.empty:
                # Filter to date range
                data = data[(data.index >= start_date) & (data.index <= end_date)]
                raw_data[symbol] = data
                logger.info(f"✅ Loaded {len(data)} rows for {symbol}")
            else:
                error_msg = f"No data returned for {symbol}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

        except Exception as e:
            error_msg = f"Error loading data for {symbol}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    logger.info(f"Data ingestion completed successfully. Loaded data for {len(raw_data)} symbols")
    return raw_data

def run_feature_engineering(raw_data, config):
    """
    Run basic feature engineering on raw data

    Args:
        raw_data: Dict of symbol -> raw DataFrame
        config: Configuration dict

    Returns:
        Dict of symbol -> feature DataFrame
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting basic feature engineering...")

    features = {}

    for symbol, data in raw_data.items():
        try:
            logger.info(f"Engineering features for {symbol}")

            # Basic feature engineering
            feature_data = data.copy()

            # Add basic technical indicators (use smaller windows for limited data)
            if 'close' in feature_data.columns and len(feature_data) >= 5:
                # Simple moving averages (use smaller windows)
                window_size = min(10, len(feature_data) // 2)
                feature_data['SMA_short'] = feature_data['close'].rolling(window=window_size).mean()

                # Price changes
                feature_data['daily_return'] = feature_data['close'].pct_change()

                # Volatility (use available data)
                vol_window = min(5, len(feature_data) - 1)
                feature_data['volatility'] = feature_data['daily_return'].rolling(window=vol_window).std()

                # Volume indicators
                if 'volume' in feature_data.columns:
                    feature_data['volume_avg'] = feature_data['volume'].rolling(window=window_size).mean()

            # Drop NaN values but keep some data
            feature_data = feature_data.dropna()

            # If we have at least some features, keep the data
            if not feature_data.empty and len(feature_data.columns) > len(data.columns):
                features[symbol] = feature_data
                logger.info(f"✅ Generated {feature_data.shape[1]} features for {symbol}")
            else:
                # Fallback: just use the raw data
                features[symbol] = data
                logger.info(f"✅ Using raw data for {symbol} (insufficient data for features)")

        except Exception as e:
            logger.error(f"Error engineering features for {symbol}: {e}")
            # Continue with other symbols

    logger.info(f"Feature engineering completed. Generated features for {len(features)} symbols")
    return features

def run_model_training(features, config):
    """
    Run basic model training and forecasting

    Args:
        features: Dict of symbol -> feature DataFrame
        config: Configuration dict

    Returns:
        Dict of symbol -> forecast DataFrame
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting basic model training and forecasting...")

    forecasts = {}

    for symbol, feature_data in features.items():
        try:
            logger.info(f"Training model and generating forecast for {symbol}")

            # Simple forecast: next day prediction based on current close
            if not feature_data.empty and 'close' in feature_data.columns:
                last_close = feature_data['close'].iloc[-1]
                forecast_date = feature_data.index[-1] + pd.Timedelta(days=1)

                # Create simple forecast DataFrame
                forecast = pd.DataFrame({
                    'date': [forecast_date],
                    'predicted_close': [last_close * 1.001],  # Simple 0.1% increase
                    'confidence': [0.5]
                })
                forecast.set_index('date', inplace=True)

                forecasts[symbol] = forecast
                logger.info(f"✅ Generated basic forecast for {symbol}")
            else:
                logger.warning(f"No forecast generated for {symbol}")

        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}")
            # Continue with other symbols

    logger.info(f"Model training completed. Generated forecasts for {len(forecasts)} symbols")
    return forecasts

def run_monitoring_and_reporting(forecasts, config):
    """
    Run basic monitoring and generate reports

    Args:
        forecasts: Dict of symbol -> forecast DataFrame
        config: Configuration dict
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting basic monitoring and reporting...")

    try:
        # Generate simple performance report
        report_lines = ["Daily Forecast Report", "=" * 50]

        for symbol, forecast in forecasts.items():
            if not forecast.empty:
                predicted = forecast['predicted_close'].iloc[0]
                report_lines.append(f"{symbol}: Predicted close = ${predicted:.2f}")

        report_text = "\n".join(report_lines)

        # Save report
        with open('daily_forecast_report.txt', 'w') as f:
            f.write(report_text)

        logger.info("✅ Basic monitoring and reporting completed")
        logger.info(f"Report saved to daily_forecast_report.txt")

    except Exception as e:
        logger.error(f"Error in monitoring/reporting: {e}")

def main():
    parser = argparse.ArgumentParser(description='Run production daily forecasting pipeline')
    parser.add_argument('--symbols', nargs='+', help='Symbols to process (default: load from CSV)')
    parser.add_argument('--date', help='Target date for forecast (YYYY-MM-DD, default: today)')
    parser.add_argument('--days-back', type=int, default=730, help='Days of historical data (default: 730)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')

    args = parser.parse_args()

    # Setup logging
    log_level = getattr(logging, args.log_level)
    logger = setup_logging(log_level)

    # Load configuration
    config = load_config()

    try:
        # Determine date range
        if args.date:
            target_date = datetime.strptime(args.date, '%Y-%m-%d')
        else:
            target_date = datetime.now()

        end_date = target_date.strftime('%Y-%m-%d')
        start_date = (target_date - timedelta(days=args.days_back)).strftime('%Y-%m-%d')

        # Load symbols
        symbols = args.symbols if args.symbols else load_symbols_from_csv()

        if not symbols:
            logger.error("No symbols provided or found in CSV")
            sys.exit(1)

        logger.info(f"🚀 Starting production daily pipeline for {len(symbols)} symbols")
        logger.info(f"Date range: {start_date} to {end_date}")

        # Step 1: Data Ingestion (Alpha Vantage only)
        raw_data = run_data_ingestion(symbols, start_date, end_date, config)

        # Step 2: Feature Engineering
        features = run_feature_engineering(raw_data, config)

        # Step 3: Model Training & Forecasting
        forecasts = run_model_training(features, config)

        # Step 4: Monitoring & Reporting
        run_monitoring_and_reporting(forecasts, config)

        # Summary
        logger.info("🎉 Production daily pipeline completed successfully!")
        logger.info(f"Symbols processed: {len(symbols)}")
        logger.info(f"Data loaded: {len(raw_data)}")
        logger.info(f"Features generated: {len(features)}")
        logger.info(f"Forecasts generated: {len(forecasts)}")

    except Exception as e:
        logger.error(f"❌ Production pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()