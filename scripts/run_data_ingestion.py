#!/usr/bin/env python3
"""
Individual Component Runner: Data Ingestion

This script runs the data ingestion component independently with proper error handling and logging.
"""

import os
import sys
import yaml
import logging
import argparse
from datetime import datetime, timedelta
import pandas as pd

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graphs.state import GraphState
from src.data.unified_ingestion_v2 import UnifiedDataIngestion

def setup_logging(log_level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/data_ingestion.log'),
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

def load_symbols_from_csv(csv_path="watchlist_ibkr.csv"):
    """Load symbols from IBKR watchlist CSV file"""
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

def run_data_ingestion(symbols=None, start_date=None, end_date=None, config=None, skip_sentiment=False):
    """
    Run data ingestion independently

    Args:
        symbols: List of symbols to process. If None, loads from CSV
        start_date: Start date for data (YYYY-MM-DD). If None, uses 2 years ago
        end_date: End date for data (YYYY-MM-DD). If None, uses today
        config: Configuration dict. If None, loads from config.yaml
        skip_sentiment: If True, skip sentiment data fetching to avoid API timeouts

    Returns:
        GraphState with raw_data populated
    """
    logger = setup_logging()

    if config is None:
        config = load_config()

    if symbols is None:
        symbols = load_symbols_from_csv()

    if not symbols:
        logger.error("No symbols provided or found in CSV")
        return None

    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')

    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    logger.info(f"Starting data ingestion for {len(symbols)} symbols")
    logger.info(f"Date range: {start_date} to {end_date}")

    # Initialize state
    state = GraphState(
        symbols=symbols,
        config=config,
        raw_data={},
        features={},
        forecasts={},
        performance_summary=pd.DataFrame(),
        drift_metrics=pd.DataFrame(),
        risk_kpis=pd.DataFrame(),
        anomalies={},
        recommended_actions=[],
        executed_actions=[],
        retrained_models={},
        best_models={},
        errors=[],
        hpo_results={},
        shap_results={},
        analytics_summary=pd.DataFrame(),
        hpo_decision={},
        retraining_history=[],
        guardrail_log=[],
        hpo_triggered=False,
        drift_detected=False,
        edge_index=None,
        node_features=None,
        symbol_to_idx={}
    )

    # Extract IBKR configuration
    ibkr_config = config.get('ibkr', {})
    market_data_type = ibkr_config.get('market_data_type', 3)

    # Initialize data ingestion
    data_ingestion = UnifiedDataIngestion(
        use_real_data=True,
        market_data_type=market_data_type,
        config=config,
        skip_sentiment=skip_sentiment
    )

    try:
        logger.info("Initializing data ingestion system...")
        data_ingestion.initialize()
        primary_source = data_ingestion.primary_source
        logger.info(f"Data ingestion initialized. Primary source: {primary_source.upper()}")

        for symbol in symbols:
            try:
                logger.info(f"Fetching data for {symbol}")
                data = data_ingestion.get_historical_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe='1 day'
                )

                if data is not None and not data.empty:
                    state['raw_data'][symbol] = data
                    logger.info(f"Loaded {len(data)} rows for {symbol}")
                else:
                    error_msg = f"No data returned for {symbol}"
                    state['errors'].append(error_msg)
                    logger.error(error_msg)

            except Exception as e:
                error_msg = f"Error loading data for {symbol}: {e}"
                state['errors'].append(error_msg)
                logger.error(error_msg)

    except RuntimeError as e:
        error_msg = f"FATAL: Could not initialize data source. {e}"
        logger.error(error_msg)
        state['errors'].append(error_msg)

        # Fallback to synthetic data
        logger.info("Falling back to synthetic data generation...")
        data_ingestion.primary_source = 'synthetic'

        for symbol in symbols:
            try:
                logger.info(f"Generating synthetic data for {symbol}")
                data = data_ingestion.get_historical_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe='1 day'
                )

                if data is not None and not data.empty:
                    state['raw_data'][symbol] = data
                    logger.info(f"Generated {len(data)} rows for {symbol}")
                else:
                    error_msg = f"No synthetic data generated for {symbol}"
                    state['errors'].append(error_msg)
                    logger.error(error_msg)

            except Exception as e:
                error_msg = f"Error generating synthetic data for {symbol}: {e}"
                state['errors'].append(error_msg)
                logger.error(error_msg)

    finally:
        try:
            data_ingestion.disconnect()
            logger.info("Data ingestion connections cleaned up")
        except Exception as e:
            logger.warning(f"Error during connection cleanup: {e}")

    # Summary
    successful_symbols = len(state['raw_data'])
    failed_symbols = len(state['errors'])

    logger.info(f"Data ingestion completed. {successful_symbols} successful, {failed_symbols} failed")

    if state['errors']:
        logger.warning("Errors encountered:")
        for error in state['errors']:
            logger.warning(f"  - {error}")

    return state

def main():
    parser = argparse.ArgumentParser(description='Run data ingestion component')
    parser.add_argument('--symbols', nargs='+', help='Symbols to process (default: load from CSV)')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD, default: 2 years ago)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD, default: today)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    parser.add_argument('--output', help='Output file to save raw data (optional)')
    parser.add_argument('--skip-sentiment', action='store_true', help='Skip sentiment data fetching to avoid API timeouts')

    args = parser.parse_args()    # Setup logging
    log_level = getattr(logging, args.log_level)
    logger = setup_logging(log_level)

    # Load config
    config = load_config()

    try:
        # Run data ingestion
        state = run_data_ingestion(
            symbols=args.symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            config=config,
            skip_sentiment=args.skip_sentiment
        )

        if state is None:
            sys.exit(1)

        # Save output if requested
        if args.output:
            import pickle
            with open(args.output, 'wb') as f:
                pickle.dump(state['raw_data'], f)
            logger.info(f"Raw data saved to {args.output}")

        # Print summary
        print(f"\nData Ingestion Summary:")
        print(f"Symbols processed: {len(state['symbols'])}")
        print(f"Data loaded: {len(state['raw_data'])}")
        print(f"Errors: {len(state['errors'])}")

        if state['errors']:
            print("\nErrors:")
            for error in state['errors']:
                print(f"  - {error}")

    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
