import os
import pandas as pd
import logging
from datetime import datetime, timedelta
from ..graphs.state import GraphState
from ..data.unified_ingestion_v2 import UnifiedDataIngestion

logger = logging.getLogger(__name__)

def load_data_node(state: GraphState) -> GraphState:
    """
    Loads data using the UnifiedDataIngestion interface.
    It respects the data source hierarchy defined in the configuration.
    """
    logger.info("--- Node: Load Data ---")

    symbols = state['symbols']
    config = state.get('config', {})
    raw_data = {}

    # Extract IBKR configuration for market data type, if needed
    ibkr_config = config.get('ibkr', {})
    market_data_type = ibkr_config.get('market_data_type', 3)

    # Initialize data ingestion
    data_ingestion = UnifiedDataIngestion(
        use_real_data=True, # Always try to use real data as per config
        market_data_type=market_data_type,
        config=config,
        skip_sentiment=True  # Skip news sentiment to avoid rate limiting
    )

    try:
        logger.info("Initializing data ingestion system...")
        data_ingestion.initialize()
        primary_source = data_ingestion.primary_source
        logger.info(f"✅ Data ingestion initialized. Primary source: {primary_source.upper()}")
        
        # Define the time window for historical data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d') # Increased to 2 years for better feature calculation

        logger.info(f"Fetching historical data for {len(symbols)} symbols from {start_date} to {end_date}")

        for symbol in symbols:
            try:
                # Fetch historical data with retry logic
                data = _fetch_with_retry(data_ingestion, symbol, start_date, end_date)
                if data is not None and not data.empty:
                    # Validate data structure
                    if _validate_data_structure(data):
                        raw_data[symbol] = data
                        logger.info(f"✅ Loaded data for {symbol} from {primary_source.upper()} ({len(data)} rows).")
                    else:
                        error_msg = f"Data validation failed for symbol {symbol}."
                        state['errors'].append(error_msg)
                        logger.error(error_msg)
                else:
                    error_msg = f"No data returned for symbol {symbol} from {primary_source.upper()}."
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
        # Fall back to synthetic data generation
        logger.info("🔄 Falling back to synthetic data generation...")
        
        # Force synthetic mode
        data_ingestion.primary_source = 'synthetic'
        
        # Define the time window for historical data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')

        logger.info(f"Generating synthetic data for {len(symbols)} symbols from {start_date} to {end_date}")

        for symbol in symbols:
            try:
                # Generate synthetic data
                data = data_ingestion.get_historical_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe='1 day'
                )
                if data is not None and not data.empty:
                    raw_data[symbol] = data
                    logger.info(f"✅ Generated synthetic data for {symbol} ({len(data)} rows).")
                else:
                    error_msg = f"No synthetic data generated for symbol {symbol}."
                    state['errors'].append(error_msg)
                    logger.error(error_msg)
            except Exception as e:
                error_msg = f"Error generating synthetic data for {symbol}: {e}"
                state['errors'].append(error_msg)
                logger.error(error_msg)

    finally:
        # Ensure proper cleanup of any open connections
        try:
            data_ingestion.disconnect()
        except Exception as e:
            logger.warning(f"Warning: Error during connection cleanup: {e}")

    state['raw_data'] = raw_data
    return state

def _fetch_with_retry(data_ingestion, symbol, start_date, end_date, max_retries=3):
    """Fetch data with retry logic for transient failures."""
    for attempt in range(max_retries):
        try:
            return data_ingestion.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe='1 day'
            )
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {e}. Retrying...")
                import time
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"All {max_retries} attempts failed for {symbol}: {e}")
                raise

def _validate_data_structure(data):
    """Validate that the data has required columns and structure."""
    required_columns = ['close', 'high', 'low', 'open', 'volume']
    if not isinstance(data, pd.DataFrame):
        logger.error("Data is not a pandas DataFrame")
        return False
    if data.empty:
        logger.error("Data is empty")
        return False
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    return True
