import os
import pandas as pd
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from ..graphs.state import GraphState
from ..alpha_vantage_client import AlphaVantageClient

logger = logging.getLogger(__name__)

class ThreadSafeAlphaVantageClient(AlphaVantageClient):
    """
    Thread-safe wrapper for AlphaVantageClient to handle rate limiting correctly
    across multiple threads.
    """
    def __init__(self, api_key=None):
        super().__init__(api_key)
        self._lock = threading.Lock()

    def _check_rate_limit(self):
        """Thread-safe rate limit check."""
        with self._lock:
            super()._check_rate_limit()

def _fetch_symbol_data(symbol, client, start_date, end_date, primary_source='alpha_vantage'):
    """
    Helper function to fetch data for a single symbol.
    """
    try:
        # Fetch historical data using Alpha Vantage with retry logic
        data = None
        max_retries = 3

        for attempt in range(max_retries):
            try:
                data = client.get_daily_data(symbol, outputsize='full')
                break  # Success, exit retry loop
            except Exception as e:
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter could be better, but simple backoff is fine
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"All {max_retries} attempts failed for {symbol}: {e}")
                    return symbol, None, f"All attempts failed: {e}"

        if data is not None and not data.empty:
            # Filter to date range
            data = data[(data.index >= start_date) & (data.index <= end_date)]

            # Convert index to string for JSON serialization
            data.index = data.index.astype(str)

            # Validate data structure
            if _validate_data_structure(data):
                return symbol, data, None
            else:
                return symbol, None, "Data validation failed"
        else:
            return symbol, None, f"No data returned from {primary_source.upper()}"

    except Exception as e:
        return symbol, None, f"Error loading data: {e}"

def _load_data_parallel(state: GraphState) -> GraphState:
    """
    Loads data using Alpha Vantage in parallel.
    """
    logger.info("--- Node: Load Data (Parallel Alpha Vantage) ---")

    symbols = state['symbols']
    config = state.get('config', {})
    raw_data = {}
    errors = []

    # Initialize thread-safe client
    try:
        data_ingestion = ThreadSafeAlphaVantageClient()
        # Update rate limit from config if available
        if 'alpha_vantage' in config and 'rate_limit' in config['alpha_vantage']:
            data_ingestion.rate_limit = int(config['alpha_vantage']['rate_limit'])
            logger.info(f"Updated Alpha Vantage rate limit to {data_ingestion.rate_limit} calls/min")
            
        primary_source = 'alpha_vantage'
        logger.info("✅ Thread-safe Alpha Vantage client initialized successfully")
    except ValueError as e:
        error_msg = f"FATAL: Alpha Vantage initialization failed: {e}"
        logger.error(error_msg)
        state['errors'].append(error_msg)
        return state

    # Define the time window for historical data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')

    logger.info(f"Fetching historical data for {len(symbols)} symbols from {start_date} to {end_date}")

    # Determine max workers based on rate limit or config
    # Default to 10 workers to be safe, or use config
    max_workers = 10
    if 'scaling' in config and 'max_workers' in config['scaling']:
        max_workers = int(config['scaling']['max_workers'])
    
    # Cap workers to avoid hitting rate limits too aggressively even with the lock
    # 1200 calls/min = 20 calls/sec. 10-20 workers is reasonable.
    max_workers = min(max_workers, 20)
    
    logger.info(f"Starting parallel download with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(_fetch_symbol_data, symbol, data_ingestion, start_date, end_date, primary_source): symbol 
            for symbol in symbols
        }

        completed_count = 0
        total_symbols = len(symbols)

        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            completed_count += 1
            
            try:
                sym, data, error = future.result()
                
                if data is not None:
                    raw_data[sym] = data
                    if completed_count % 10 == 0: # Log every 10 symbols to reduce noise
                        logger.info(f"[{completed_count}/{total_symbols}] ✅ Loaded {sym} ({len(data)} rows)")
                else:
                    logger.error(f"[{completed_count}/{total_symbols}] ❌ Failed {sym}: {error}")
                    errors.append(f"{sym}: {error}")
            except Exception as e:
                logger.error(f"[{completed_count}/{total_symbols}] ❌ Exception for {symbol}: {e}")
                errors.append(f"{symbol}: {e}")

    state['raw_data'] = raw_data
    if errors:
        state['errors'].extend(errors)
        
    logger.info(f"Parallel data loading complete. Loaded {len(raw_data)}/{len(symbols)} symbols.")
    return state

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

def load_data_node(state: GraphState) -> GraphState:
    """
    Wrapper for the data loading function.
    """
    return _load_data_parallel(state)
