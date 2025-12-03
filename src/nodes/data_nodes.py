import os
import pandas as pd
import logging
import asyncio
import time
from datetime import datetime, timedelta
from ..graphs.state import GraphState
from ..alpha_vantage_client import AlphaVantageClient

logger = logging.getLogger(__name__)

def _load_data_sync(state: GraphState) -> GraphState:
    """
    Loads data using Alpha Vantage only (production-compliant - no synthetic fallback).
    Follows documented architecture: Alpha Vantage only, no IBKR required.
    """
    logger.info("--- Node: Load Data (Alpha Vantage Only) ---")

    symbols = state['symbols']
    config = state.get('config', {})
    raw_data = {}

    # PRODUCTION RULE: Alpha Vantage only - no synthetic fallback for production runs
    try:
        data_ingestion = AlphaVantageClient()
        # Update rate limit from config if available
        if 'alpha_vantage' in config and 'rate_limit' in config['alpha_vantage']:
            data_ingestion.rate_limit = int(config['alpha_vantage']['rate_limit'])
            logger.info(f"Updated Alpha Vantage rate limit to {data_ingestion.rate_limit} calls/min")
            
        primary_source = 'alpha_vantage'
        logger.info("✅ Alpha Vantage client initialized successfully")
    except ValueError as e:
        error_msg = f"FATAL: Alpha Vantage initialization failed: {e}"
        logger.error(error_msg)
        state['errors'].append(error_msg)
        return state  # Fail hard - no synthetic fallback in production

    # Define the time window for historical data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d') # 2 years for better feature calculation

    # Initialize FMP components if enabled
    fmp_enabled = False
    fundamentals_agent = None
    cross_asset_features = None
    
    if config.get('fmp', {}).get('enabled', False) and os.getenv('FMP_API_KEY'):
        try:
            from ..agents.fundamentals_data_agent import FundamentalsDataAgent
            from ..data.cross_asset_features import CrossAssetFeatures
            
            fundamentals_agent = FundamentalsDataAgent(api_key=os.getenv('FMP_API_KEY'), config=config)
            cross_asset_features = CrossAssetFeatures(config=config)
            fmp_enabled = True
            logger.info("✅ FMP Fundamentals enabled")
        except Exception as e:
            logger.warning(f"Failed to initialize FMP components: {e}")

    logger.info(f"Fetching historical data for {len(symbols)} symbols from {start_date} to {end_date}")

    for i, symbol in enumerate(symbols):
        try:
            logger.info(f"[{i+1}/{len(symbols)}] Fetching data for {symbol}...")
            # Fetch historical data using Alpha Vantage with retry logic
            data = None
            max_retries = 3

            for attempt in range(max_retries):
                try:
                    data = data_ingestion.get_daily_data(symbol, outputsize='full')
                    break  # Success, exit retry loop
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {e}. Retrying...")
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        logger.error(f"All {max_retries} attempts failed for {symbol}: {e}")
                        # Don't raise, just log and continue to next symbol
                        pass

            if data is not None and not data.empty:
                # Filter to date range
                data = data[(data.index >= start_date) & (data.index <= end_date)]

                # Convert index to string for JSON serialization
                data.index = data.index.astype(str)

                # Validate data structure
                if _validate_data_structure(data):
                    
                    # Enrich with fundamentals if enabled
                    if fmp_enabled and fundamentals_agent and cross_asset_features:
                        try:
                            # Fetch fundamentals (cached internally by agent)
                            fundamentals = fundamentals_agent.update_symbol_fundamentals(symbol)
                            
                            if fundamentals is not None and not fundamentals.empty:
                                # Merge using cross_asset_features logic
                                # Note: We need to convert index back to datetime for merging
                                price_df = data.copy()
                                price_df.index = pd.to_datetime(price_df.index)
                                
                                # Forward fill fundamentals to match price dates
                                fund_daily = cross_asset_features._forward_fill_fundamentals(fundamentals, price_df.index)
                                
                                # Join
                                data = price_df.join(fund_daily, how='left')
                                
                                # Convert index back to string
                                data.index = data.index.astype(str)
                                logger.info(f"   + Enriched {symbol} with {len(fundamentals.columns)} fundamental features")
                        except Exception as e:
                            logger.warning(f"Failed to enrich {symbol} with fundamentals: {e}")

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

    state['raw_data'] = raw_data
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
    Synchronous wrapper for the data loading function.
    """
    return _load_data_sync(state)
