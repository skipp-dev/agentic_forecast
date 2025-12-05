import os
import pandas as pd
import logging
import asyncio
import time
from datetime import datetime, timedelta
from ..graphs.state import GraphState
from ..alpha_vantage_client import AlphaVantageClient
from ..agents.liquidity_agent import LiquidityAgent

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
    
    # Initialize Liquidity Agent
    liquidity_agent = LiquidityAgent(config=config.get('liquidity', {}))
    rejected_symbols = []

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
    
    # Check config and run_type
    run_type = state.get('run_type', 'DAILY')
    # Also check env var as fallback
    if not run_type and os.environ.get('RUN_TYPE'):
        run_type = os.environ.get('RUN_TYPE')
        
    config_fmp_enabled = config.get('fmp', {}).get('enabled', False)
    
    # Disable FMP for BACKTEST unless explicitly enabled (fail-safe)
    if run_type == 'BACKTEST' and not config.get('fmp', {}).get('allow_in_backtest', False):
        logger.info("ℹ️ FMP Fundamentals disabled for BACKTEST run")
        fmp_enabled = False
    elif config_fmp_enabled and os.getenv('FMP_API_KEY'):
        try:
            from ..agents.fundamentals_data_agent import FundamentalsDataAgent
            from ..data.cross_asset_features import CrossAssetFeatures
            
            fundamentals_agent = FundamentalsDataAgent(api_key=os.getenv('FMP_API_KEY'), config=config)
            cross_asset_features = CrossAssetFeatures(config=config)
            fmp_enabled = True
            logger.info("✅ FMP Fundamentals enabled")
        except Exception as e:
            logger.warning(f"Failed to initialize FMP components: {e}")
            fmp_enabled = False

    logger.info(f"Fetching historical data for {len(symbols)} symbols from {start_date} to {end_date}")

    # 1. Fetch Prices
    prices_dict = {}
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
                        pass

            if data is not None and not data.empty:
                # Filter to date range
                data = data[(data.index >= start_date) & (data.index <= end_date)]
                # Ensure index is datetime for merging
                data.index = pd.to_datetime(data.index)
                
                # --- LIQUIDITY CHECK ---
                # Calculate metrics and check tradeability
                lq_metrics = liquidity_agent.calculate_metrics(data)
                tradeability = liquidity_agent.check_tradeability(symbol, lq_metrics)
                
                if not tradeability['tradeable']:
                    logger.warning(f"Symbol {symbol} rejected by LiquidityAgent: {tradeability['reasons']}")
                    rejected_symbols.append({'symbol': symbol, 'reasons': tradeability['reasons']})
                    # Skip adding to raw_data
                    continue
                else:
                    logger.info(f"Symbol {symbol} passed liquidity check.")
                # -----------------------
                
                if _validate_data_structure(data):
                    prices_dict[symbol] = data
                else:
                    state['errors'].append(f"Data validation failed for {symbol}")
            else:
                state['errors'].append(f"No data for {symbol}")
                
        except Exception as e:
            state['errors'].append(f"Error loading data for {symbol}: {e}")

    # 2. Fetch Fundamentals (if enabled)
    fundamentals_dict = {}
    if fmp_enabled and fundamentals_agent:
        logger.info("Fetching fundamentals for enrichment...")
        for symbol in prices_dict.keys():
            try:
                # Fetch fundamentals (cached internally by agent)
                # We use update_symbol_fundamentals which handles caching and feature calculation
                fund_df = fundamentals_agent.update_symbol_fundamentals(symbol)
                if fund_df is not None and not fund_df.empty:
                    fundamentals_dict[symbol] = fund_df
            except Exception as e:
                logger.warning(f"Failed to fetch fundamentals for {symbol}: {e}")

    # 3. Merge
    if fmp_enabled and cross_asset_features and fundamentals_dict:
        try:
            merged_df = cross_asset_features.merge_prices_and_fundamentals(prices_dict, fundamentals_dict)
            
            # Split back to dict for state['raw_data']
            # merged_df has MultiIndex (date, symbol)
            if not merged_df.empty:
                for symbol in prices_dict.keys():
                    if symbol in merged_df.index.get_level_values('symbol'):
                        # Extract symbol data
                        symbol_data = merged_df.xs(symbol, level='symbol')
                        # Ensure index is string for JSON serialization compatibility if needed, 
                        # but usually we keep it as datetime until serialization.
                        # The original code converted to string: data.index = data.index.astype(str)
                        # We should probably do that at the end.
                        raw_data[symbol] = symbol_data
                        logger.info(f"   + Enriched {symbol} with fundamentals (Total cols: {len(symbol_data.columns)})")
                    else:
                        # Fallback to price only if merge failed for this symbol
                        raw_data[symbol] = prices_dict[symbol]
            else:
                raw_data = prices_dict
        except Exception as e:
            logger.error(f"Error during cross-asset merge: {e}")
            raw_data = prices_dict
    else:
        raw_data = prices_dict

    # Final cleanup: Ensure indices are DatetimeIndex for downstream processing
    # We do NOT convert to string here because FeatureEngineer requires DatetimeIndex
    # Serialization happens in the nodes that produce final outputs (e.g. features, forecasts)
    
    # Ensure output directory exists
    output_dir = os.path.join("data", "raw", "alpha_vantage")
    os.makedirs(output_dir, exist_ok=True)

    for symbol, df in raw_data.items():
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Save to Parquet for robust serialization/checkpointing
        try:
            parquet_path = os.path.join(output_dir, f"{symbol}.parquet")
            df.to_parquet(parquet_path)
            logger.debug(f"Saved {symbol} data to {parquet_path}")
        except Exception as e:
            logger.warning(f"Failed to save Parquet for {symbol}: {e}")

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
