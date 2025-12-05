
import sys
import os
import pandas as pd
import logging

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from agents.macro_data_agent import MacroDataAgent
from agents.regime_detection_agent import RegimeDetectionAgent
from agents.regime_agent import RegimeAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_macro_agent():
    logger.info("Testing MacroDataAgent...")
    agent = MacroDataAgent()
    
    # Test fetching data
    # We use a longer lookback to ensure we have enough data for 90-day changes
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=150)
    
    try:
        macro_data = agent.get_macro_data(start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))
        logger.info(f"Fetched macro data keys: {list(macro_data.keys())}")
        
        if 'processed_features' in macro_data:
            df = macro_data['processed_features']
            logger.info(f"Processed features shape: {df.shape}")
            logger.info(f"Processed features columns: {df.columns.tolist()}")
            if df.empty:
                logger.warning("Processed features DataFrame is empty!")
        else:
            logger.warning("No 'processed_features' in macro_data!")
            
        return macro_data
    except Exception as e:
        logger.error(f"MacroDataAgent failed: {e}")
        return None

def test_regime_agent(macro_data):
    logger.info("Testing RegimeDetectionAgent...")
    agent = RegimeDetectionAgent()
    
    # Mock market data (needed for regime detection usually)
    # But let's see what detect_regime needs
    # It usually needs market data (SPY/VIX)
    
    # Let's try to fetch some market data using yfinance or mock it
    # For now, let's see if we can run it with minimal data
    
    try:
        # We need macro features
        if 'processed_features' not in macro_data:
            logger.error("No processed features in macro_data")
            return None
            
        macro_features = macro_data['processed_features']
        
        regime = agent.detect_regimes(macro_features)
        logger.info(f"Detected Regimes: {list(regime.keys())}")
        
        # Print last detected regime for each type
        for r_type, r_series in regime.items():
            if not r_series.empty:
                logger.info(f"Last {r_type}: {r_series.iloc[-1]}")
                
        return regime
    except Exception as e:
        logger.error(f"RegimeDetectionAgent failed: {e}")
        return None

def test_regime_agent_simple(macro_data):
    logger.info("Testing RegimeAgent (Simple)...")
    try:
        agent = RegimeAgent()
        # RegimeAgent.detect_regime takes target_date and macro_data (raw)
        raw_data = macro_data.get('raw_data', {})
        target_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        
        regimes = agent.detect_regime(target_date, raw_data)
        logger.info(f"Detected Regimes (Simple): {regimes}")
        return regimes
    except Exception as e:
        logger.error(f"RegimeAgent (Simple) failed: {e}")
        return None

if __name__ == "__main__":
    macro_data = test_macro_agent()
    if macro_data:
        test_regime_agent(macro_data)
        test_regime_agent_simple(macro_data)
