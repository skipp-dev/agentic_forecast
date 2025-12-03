import os
import sys
import logging
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock state and config
state = {
    'symbols': ['AAPL'],
    'config': {
        'alpha_vantage': {'rate_limit': 1200},
        'fmp': {
            'enabled': True,
            'universe': 'top_liquid'
        }
    },
    'errors': []
}

# Ensure API key is present (it should be in env from previous steps)
if not os.getenv('FMP_API_KEY'):
    logger.error("FMP_API_KEY not found in environment variables!")
    sys.exit(1)

try:
    from src.nodes.data_nodes import load_data_node
    
    logger.info("Running load_data_node...")
    new_state = load_data_node(state)
    
    raw_data = new_state.get('raw_data', {})
    if 'AAPL' in raw_data:
        df = raw_data['AAPL']
        logger.info(f"Data loaded for AAPL. Shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Check for fundamental columns
        fund_cols = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume']]
        if fund_cols:
            logger.info(f"✅ Found {len(fund_cols)} fundamental columns: {fund_cols[:5]}...")
            logger.info("Sample data:")
            logger.info(df[fund_cols].tail())
        else:
            logger.warning("❌ No fundamental columns found!")
    else:
        logger.error("❌ AAPL data not found in state!")

except Exception as e:
    logger.error(f"Execution failed: {e}")
    import traceback
    traceback.print_exc()
