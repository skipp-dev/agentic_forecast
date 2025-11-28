#!/usr/bin/env python3
"""
Test unified data ingestion with IBKR as primary source
"""

import asyncio
import logging
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.unified_ingestion_v2 import UnifiedDataIngestion
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_unified_ingestion():
    """Test unified data ingestion with IBKR as primary."""
    logger.info("Testing unified data ingestion with IBKR as primary source...")

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize unified ingestion
    data_ingestion = UnifiedDataIngestion(
        use_real_data=True,
        config=config,
        skip_sentiment=True  # Skip sentiment to avoid API timeouts
    )

    # Initialize
    if not data_ingestion.initialize():
        logger.error("Failed to initialize data ingestion")
        return False

    logger.info(f"Primary source: {data_ingestion.primary_source}")
    logger.info(f"IBKR available: {data_ingestion.ib_available}")
    logger.info(f"Alpha Vantage available: {data_ingestion.av_available}")

    # Test data fetching for AAPL with a longer date range
    logger.info("Testing data fetch for AAPL...")
    df = data_ingestion.get_historical_data(
        symbol="AAPL",
        start_date="2023-12-01",  # Use a date range that should have data
        end_date="2024-01-10",
        timeframe="1 day"
    )

    if df is not None and len(df) > 0:
        logger.info(f"✅ Successfully fetched {len(df)} rows of data for AAPL")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        logger.info(f"Sample data:\n{df.head()}")

        # Check if it's IBKR data (should have 'bar_count' column)
        if 'bar_count' in df.columns:
            logger.info("✅ Data appears to be from IBKR (has bar_count column)")
        else:
            logger.info("ℹ️ Data appears to be from Alpha Vantage (no bar_count column)")

        return True
    else:
        logger.error("❌ No data returned for AAPL")
        return False

if __name__ == "__main__":
    result = test_unified_ingestion()
    sys.exit(0 if result else 1)