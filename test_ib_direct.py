#!/usr/bin/env python3
"""
Test IBKR connection directly using ib_insync
"""

import asyncio
import logging
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.ib_data_ingestion_real import IBDataIngestion

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_ib_connection():
    """Test IBKR connection directly."""
    logger.info("Testing IBKR connection...")

    # Initialize IB ingestion
    ib_ingestion = IBDataIngestion(
        host="127.0.0.1",
        port=7497,
        client_id=1,
        market_data_type=3
    )

    # Try to connect
    connected = await ib_ingestion.connect()

    if connected:
        logger.info("✅ IBKR connection successful!")
        logger.info(f"Connected to {ib_ingestion.host}:{ib_ingestion.port} with client_id {ib_ingestion.client_id}")

        # Try to get some data
        try:
            logger.info("Testing data fetch for AAPL...")
            df = await ib_ingestion.get_historical_data(
                symbol="AAPL",
                start_date="2024-01-01",
                end_date="2024-01-10",
                timeframe="1 day"
            )

            if df is not None and len(df) > 0:
                logger.info(f"✅ Successfully fetched {len(df)} rows of data for AAPL")
                logger.info(f"Columns: {list(df.columns)}")
                logger.info(f"Sample data:\n{df.head()}")
            else:
                logger.warning("⚠️ No data returned for AAPL")

        except Exception as e:
            logger.error(f"❌ Failed to fetch data: {e}")

        # Disconnect
        await ib_ingestion.disconnect()
        logger.info("Disconnected from IBKR")

    else:
        logger.error("❌ IBKR connection failed!")
        return False

    return True

if __name__ == "__main__":
    result = asyncio.run(test_ib_connection())
    sys.exit(0 if result else 1)