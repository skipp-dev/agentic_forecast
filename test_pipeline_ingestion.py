import asyncio
import sys
import os
sys.path.insert(0, 'src')

from data.unified_ingestion_v2 import UnifiedDataIngestion
import yaml

async def test_pipeline_ingestion():
    """Test the unified ingestion as used in the pipeline."""
    print("Testing unified ingestion with IBKR primary...")
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize unified ingestion
    ingestion = UnifiedDataIngestion(config=config)
    
    print(f"Primary source: {ingestion.primary_source}")
    print(f"IBKR host: {ingestion.ib_ingestion.host}")
    print(f"IBKR port: {ingestion.ib_ingestion.port}")
    
    # Test fetching data for a single symbol
    try:
        print("Testing data fetch for AAPL...")
        data = await ingestion.fetch_historical_data('AAPL', '2024-01-01', '2024-01-05')
        if data is not None:
            print(f" Successfully fetched {len(data)} rows from {ingestion.last_source_used}")
        else:
            print(" Failed to fetch data")
    except Exception as e:
        print(f" Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_pipeline_ingestion())
