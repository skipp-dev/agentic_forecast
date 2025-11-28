import asyncio
import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.data.ib_data_ingestion_real import IBDataIngestion

async def test_ib_connection():
    """Test IBKR connection with improved client ID handling."""
    print("Testing IBKR connection with improved client ID rotation...")
    
    # Initialize with config values
    ib_ingestion = IBDataIngestion(
        host="127.0.0.1",
        port=7497,
        client_id=1,  # This will be overridden with random ID
        readonly=True,
        market_data_type=3
    )
    
    try:
        # Attempt connection
        connected = await ib_ingestion.connect()
        
        if connected:
            print(" Connection successful!")
            print(f"Connected to: {ib_ingestion.host}:{ib_ingestion.port}")
            print(f"Client ID used: {ib_ingestion.client_id}")
            
            # Test getting account info
            try:
                accounts = ib_ingestion.ib.managedAccounts()
                print(f"Available accounts: {accounts}")
            except Exception as e:
                print(f"Could not get account info: {e}")
            
            # Disconnect
            await ib_ingestion.disconnect()
            print(" Disconnected successfully")
        else:
            print(" Connection failed")
            
    except Exception as e:
        print(f" Test failed with error: {e}")

if __name__ == "__main__":
    asyncio.run(test_ib_connection())
