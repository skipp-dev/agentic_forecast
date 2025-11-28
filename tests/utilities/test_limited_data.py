#!/usr/bin/env python3

import sys
import os
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.data.unified_ingestion_v2 import UnifiedDataIngestion

def test_data_ingestion():
    """Test data ingestion with a small subset of symbols."""

    print("🧪 Testing Data Ingestion with Limited Symbols")
    print("=" * 50)

    # Test with just a few symbols
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']

    # Initialize data ingestion
    data_ingestion = UnifiedDataIngestion(
        use_real_data=True,
        market_data_type=3,  # Delayed data
        config={}
    )

    try:
        print("Initializing data ingestion...")
        data_ingestion.initialize()
        print(f"✅ Primary source: {data_ingestion.primary_source.upper()}")

        # Define time window
        end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        start_date = (pd.Timestamp.now() - pd.Timedelta(days=365)).strftime('%Y-%m-%d')

        print(f"📅 Date range: {start_date} to {end_date}")
        print(f"📊 Testing symbols: {test_symbols}")

        results = {}

        for symbol in test_symbols:
            print(f"\n🔍 Fetching data for {symbol}...")
            try:
                df = data_ingestion.get_historical_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe='1D'
                )

                if df is not None and not df.empty:
                    print(f"✅ {symbol}: {len(df)} rows retrieved")
                    results[symbol] = len(df)

                    # Save to file for inspection
                    output_file = f"data/raw/{symbol}_test.csv"
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    df.to_csv(output_file)
                    print(f"💾 Saved to {output_file}")
                else:
                    print(f"❌ {symbol}: No data retrieved")
                    results[symbol] = 0

            except Exception as e:
                print(f"❌ {symbol}: Error - {e}")
                results[symbol] = f"Error: {e}"

        # Summary
        print("\n📋 Test Results Summary:")
        print("-" * 25)
        successful = sum(1 for v in results.values() if isinstance(v, int) and v > 0)
        print(f"✅ Successful: {successful}/{len(test_symbols)} symbols")

        for symbol, result in results.items():
            if isinstance(result, int) and result > 0:
                print(f"   {symbol}: ✅ {result} rows")
            else:
                print(f"   {symbol}: ❌ {result}")

        # Check connection status
        status = data_ingestion.get_data_status()
        print("\n🔗 Connection Status:")
        print(f"   IB Connected: {status.get('ib_connected', False)}")
        print(f"   Host: {status.get('ib_host', 'N/A')}")
        print(f"   Cached Symbols: {status.get('cached_symbols', 0)}")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\n🔌 Cleaning up connections...")
        data_ingestion.disconnect()

if __name__ == "__main__":
    test_data_ingestion()