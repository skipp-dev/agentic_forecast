#!/usr/bin/env python3

import sys
import os
import pandas as pd
import time
sys.path.append(os.path.dirname(__file__))

from src.data.unified_ingestion_v2 import UnifiedDataIngestion

def test_sequential_ingestion():
    """Test data ingestion with sequential processing."""

    print("🧪 Testing Sequential Data Ingestion")
    print("=" * 40)

    # Test with just a few symbols
    test_symbols = ['AAPL', 'MSFT']

    results = {}

    for symbol in test_symbols:
        print(f"\n🔍 Processing {symbol}...")

        # Create a fresh ingestion instance for each symbol
        data_ingestion = UnifiedDataIngestion(
            use_real_data=True,
            market_data_type=3,
            config={}
        )

        try:
            # Initialize
            data_ingestion.initialize()
            print(f"   ✅ Initialized (source: {data_ingestion.primary_source})")

            # Define time window
            end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
            start_date = (pd.Timestamp.now() - pd.Timedelta(days=365)).strftime('%Y-%m-%d')

            # Get data
            df = data_ingestion.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe='1D'
            )

            if df is not None and not df.empty:
                print(f"   ✅ {symbol}: {len(df)} rows retrieved")
                results[symbol] = len(df)

                # Save to file
                output_file = f"data/raw/{symbol}_seq.csv"
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                df.to_csv(output_file)
                print(f"   💾 Saved to {output_file}")
            else:
                print(f"   ❌ {symbol}: No data retrieved")
                results[symbol] = 0

        except Exception as e:
            print(f"   ❌ {symbol}: Error - {e}")
            results[symbol] = f"Error: {e}"

        finally:
            # Clean disconnect
            try:
                data_ingestion.disconnect()
                print(f"   🔌 Disconnected {symbol}")
            except:
                pass

        # Small delay between symbols
        time.sleep(1)

    # Summary
    print("\n📋 Sequential Test Results:")
    print("-" * 30)
    successful = sum(1 for v in results.values() if isinstance(v, int) and v > 0)
    print(f"✅ Successful: {successful}/{len(test_symbols)} symbols")

    for symbol, result in results.items():
        if isinstance(result, int) and result > 0:
            print(f"   {symbol}: ✅ {result} rows")
        else:
            print(f"   {symbol}: ❌ {result}")

if __name__ == "__main__":
    test_sequential_ingestion()