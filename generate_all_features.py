#!/usr/bin/env python3
"""
Generate features for all symbols using existing raw data.
This bypasses the dependency issues in the main pipeline.
"""

import sys
import os
import pandas as pd
import glob
from pathlib import Path

# Add project root to path
sys.path.insert(0, '.')

def generate_features_for_all_symbols():
    """Generate features for all symbols with existing raw data."""

    print("🔧 Generating features for all symbols...")

    # Import the FeatureEngineer
    from pipelines.run_features import FeatureEngineer
    engineer = FeatureEngineer()

    # Find all raw data files
    raw_data_dir = Path("data/raw/alpha_vantage")
    if not raw_data_dir.exists():
        print(f"❌ Raw data directory not found: {raw_data_dir}")
        return False

    # Get all parquet files
    parquet_files = list(raw_data_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"❌ No parquet files found in {raw_data_dir}")
        return False

    print(f"📊 Found {len(parquet_files)} raw data files")

    # Create processed directory
    processed_dir = Path("data/processed/default")
    processed_dir.mkdir(parents=True, exist_ok=True)

    successful = 0
    failed = 0

    for parquet_file in parquet_files:
        symbol = parquet_file.stem  # Remove .parquet extension

        try:
            # Load raw data
            raw_data = pd.read_parquet(parquet_file)
            if raw_data.empty:
                print(f"⚠️  Empty data for {symbol}, skipping")
                failed += 1
                continue

            # Generate features
            features_df = engineer.engineer_features_for_symbol(symbol, raw_data, experiment="default")

            if features_df.empty:
                print(f"⚠️  No features generated for {symbol}")
                failed += 1
                continue

            # Save features
            output_file = processed_dir / f"{symbol}_features.parquet"
            features_df.to_parquet(output_file)

            successful += 1
            print(f"✅ Generated {features_df.shape[1]} features for {symbol}")

        except Exception as e:
            print(f"❌ Failed to process {symbol}: {e}")
            failed += 1

    print(f"\n📈 Feature Generation Summary:")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📊 Total processed: {successful + failed}")

    return successful > 0

if __name__ == "__main__":
    success = generate_features_for_all_symbols()
    sys.exit(0 if success else 1)