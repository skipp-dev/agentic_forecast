#!/usr/bin/env python3
"""
Run Cross-Asset Feature Engineering

This script generates cross-asset features that capture relationships between
stocks, crypto, commodities, and macro indicators.
"""

import sys
from pathlib import Path
from datetime import datetime
import logging
import pandas as pd
from typing import Dict, List, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cross_asset_features import CrossAssetFeatureEngineer
from data.feature_store import TimeSeriesFeatureStore

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_cross_asset_feature_engineering(symbols: Optional[List[str]] = None,
                                      start_date: Optional[str] = None,
                                      end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Run cross-asset feature engineering for specified symbols.

    Args:
        symbols: List of symbols to process (default: major symbols)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        Dictionary mapping symbols to their cross-asset feature DataFrames
    """
    logger.info("Starting cross-asset feature engineering")

    # Initialize components
    cross_asset_engineer = CrossAssetFeatureEngineer()
    feature_store = TimeSeriesFeatureStore(store_path='data/feature_store')

    # Default symbols if none provided
    if symbols is None:
        symbols = [
            # Major tech stocks
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
            # Crypto-related
            'COIN', 'MSTR',
            # Other major stocks
            'JPM', 'BAC', 'WFC', 'XOM', 'JNJ'
        ]

    logger.info(f"Processing {len(symbols)} symbols: {symbols}")

    results = {}

    for symbol in symbols:
        try:
            logger.info(f"Engineering cross-asset features for {symbol}")

            # Generate cross-asset features
            features_df = cross_asset_engineer.engineer_cross_asset_features(
                target_symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )

            if not features_df.empty:
                # Store features in feature store
                feature_set_id = feature_store.store_features(
                    symbol=symbol,
                    features_df=features_df,
                    feature_set_name='cross_asset_features'
                )

                results[symbol] = features_df
                logger.info(f"Generated {len(features_df.columns)} features for {symbol}")
            else:
                logger.warning(f"No features generated for {symbol}")

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            continue

    # Summary
    successful_symbols = len(results)
    total_features = sum(len(df.columns) for df in results.values()) if results else 0

    logger.info("Cross-asset feature engineering completed")
    logger.info(f"Successfully processed {successful_symbols}/{len(symbols)} symbols")
    logger.info(f"Total features generated: {total_features}")

    return results

def validate_cross_asset_features(results: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Validate the generated cross-asset features.

    Args:
        results: Results from feature engineering

    Returns:
        Validation report
    """
    validation_report = {
        'total_symbols': len(results),
        'features_per_symbol': {},
        'feature_types': set(),
        'data_quality': {},
        'cross_asset_coverage': {}
    }

    for symbol, features_df in results.items():
        validation_report['features_per_symbol'][symbol] = len(features_df.columns)

        # Check for NaN values
        nan_percentage = features_df.isnull().mean().mean() * 100
        validation_report['data_quality'][symbol] = {
            'nan_percentage': nan_percentage,
            'total_rows': len(features_df),
            'date_range': f"{features_df.index.min()} to {features_df.index.max()}"
        }

        # Collect feature types
        validation_report['feature_types'].update(features_df.columns)

    # Cross-asset feature coverage
    cross_asset_features = [
        'crypto_', 'ai_sector_', 'tech_sector_',
        'gold_', 'oil_', 'fed_funds_', 'unemployment_',
        'correlation', 'momentum_divergence'
    ]

    for symbol, features_df in results.items():
        coverage = {}
        for feature_prefix in cross_asset_features:
            matching_features = [col for col in features_df.columns if feature_prefix in col]
            coverage[feature_prefix] = len(matching_features)

        validation_report['cross_asset_coverage'][symbol] = coverage

    return validation_report

def main():
    """Main execution function."""
    print("Cross-Asset Feature Engineering")
    print("=" * 40)

    # Run cross-asset feature engineering
    results = run_cross_asset_feature_engineering()

    if results:
        print(f"\nSuccessfully generated features for {len(results)} symbols")

        # Show sample features for first symbol
        first_symbol = list(results.keys())[0]
        features_df = results[first_symbol]

        print(f"\nSample features for {first_symbol}:")
        print(f"Total features: {len(features_df.columns)}")
        print(f"Date range: {features_df.index.min()} to {features_df.index.max()}")
        print(f"Sample feature columns: {list(features_df.columns[:10])}")

        # Validate results
        validation = validate_cross_asset_features(results)
        print("\nValidation Summary:")
        print(f"Symbols processed: {validation['total_symbols']}")
        print(f"Feature types: {len(validation['feature_types'])}")

        # Show cross-asset coverage for first symbol
        coverage = validation['cross_asset_coverage'][first_symbol]
        print(f"\nCross-asset feature coverage for {first_symbol}:")
        for feature_type, count in coverage.items():
            if count > 0:
                print(f"  {feature_type}: {count} features")

    else:
        print("No features were generated")

    print("\nCross-asset feature engineering completed!")

if __name__ == "__main__":
    main()