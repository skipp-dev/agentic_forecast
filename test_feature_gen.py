#!/usr/bin/env python3
"""
Simple test to verify feature generation is working.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, '.')

def test_feature_generation():
    """Test that feature generation works with the new FeatureEngineer."""

    print("Testing feature generation...")

    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'open': np.random.uniform(100, 110, 100),
        'high': np.random.uniform(105, 115, 100),
        'low': np.random.uniform(95, 105, 100),
        'close': np.random.uniform(100, 110, 100),
        'volume': np.random.uniform(1000000, 5000000, 100)
    }, index=dates)

    print(f"Sample data shape: {sample_data.shape}")
    print(f"Sample data columns: {sample_data.columns.tolist()}")

    try:
        from pipelines.run_features import FeatureEngineer

        engineer = FeatureEngineer()
        print("FeatureEngineer instantiated successfully")

        # Test feature generation
        features = engineer.engineer_features_for_symbol('TEST', sample_data, experiment='default')
        print(f"Feature generation successful! Generated {features.shape[1]} features")
        print(f"Feature columns: {features.columns.tolist()[:10]}...")

        # Check for expected features
        expected_features = ['returns_1d', 'returns_5d', 'volatility_5d', 'sma_20', 'rsi_14']
        found_features = [f for f in expected_features if f in features.columns]
        print(f"Found expected features: {found_features}")

        return True

    except Exception as e:
        print(f"Error during feature generation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_feature_generation()
    sys.exit(0 if success else 1)