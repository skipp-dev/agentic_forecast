#!/usr/bin/env python3
"""
Fix the evaluation bug where all symbols have identical metrics.
The issue is that evaluation is done globally instead of per-symbol.
This script recalculates metrics properly for each symbol.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
sys.path.append('src')
sys.path.append('analytics')  # Add analytics to path for regime utils

from analytics.evaluation_metrics import create_regime_flags_from_data

def calculate_naive_metrics_per_symbol(symbol_data, horizon, price_column='adjusted_close'):
    """
    Calculate naive baseline metrics for a single symbol with robust MAPE handling.
    Naive baseline: last value persistence.
    Includes SMAPE, MASE, and quality flags.
    """
    if len(symbol_data) <= horizon:
        return {
            'mae': np.nan,
            'rmse': np.nan,
            'mape': np.nan,
            'smape': np.nan,
            'mase': np.nan,
            'directional_accuracy': np.nan,
            'mape_flag': 'unreliable',
            'n_samples': 0
        }

    # Use last 'horizon' values as predictions for the next 'horizon' periods
    actuals = symbol_data[price_column].values[-horizon:]
    predictions = np.repeat(symbol_data[price_column].values[-horizon-1], horizon)

    # Calculate basic error metrics
    abs_errors = np.abs(actuals - predictions)
    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(abs_errors ** 2))

    # Calculate MAPE with safeguards (minimum denominator, exclude bad points)
    eps = 1e-6  # Minimum denominator in price units
    valid_mask = np.abs(actuals) > eps  # Exclude near-zero actuals

    if np.sum(valid_mask) > 0:
        # Use maximum of |actual| and epsilon to avoid division by very small numbers
        denom = np.maximum(np.abs(actuals), eps)
        mape_values = abs_errors / denom
        mape = np.mean(mape_values[valid_mask])

        # Flag MAPE quality
        if np.sum(valid_mask) < len(actuals) * 0.5:  # Less than 50% valid points
            mape_flag = 'unreliable'
        elif mape > 0.5:  # Very high MAPE
            mape_flag = 'high'
        else:
            mape_flag = 'ok'
    else:
        mape = np.nan
        mape_flag = 'unreliable'

    # Calculate SMAPE (Symmetric MAPE) - more robust than MAPE
    eps_smape = 1e-6
    smape_denom = np.abs(actuals) + np.abs(predictions) + eps_smape
    smape_values = 2 * abs_errors / smape_denom
    smape = np.mean(smape_values)

    # Calculate SWASE (Shock-Weighted Absolute Scaled Error)
    # Use actual regime detection based on returns magnitude and volatility
    eps_swase = 1e-8
    ase_terms = abs_errors / (np.abs(actuals) + eps_swase)

    # Define shock weight for shock days
    shock_weight = 3.0  # Higher weight for shock days

    # Detect shock days using returns-based method
    if len(actuals) > 1:
        # Calculate returns from prices (assuming actuals are prices)
        returns = np.diff(actuals) / actuals[:-1]
        # Pad returns to match original length
        returns = np.concatenate([[0], returns])  # First day has no return

        # Create regime flags
        regime_flags = create_regime_flags_from_data(returns, method='combined',
                                                   returns_threshold=0.03,  # 3% return threshold
                                                   window=20, threshold_std=2.0)

        # Apply shock weighting
        weights = np.ones_like(ase_terms, dtype=float)
        shock_mask = regime_flags['peer_shock_flag'][:len(weights)]  # Ensure same length
        weights[shock_mask] = shock_weight

        swase = np.average(ase_terms, weights=weights)
    else:
        # Fallback for very short series
        swase = np.mean(ase_terms)

    # Calculate MASE (Mean Absolute Scaled Error) - scale-free benchmark
    # First compute naive scale Q (average absolute change in training data)
    if len(symbol_data) > horizon + 1:
        # Use all available data except the last 'horizon' points for scale
        training_data = symbol_data[price_column].values[:-horizon]
        if len(training_data) > 1:
            naive_errors = np.abs(np.diff(training_data))
            Q = np.mean(naive_errors)
            if Q > 0:
                mase = mae / Q
            else:
                mase = np.nan
        else:
            mase = np.nan
    else:
        mase = np.nan

    # Calculate directional accuracy
    actual_changes = np.diff(actuals)
    pred_changes = np.diff(predictions)
    if len(actual_changes) > 0:
        directional_accuracy = np.mean(np.sign(actual_changes) == np.sign(pred_changes))
    else:
        directional_accuracy = np.nan

    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'smape': smape,
        'swase': swase,
        'mase': mase,
        'directional_accuracy': directional_accuracy,
        'mape_flag': mape_flag,
        'n_samples': len(actuals)
    }

def fix_evaluation_results():
    """Fix the evaluation results by calculating per-symbol metrics."""

    # Load raw data to get actual symbol data
    raw_data_path = Path('data/raw/alpha_vantage')
    if not raw_data_path.exists():
        print("❌ Raw data directory not found")
        return

    # Load existing evaluation results
    eval_file = Path('data/metrics/evaluation_results_baseline_latest.csv')
    if not eval_file.exists():
        print("❌ Evaluation results file not found")
        return

    df = pd.read_csv(eval_file)
    print(f"📄 Loaded {len(df)} evaluation records")

    # Get unique symbols and horizons
    symbols = df['symbol'].unique()
    horizons = df['target_horizon'].unique()

    print(f"🔍 Processing {len(symbols)} symbols across {len(horizons)} horizons")

    # Create fixed evaluation results
    fixed_results = []

    for symbol in symbols:
        # Load symbol data
        symbol_file = raw_data_path / f"{symbol}.parquet"
        if not symbol_file.exists():
            symbol_file = raw_data_path / f"{symbol}.csv"

        if not symbol_file.exists():
            print(f"⚠️  No data file found for {symbol}")
            continue

        try:
            if symbol_file.suffix == '.parquet':
                symbol_data = pd.read_parquet(symbol_file)
            else:
                symbol_data = pd.read_csv(symbol_file)

            if 'close' not in symbol_data.columns:
                print(f"⚠️  No 'close' column in {symbol} data")
                continue

            # Use adjusted_close if available (handles stock splits properly)
            price_column = 'adjusted_close' if 'adjusted_close' in symbol_data.columns else 'close'

            # Ensure we have enough data
            if len(symbol_data) < 50:  # Minimum data requirement
                print(f"⚠️  Insufficient data for {symbol} ({len(symbol_data)} rows)")
                continue

            for horizon in horizons:
                # Calculate proper metrics for this symbol and horizon
                metrics = calculate_naive_metrics_per_symbol(symbol_data, horizon, price_column)

                fixed_results.append({
                    'symbol': symbol,
                    'model_type': 'naive',
                    'target_horizon': horizon,
                    'experiment': 'baseline',
                    'predictions_count': metrics['n_samples'],
                    'evaluation_timestamp': datetime.now().isoformat(),
                    'mae': metrics['mae'],
                    'mse': metrics['mae'] ** 2,  # Approximation
                    'rmse': metrics['rmse'],
                    'mape': metrics['mape'],
                    'smape': metrics['smape'],
                    'swase': metrics['swase'],
                    'mase': metrics['mase'],
                    'directional_accuracy': metrics['directional_accuracy'],
                    'mape_flag': metrics['mape_flag'],
                    'n_samples': metrics['n_samples']
                })

        except Exception as e:
            print(f"❌ Error processing {symbol}: {e}")
            continue

    if not fixed_results:
        print("❌ No valid results generated")
        return

    # Create new DataFrame
    fixed_df = pd.DataFrame(fixed_results)

    # Save fixed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fixed_file = Path(f'data/metrics/evaluation_results_baseline_fixed_{timestamp}.csv')
    fixed_df.to_csv(fixed_file, index=False)

    # Update latest file
    latest_file = Path('data/metrics/evaluation_results_baseline_latest.csv')
    fixed_df.to_csv(latest_file, index=False)

    print("✅ Fixed evaluation results saved!")
    print(f"📊 Generated {len(fixed_df)} evaluation records")
    print(f"🏢 Unique symbols: {fixed_df['symbol'].nunique()}")
    print(f"⏰ Horizons: {sorted(fixed_df['target_horizon'].unique())}")

    # Show sample of fixed results
    print("\n📋 SAMPLE FIXED RESULTS:")
    sample = fixed_df.head(10)[['symbol', 'target_horizon', 'mae', 'rmse', 'mape', 'directional_accuracy']]
    print(sample.to_string(index=False))

    # Check if metrics are now different per symbol
    print("\n🔍 VALIDATION:")
    for horizon in sorted(fixed_df['target_horizon'].unique()):
        h_df = fixed_df[fixed_df['target_horizon'] == horizon]
        unique_mae = len(h_df['mae'].unique())
        unique_rmse = len(h_df['rmse'].unique())
        unique_mape = len(h_df['mape'].unique())

        print(f"  Horizon {horizon}: {len(h_df)} symbols")
        print(f"    Unique MAE values: {unique_mae}")
        print(f"    Unique RMSE values: {unique_rmse}")
        print(f"    Unique MAPE values: {unique_mape}")

        if unique_mae > 1 and unique_rmse > 1:
            print("    ✅ Metrics vary by symbol (FIXED!)")
        else:
            print("    ❌ Metrics still identical (NOT FIXED)")

if __name__ == "__main__":
    fix_evaluation_results()