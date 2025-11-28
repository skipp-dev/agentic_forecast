#!/usr/bin/env python3
"""
Analyze the evaluation results to identify the metrics bug.
"""
import pandas as pd
import numpy as np

def analyze_evaluation_results():
    # Load the evaluation results
    df = pd.read_csv('data/metrics/evaluation_results_baseline_latest.csv')

    print("=== EVALUATION RESULTS ANALYSIS ===")
    print(f"Total records: {len(df)}")
    print(f"Unique symbols: {df['symbol'].nunique()}")
    print(f"Horizons: {sorted(df['target_horizon'].unique())}")
    print(f"Model types: {df['model_type'].unique()}")

    # Check for identical metrics across symbols
    print("\n=== METRICS ANALYSIS ===")

    for horizon in sorted(df['target_horizon'].unique()):
        h_df = df[df['target_horizon'] == horizon]
        print(f"\nHorizon {horizon}:")
        print(f"  Symbols: {len(h_df)}")
        print(f"  Unique MAE values: {len(h_df['mae'].unique())}")
        print(f"  Unique RMSE values: {len(h_df['rmse'].unique())}")
        print(f"  Unique MAPE values: {len(h_df['mape'].unique())}")
        print(f"  Unique directional_accuracy values: {len(h_df['directional_accuracy'].unique())}")

        # Show sample values
        sample = h_df.head(3)[['symbol', 'mae', 'rmse', 'mape', 'directional_accuracy']]
        print("  Sample values:")
        for _, row in sample.iterrows():
            print(f"    {row['symbol']}: MAE={row['mae']:.6f}, RMSE={row['rmse']:.6f}, MAPE={row['mape']}, DIR_ACC={row['directional_accuracy']}")

    # Check if MAPE is exactly 1.0 (indicating clipping/fallback)
    mape_values = df['mape'].unique()
    print(f"\nMAPE values found: {mape_values}")
    if 1.0 in mape_values:
        print("⚠️  MAPE = 1.0 detected - this suggests clipping or fallback behavior")

    # Check directional accuracy
    dir_acc_values = df['directional_accuracy'].unique()
    print(f"Directional accuracy values: {dir_acc_values}")
    if 0.0 in dir_acc_values:
        print("⚠️  Directional accuracy = 0.0 for all symbols - this suggests a calculation bug")

if __name__ == "__main__":
    analyze_evaluation_results()