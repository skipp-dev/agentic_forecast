#!/usr/bin/env python3
"""
Human-Readable Forecast Results Display
Shows forecast predictions in a clear, readable format for your 567 symbols.
"""

import pandas as pd
import os
from datetime import datetime
import numpy as np

def format_currency(value):
    """Format numerical values as currency"""
    try:
        return f"${float(value):.2f}"
    except:
        return str(value)

def format_percentage(value):
    """Format numerical values as percentages"""
    try:
        return f"{float(value)*100:.2f}%"
    except:
        return str(value)

def load_forecast_results():
    """Load and combine all forecast results"""
    results = {}

    # Load parquet forecast files
    forecast_files = [
        'results/hpo/v2_on/val_predictions.parquet',
        'results/hpo/v2_off/val_predictions.parquet',
        'results/hpo/hpo_cross_asset_on/val_predictions.parquet',
        'results/hpo/hpo_cross_asset_off/val_predictions.parquet'
    ]

    for file_path in forecast_files:
        if os.path.exists(file_path):
            try:
                df = pd.read_parquet(file_path)
                file_name = os.path.basename(file_path).replace('.parquet', '')
                results[file_name] = df
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    return results

def display_forecast_summary():
    """Display a human-readable summary of forecast results"""
    print("=" * 80)
    print("🤖 AGENTIC FORECAST SYSTEM - HUMAN READABLE RESULTS")
    print("=" * 80)
    print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    results = load_forecast_results()

    if not results:
        print("❌ No forecast results found!")
        return

    # Combine all results
    all_predictions = []
    for source, df in results.items():
        df_copy = df.copy()
        df_copy['source'] = source
        all_predictions.append(df_copy)

    if all_predictions:
        combined_df = pd.concat(all_predictions, ignore_index=True)

        # Group by symbol and show latest predictions
        print("📊 FORECAST PREDICTIONS BY SYMBOL")
        print("-" * 80)

        for symbol in sorted(combined_df['symbol'].unique()):
            symbol_data = combined_df[combined_df['symbol'] == symbol].copy()

            # Get the most recent prediction
            latest_pred = symbol_data.iloc[-1] if len(symbol_data) > 0 else None

            if latest_pred is not None:
                print(f"\n🏢 SYMBOL: {symbol}")
                print(f"   📅 Date: {latest_pred['ds'].strftime('%Y-%m-%d') if hasattr(latest_pred['ds'], 'strftime') else latest_pred['ds']}")
                print(f"   💰 Actual Price: {format_currency(latest_pred['y_true'])}")
                print(f"   🔮 Predicted Price: {format_currency(latest_pred['y_pred'])}")

                # Calculate prediction accuracy
                if pd.notna(latest_pred['y_true']) and pd.notna(latest_pred['y_pred']):
                    error = abs(float(latest_pred['y_true']) - float(latest_pred['y_pred']))
                    error_pct = (error / abs(float(latest_pred['y_true']))) * 100
                    print(f"   📈 Prediction Error: {format_currency(error)} ({error_pct:.2f}%)")

                # Show peer analysis if available
                if 'peer_mean_ret_1d' in latest_pred and pd.notna(latest_pred['peer_mean_ret_1d']):
                    print(f"   👥 Peer Avg Return (1D): {format_percentage(latest_pred['peer_mean_ret_1d'])}")

                if 'peer_mean_ret_5d' in latest_pred and pd.notna(latest_pred['peer_mean_ret_5d']):
                    print(f"   👥 Peer Avg Return (5D): {format_percentage(latest_pred['peer_mean_ret_5d'])}")

                print(f"   🎯 Model: {latest_pred.get('model_family', 'Unknown')}")
                print(f"   📁 Source: {latest_pred['source']}")

        print("\n" + "=" * 80)
        print("📈 SUMMARY STATISTICS")
        print("-" * 80)

        # Calculate overall statistics
        total_predictions = len(combined_df)
        unique_symbols = combined_df['symbol'].nunique()
        avg_error_pct = 0

        if 'y_true' in combined_df.columns and 'y_pred' in combined_df.columns:
            valid_predictions = combined_df.dropna(subset=['y_true', 'y_pred'])
            if len(valid_predictions) > 0:
                errors = []
                for _, row in valid_predictions.iterrows():
                    if pd.notna(row['y_true']) and pd.notna(row['y_pred']) and float(row['y_true']) != 0:
                        error_pct = abs(float(row['y_true']) - float(row['y_pred'])) / abs(float(row['y_true'])) * 100
                        errors.append(error_pct)

                if errors:
                    avg_error_pct = np.mean(errors)
                    print(f"🎯 Total Predictions: {total_predictions}")
                    print(f"🏢 Unique Symbols: {unique_symbols}")
                    print(f"📊 Average Prediction Error: {avg_error_pct:.2f}%")
                    print(f"✅ Valid Predictions: {len(valid_predictions)}")

        print("\n" + "=" * 80)
        print("💡 INTERPRETATION GUIDE")
        print("-" * 80)
        print("• Lower prediction error % = More accurate forecasts")
        print("• Compare predicted vs actual prices for each symbol")
        print("• Peer returns show how your symbol compares to similar stocks")
        print("• Multiple sources provide ensemble forecasting reliability")
        print("=" * 80)

def display_evaluation_results():
    """Display evaluation metrics in human-readable format"""
    print("\n" + "=" * 80)
    print("📊 MODEL EVALUATION RESULTS - ALL 567+ SYMBOLS")
    print("=" * 80)

    eval_file = 'data/metrics/evaluation_results_baseline_latest.csv'
    if os.path.exists(eval_file):
        try:
            df = pd.read_csv(eval_file)
            print(f"📄 Loaded {len(df)} evaluation records for {df['symbol'].nunique()} unique symbols")

            if len(df) > 0:
                # Group by symbol and show summary
                symbol_summary = df.groupby('symbol').agg({
                    'mae': ['mean', 'min', 'max'],
                    'rmse': 'mean',
                    'predictions_count': 'first',
                    'target_horizon': 'first'
                }).round(4)

                print(f"\n🏆 SYSTEM OVERVIEW:")
                print(f"   🎯 Total Symbols: {df['symbol'].nunique()}")
                print(f"   📊 Total Evaluation Records: {len(df):,}")
                print(f"   📈 Average MAE: {df['mae'].mean():.4f}")
                print(f"   📏 Average RMSE: {df['rmse'].mean():.4f}")
                print(f"   ⏰ Forecast Horizons: {sorted(df['target_horizon'].unique())}")
                print(f"   🤖 Models Used: {df['model_type'].unique()}")

                # Show sample of actual symbols
                print(f"\n📋 SAMPLE SYMBOLS EVALUATED:")
                sample_symbols = df['symbol'].drop_duplicates().head(20).tolist()
                for i, symbol in enumerate(sample_symbols, 1):
                    symbol_data = df[df['symbol'] == symbol]
                    avg_mae = symbol_data['mae'].mean()
                    horizons = sorted(symbol_data['target_horizon'].unique())
                    print(f"   {i:2d}. {symbol:<12} | MAE: {avg_mae:.4f} | Horizons: {horizons}")

                print(f"\n💡 NOTE: MAPE values appear to be placeholder (1.0 = 100%)")
                print(f"         Actual performance measured by MAE and RMSE metrics")

        except Exception as e:
            print(f"❌ Error loading evaluation results: {e}")
    else:
        print("❌ No evaluation results file found")

if __name__ == "__main__":
    display_forecast_summary()
    display_evaluation_results()