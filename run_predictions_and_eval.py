#!/usr/bin/env python3
"""
Predictions and Evaluation Pipeline for Agentic Forecasting System

This script generates predictions and evaluates model performance:
- Generate predictions for all trained models
- Calculate evaluation metrics (MAPE, MAE, DA)
- Store results in data/metrics/

Usage:
    python run_predictions_and_eval.py [--symbols SYMBOL1 SYMBOL2] [--experiment EXPERIMENT]
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging
import yaml
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import joblib

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_loader import DataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/predictions_eval.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Ensure console handler uses UTF-8 encoding
for handler in logging.getLogger().handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setStream(sys.stdout)
        # Try to set encoding if possible
        if hasattr(handler, 'stream') and hasattr(handler.stream, 'reconfigure'):
            try:
                handler.stream.reconfigure(encoding='utf-8')
            except Exception:
                pass  # Fallback to default if reconfigure not available

logger = logging.getLogger(__name__)

class PredictionEvaluator:
    """
    Generate predictions and evaluate model performance.
    """

    def __init__(self):
        """Initialize prediction evaluator."""
        self.data_loader = DataLoader()
        self.models = {}
        self.scalers = {}

        logger.info("PredictionEvaluator initialized")

    def load_trained_models(self, models_dir: str = "models/baseline") -> pd.DataFrame:
        """
        Load trained model configurations.

        Args:
            models_dir: Directory containing trained models

        Returns:
            DataFrame with model configurations
        """
        models_path = Path(models_dir) / "models_info.csv"
        if not models_path.exists():
            logger.error(f"Models info file not found: {models_path}")
            return pd.DataFrame()

        models_df = pd.read_csv(models_path)
        logger.info(f"Loaded {len(models_df)} trained model configurations")

        # Load actual model objects for sklearn models
        for _, model_info in models_df.iterrows():
            if 'model_key' in model_info and pd.notna(model_info['model_key']):
                model_key = model_info['model_key']

                # Load model
                model_path = Path(models_dir) / f"{model_key}.joblib"
                if model_path.exists():
                    try:
                        self.models[model_key] = joblib.load(model_path)
                    except Exception as e:
                        logger.error(f"Failed to load model {model_key}: {e}")

                # Load scaler
                scaler_path = Path(models_dir) / f"{model_key}_scaler.joblib"
                if scaler_path.exists():
                    try:
                        self.scalers[model_key] = joblib.load(scaler_path)
                    except Exception as e:
                        logger.error(f"Failed to load scaler {model_key}: {e}")

        return models_df

    def generate_naive_predictions(self, symbol: str, features_df: pd.DataFrame,
                                 target_horizon: int) -> pd.Series:
        """
        Generate naive baseline predictions.

        Args:
            symbol: Stock symbol
            features_df: Feature data
            target_horizon: Prediction horizon

        Returns:
            Series of predictions
        """
        # For naive model, predict zero return (no change from current price)
        # This is a proper baseline that doesn't use future data
        predictions = pd.Series(0.0, index=features_df.index)

        return predictions

    def generate_sklearn_predictions(self, symbol: str, features_df: pd.DataFrame,
                                   model_info: pd.Series) -> pd.Series:
        """
        Generate predictions from sklearn model.

        Args:
            symbol: Stock symbol
            features_df: Feature data
            model_info: Model configuration

        Returns:
            Series of predictions
        """
        model_key = model_info['model_key']
        target_horizon = model_info['target_horizon']

        if model_key not in self.models:
            logger.error(f"Model {model_key} not loaded")
            return pd.Series(dtype=float)

        model = self.models[model_key]
        scaler = self.scalers.get(model_key)

        # Prepare features for prediction
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', f'returns_{target_horizon}d']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]

        # Create prediction features (rolling window)
        lookback_window = 30  # Same as training
        predictions = []

        for i in range(lookback_window, len(features_df)):
            # Get feature window
            feature_window = features_df[feature_cols].iloc[i-lookback_window:i].values.flatten()

            # Scale features
            if scaler:
                feature_window = scaler.transform(feature_window.reshape(1, -1))

            # Make prediction
            pred = model.predict(feature_window.reshape(1, -1))[0]
            predictions.append((features_df.index[i], pred))

        # Convert to series
        if predictions:
            pred_df = pd.DataFrame(predictions, columns=['date', 'prediction'])
            pred_df = pred_df.set_index('date')
            pred_series = pred_df['prediction']
        else:
            pred_series = pd.Series(dtype=float)

        return pred_series

    def evaluate_predictions(self, actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
        """
        Calculate evaluation metrics.

        Args:
            actual: Actual values
            predicted: Predicted values

        Returns:
            Dictionary of metrics
        """
        # Align the series
        combined = pd.concat([actual, predicted], axis=1, keys=['actual', 'predicted']).dropna()

        if len(combined) == 0:
            return {
                'mae': np.nan,
                'mse': np.nan,
                'rmse': np.nan,
                'mape': np.nan,
                'directional_accuracy': np.nan,
                'n_samples': 0
            }

        actual_vals = combined['actual'].values
        pred_vals = combined['predicted'].values

        # Ensure we have arrays, not scalars
        if actual_vals.ndim == 0:
            actual_vals = np.array([actual_vals])
        if pred_vals.ndim == 0:
            pred_vals = np.array([pred_vals])

        # Calculate metrics
        mae = mean_absolute_error(actual_vals, pred_vals)
        mse = mean_squared_error(actual_vals, pred_vals)
        rmse = np.sqrt(mse)

        # Handle MAPE carefully (avoid division by zero)
        non_zero_mask = actual_vals != 0
        if non_zero_mask.any():
            mape = mean_absolute_percentage_error(actual_vals[non_zero_mask], pred_vals[non_zero_mask])
        else:
            mape = np.nan

        # Directional accuracy (sign of prediction matches sign of actual)
        actual_direction = np.sign(actual_vals)
        pred_direction = np.sign(pred_vals)
        directional_accuracy = np.mean(actual_direction == pred_direction)

        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'n_samples': len(combined)
        }

    def generate_predictions_for_model(self, model_info: pd.Series,
                                     experiment: str = "baseline") -> Tuple[pd.Series, pd.Series]:
        """
        Generate predictions for a single model.

        Args:
            model_info: Model configuration
            experiment: Feature experiment

        Returns:
            Tuple of (predictions, actuals)
        """
        symbol = model_info['symbol']
        model_type = model_info['model_type']
        target_horizon = model_info['target_horizon']

        # Load feature data
        feature_data = self.data_loader.load_feature_data(
            feature_set=experiment,
            symbols=[symbol]
        )

        if symbol not in feature_data:
            logger.error(f"No feature data found for {symbol}")
            return pd.Series(dtype=float), pd.Series(dtype=float)

        features_df = feature_data[symbol]

        # Generate predictions
        if model_type == 'naive':
            predictions = self.generate_naive_predictions(symbol, features_df, target_horizon)
        elif model_type in ['linear', 'ridge', 'rf']:
            predictions = self.generate_sklearn_predictions(symbol, features_df, model_info)
        else:
            logger.error(f"Unknown model type: {model_type}")
            return pd.Series(dtype=float), pd.Series(dtype=float)

        # Get actual values for evaluation
        target_col = f'returns_{target_horizon}d'
        if target_col in features_df.columns:
            actuals = features_df[target_col].shift(-target_horizon)
        else:
            actuals = pd.Series(dtype=float)

        return predictions, actuals

    def run_predictions_and_evaluation(self, experiment: str = "baseline",
                                     models_dir: str = "models/baseline",
                                     max_models: Optional[int] = None) -> pd.DataFrame:
        """
        Run complete predictions and evaluation pipeline.

        Args:
            experiment: Feature experiment
            models_dir: Directory with trained models

        Returns:
            DataFrame with evaluation results
        """
        logger.info(f"Starting predictions and evaluation for experiment: {experiment}")

        # Load trained models
        models_df = self.load_trained_models(models_dir)
        if models_df.empty:
            logger.error("No trained models found")
            return pd.DataFrame()

        logger.info(f"Evaluating {len(models_df)} trained models")

        evaluation_results = []

        # Limit models for testing if specified
        if max_models:
            models_df = models_df.head(max_models)
            logger.info(f"Limited to first {max_models} models for testing")

        for _, model_info in models_df.iterrows():
            symbol = model_info['symbol']
            model_type = model_info['model_type']
            target_horizon = model_info['target_horizon']

            try:
                logger.info(f"Generating predictions for {symbol} {model_type} {target_horizon}d")

                # Generate predictions
                predictions, actuals = self.generate_predictions_for_model(model_info, experiment)

                # Evaluate predictions
                metrics = self.evaluate_predictions(actuals, predictions)

                # Store results
                result = {
                    'symbol': symbol,
                    'model_type': model_type,
                    'target_horizon': target_horizon,
                    'experiment': experiment,
                    'predictions_count': len(predictions),
                    'evaluation_timestamp': datetime.now(),
                    **metrics
                }

                evaluation_results.append(result)

                logger.info(f"[SUCCESS] {symbol} {model_type} {target_horizon}d: "
                           f"MAE={metrics['mae']:.4f}, DA={metrics['directional_accuracy']:.3f}")

            except Exception as e:
                logger.error(f"[FAILED] Failed to evaluate {symbol} {model_type} {target_horizon}d: {e}")

        # Convert to DataFrame
        results_df = pd.DataFrame(evaluation_results)

        # Save results
        self.save_evaluation_results(results_df, experiment)

        logger.info("\nEvaluation Summary:")
        logger.info("=" * 80)
        logger.info(f"Total evaluations: {len(results_df)}")

        # Summary by model type
        if not results_df.empty:
            summary = results_df.groupby('model_type').agg({
                'mae': 'mean',
                'directional_accuracy': 'mean',
                'n_samples': 'sum'
            }).round(4)

            logger.info("\nAverage performance by model type:")
            for idx, metrics in summary.iterrows():
                model_type = idx if isinstance(idx, str) else str(idx)
                logger.info(f"  {model_type}: MAE={metrics['mae']:.4f}, "
                           f"DA={metrics['directional_accuracy']:.3f} "
                           f"({int(metrics['n_samples'])} samples)")

        return results_df

    def save_evaluation_results(self, results_df: pd.DataFrame, experiment: str):
        """Save evaluation results to disk."""
        output_dir = Path("data/metrics")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results_{experiment}_{timestamp}.csv"
        filepath = output_dir / filename
        results_df.to_csv(filepath, index=False)

        # Save latest results (overwrite)
        latest_filepath = output_dir / f"evaluation_results_{experiment}_latest.csv"
        results_df.to_csv(latest_filepath, index=False)

        logger.info(f"Saved evaluation results to {filepath}")

        # Generate summary report
        self.generate_summary_report(results_df, experiment)

    def generate_summary_report(self, results_df: pd.DataFrame, experiment: str):
        """Generate a summary report of evaluation results."""
        if results_df.empty:
            return

        output_dir = Path("data/metrics")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"evaluation_summary_{experiment}_{timestamp}.txt"

        with open(report_path, 'w') as f:
            f.write("Agentic Forecasting System - Model Evaluation Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Experiment: {experiment}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("Overall Statistics:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total evaluations: {len(results_df)}\n")
            f.write(f"Symbols evaluated: {results_df['symbol'].nunique()}\n")
            f.write(f"Model types: {results_df['model_type'].nunique()}\n\n")

            # Performance by model type
            f.write("Performance by Model Type:\n")
            f.write("-" * 30 + "\n")
            for model_type in results_df['model_type'].unique():
                model_data = results_df[results_df['model_type'] == model_type]
                f.write(f"\n{model_type.upper()}:\n")
                f.write(f"  MAE: {model_data['mae'].mean():.4f} ± {model_data['mae'].std():.4f}\n")
                f.write(f"  Directional Accuracy: {model_data['directional_accuracy'].mean():.3f} ± {model_data['directional_accuracy'].std():.3f}\n")
                f.write(f"  Total Samples: {len(model_data)}\n")

            # Performance by horizon
            f.write("\n\nPerformance by Horizon:\n")
            f.write("-" * 30 + "\n")
            for horizon in sorted(results_df['target_horizon'].unique()):
                horizon_data = results_df[results_df['target_horizon'] == horizon]
                f.write(f"\n{horizon} Day Horizon:\n")
                f.write(f"  MAE: {horizon_data['mae'].mean():.4f}\n")
                f.write(f"  Directional Accuracy: {horizon_data['directional_accuracy'].mean():.3f}\n")
                f.write(f"  Total Samples: {len(horizon_data)}\n")

            # Best performing models
            f.write("\n\nBest Performing Models (by Directional Accuracy):\n")
            f.write("-" * 50 + "\n")
            best_models = results_df.nlargest(10, 'directional_accuracy')[
                ['symbol', 'model_type', 'target_horizon', 'directional_accuracy', 'mae']
            ]
            for _, model in best_models.iterrows():
                f.write(f"{model['symbol']} {model['model_type']} {model['target_horizon']}d: "
                       f"DA={model['directional_accuracy']:.3f}, MAE={model['mae']:.4f}\n")

        logger.info(f"Generated summary report: {report_path}")

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Run predictions and evaluation pipeline')
    parser.add_argument('--experiment', default='baseline', help='Feature experiment')
    parser.add_argument('--models-dir', default='models/baseline', help='Directory with trained models')
    parser.add_argument('--max-models', type=int, default=None, help='Maximum number of models to evaluate (for testing)')

    args = parser.parse_args()

    try:
        logger.info("Starting predictions and evaluation pipeline...")

        # Initialize evaluator
        evaluator = PredictionEvaluator()

        # Run evaluation
        results_df = evaluator.run_predictions_and_evaluation(
            experiment=args.experiment,
            models_dir=args.models_dir,
            max_models=args.max_models
        )

        logger.info("\nPredictions and evaluation completed successfully! [SUCCESS]")

    except Exception as e:
        logger.error(f"Predictions and evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()