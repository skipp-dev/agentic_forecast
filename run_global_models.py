#!/usr/bin/env python3
"""
Global Model Training and Evaluation Script

Trains global time series models (NHITS-style) that learn across multiple symbols.
Part of Phase 2 implementation for advanced forecasting.

Usage:
    python run_global_models.py --symbols AAPL MSFT GOOGL --experiment baseline
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import logging
import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agents.global_model_agent import GlobalModelAgent
from src.data_loader import DataLoader
from data.feature_store import TimeSeriesFeatureStore

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/global_models.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class GlobalModelTrainer:
    """
    Trainer for global time series models.
    """

    def __init__(self):
        """Initialize the global model trainer."""
        self.data_loader = DataLoader()
        # Use GPU if available and compatible, otherwise CPU
        import torch
        if torch.cuda.is_available():
            try:
                # Test CUDA compatibility by creating a small tensor
                torch.cuda.init()
                test_tensor = torch.randn(1, device='cuda')
                device = 'cuda'
                logger.info("Using CUDA GPU for training")
            except Exception as e:
                logger.warning(f"CUDA GPU not compatible: {e}. Falling back to CPU.")
                device = 'cpu'
        else:
            device = 'cpu'
            logger.info("Using CPU for training")
        
        self.agent = GlobalModelAgent(model_type='nhits', device=device)
        self.feature_store = TimeSeriesFeatureStore(store_path='data/feature_store')
        logger.info(f"GlobalModelTrainer initialized on {device}")

    def load_symbol_data(self, symbols: List[str], experiment: str = "baseline") -> Dict[str, pd.DataFrame]:
        """
        Load feature data for multiple symbols.

        Args:
            symbols: List of symbols to load
            experiment: Feature experiment to use

        Returns:
            Dictionary mapping symbols to their feature DataFrames
        """
        symbol_data = {}

        for symbol in symbols:
            try:
                if experiment == "cross_asset":
                    # Try to load from feature store for cross-asset features
                    from data.feature_store import FeatureQuery
                    query = FeatureQuery(
                        symbol=symbol,
                        feature_names=[],  # Get all features
                    )
                    data = self.feature_store.retrieve_features(query)
                    if not data.empty:
                        symbol_data[symbol] = data
                        logger.info(f"Loaded {len(data)} rows of cross-asset features for {symbol}")
                    else:
                        # Fall back to baseline features
                        logger.warning(f"Cross-asset features not found for {symbol}, falling back to baseline")
                        features_path = Path(f"data/processed/baseline/{symbol}_features.parquet")
                        if features_path.exists():
                            data = pd.read_parquet(features_path)
                            symbol_data[symbol] = data
                            logger.info(f"Loaded {len(data)} rows of baseline features for {symbol}")
                        else:
                            logger.warning(f"Baseline features not found for {symbol}: {features_path}")
                else:
                    # Load processed features from parquet files
                    features_path = Path(f"data/processed/{experiment}/{symbol}_features.parquet")
                    if features_path.exists():
                        data = pd.read_parquet(features_path)
                        symbol_data[symbol] = data
                        logger.info(f"Loaded {len(data)} rows of features for {symbol}")
                    else:
                        logger.warning(f"Features not found for {symbol}: {features_path}")

            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {e}")

        return symbol_data

    def train_global_model(self, symbols: List[str], experiment: str = "baseline",
                          model_save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Train a global model using data from multiple symbols.

        Args:
            symbols: List of symbols to use for training
            experiment: Feature experiment to use
            model_save_path: Path to save the trained model

        Returns:
            Training results and metrics
        """
        logger.info(f"Training global model with symbols: {symbols}")

        # Load data for all symbols
        symbol_data = self.load_symbol_data(symbols, experiment)

        if not symbol_data:
            raise ValueError("No valid symbol data found")

        # Prepare global dataset
        train_loader, val_loader = self.agent.prepare_global_dataset(symbol_data)

        # Get input size from first batch
        sample_batch = next(iter(train_loader))
        input_size = sample_batch[0].shape[2]  # (batch, seq_len, features)

        logger.info(f"Input size: {input_size} features")

        # Train the model
        history = self.agent.train_global_model(train_loader, val_loader, input_size)

        # Save model if path provided
        if model_save_path:
            Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
            self.agent.save_model(model_save_path)
            logger.info(f"Model saved to {model_save_path}")

        # Calculate final metrics
        final_metrics = {
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'final_val_mae': history['val_mae'][-1],
            'best_val_loss': min(history['val_loss']),
            'best_val_mae': min(history['val_mae']),
            'training_epochs': len(history['train_loss'])
        }

        logger.info("Global model training completed")
        logger.info(f"Final metrics: {final_metrics}")

        return {
            'history': history,
            'metrics': final_metrics,
            'model_info': self.agent.get_model_info(),
            'symbols_used': symbols,
            'experiment': experiment
        }

    def evaluate_global_model(self, symbols: List[str], experiment: str = "baseline",
                             model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate the global model on test data.

        Args:
            symbols: List of symbols to evaluate on
            experiment: Feature experiment to use
            model_path: Path to load the model from

        Returns:
            Evaluation results
        """
        if model_path and Path(model_path).exists():
            self.agent.load_model(model_path)
            logger.info(f"Loaded model from {model_path}")
        elif self.agent.model is None:
            raise ValueError("No model available for evaluation")

        # Load data for evaluation
        symbol_data = self.load_symbol_data(symbols, experiment)

        results = {}

        for symbol, data in symbol_data.items():
            try:
                # Use same feature filtering as training
                feature_cols = [
                    'open', 'high', 'low', 'close', 'volume',
                    'returns', 'log_returns', 'sma_5', 'sma_20',
                    'volatility', 'volume_sma'
                ]
                available_cols = [col for col in feature_cols if col in data.columns]

                if not available_cols:
                    logger.warning(f"No valid features for {symbol}, skipping evaluation")
                    continue

                # Generate predictions
                predictions = self.agent.predict(data, feature_cols=available_cols)

                if len(predictions) == 0:
                    logger.warning(f"No predictions generated for {symbol}")
                    continue

                # Get actual values for comparison
                actuals = data['close'].iloc[self.agent.lookback:].values[:len(predictions)]

                # Calculate metrics
                mae = np.mean(np.abs(predictions - actuals))
                mse = np.mean((predictions - actuals) ** 2)
                rmse = np.sqrt(mse)

                # Directional accuracy - handle edge cases
                if len(actuals) > 1 and len(predictions) > 1:
                    actual_returns = np.diff(actuals)
                    pred_returns = np.diff(predictions)
                    directional_acc = np.mean((actual_returns * pred_returns) > 0)
                else:
                    directional_acc = 0.0  # Not enough data for directional accuracy

                results[symbol] = {
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'directional_accuracy': directional_acc,
                    'num_predictions': len(predictions)
                }

                logger.info(f"{symbol} - MAE: {mae:.4f}, DA: {directional_acc:.2f}")

            except Exception as e:
                logger.error(f"Error evaluating {symbol}: {e}")

        # Aggregate results
        if results:
            avg_metrics = {
                'avg_mae': np.mean([r['mae'] for r in results.values()]),
                'avg_mse': np.mean([r['mse'] for r in results.values()]),
                'avg_rmse': np.mean([r['rmse'] for r in results.values()]),
                'avg_directional_accuracy': np.mean([r['directional_accuracy'] for r in results.values()])
            }
            results['aggregate'] = avg_metrics
            logger.info(f"Aggregate metrics: {avg_metrics}")

        return results

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Global Model Training and Evaluation")
    parser.add_argument('--symbols', nargs='+', required=False,
                       help='Symbols to use for training/evaluation')
    parser.add_argument('--watchlist', default='watchlist_ibkr.csv',
                       help='Path to watchlist CSV file to load all symbols')
    parser.add_argument('--experiment', default='cross_asset',
                       help='Feature experiment to use')
    parser.add_argument('--action', choices=['train', 'evaluate', 'both'], default='both',
                       help='Action to perform')
    parser.add_argument('--model-path', default='models/global/nhits_global_model.pth',
                       help='Path to save/load model')
    parser.add_argument('--output-dir', default='results/global_models',
                       help='Output directory for results')
    parser.add_argument('--max-symbols', type=int, default=None,
                       help='Maximum number of symbols to use (for testing)')

    args = parser.parse_args()

    # Load symbols from watchlist if not provided
    if not args.symbols:
        try:
            import pandas as pd
            df = pd.read_csv(args.watchlist)
            symbol_col = 'Symbol' if 'Symbol' in df.columns else df.columns[0]
            all_symbols = df[symbol_col].dropna().unique().tolist()
            args.symbols = all_symbols[:args.max_symbols] if args.max_symbols else all_symbols
            logger.info(f"Loaded {len(args.symbols)} symbols from {args.watchlist}")
        except Exception as e:
            logger.error(f"Error loading watchlist: {e}")
            return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize trainer
    trainer = GlobalModelTrainer()

    results = {}

    if args.action in ['train', 'both']:
        logger.info("Starting global model training...")

        # Train the model
        training_results = trainer.train_global_model(
            args.symbols,
            args.experiment,
            args.model_path
        )

        results['training'] = training_results

        # Save training results
        training_file = output_dir / f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(training_file, 'w') as f:
            # Convert numpy values to native Python types for JSON serialization
            json_results = {
                'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                           for k, v in training_results['metrics'].items()},
                'model_info': training_results['model_info'],
                'symbols_used': training_results['symbols_used'],
                'experiment': training_results['experiment']
            }
            import json
            json.dump(json_results, f, indent=2)

        logger.info(f"Training results saved to {training_file}")

    if args.action in ['evaluate', 'both']:
        logger.info("Starting global model evaluation...")

        # Evaluate the model
        eval_results = trainer.evaluate_global_model(
            args.symbols,
            args.experiment,
            args.model_path
        )

        results['evaluation'] = eval_results

        # Save evaluation results
        eval_file = output_dir / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(eval_file, 'w') as f:
            # Convert numpy values for JSON
            json_eval = {}
            for symbol, metrics in eval_results.items():
                json_eval[symbol] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                                   for k, v in metrics.items()}
            import json
            json.dump(json_eval, f, indent=2)

        logger.info(f"Evaluation results saved to {eval_file}")

    logger.info("Global model training/evaluation completed successfully")

if __name__ == "__main__":
    main()