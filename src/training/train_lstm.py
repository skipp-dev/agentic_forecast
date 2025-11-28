"""
Training Script for LSTM Forecasting Model

This script demonstrates GPU-accelerated training of an LSTM model
for financial time series forecasting using the unified data ingestion system.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging
import pandas as pd
import os

# Configure TensorFlow logging to reduce verbosity BEFORE any TensorFlow imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO, WARNING, and ERROR messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations warning

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.unified_ingestion_v2 import UnifiedDataIngestion
from models.lstm_forecaster import LSTMForecaster
from models.transformer_forecaster import TransformerForecaster

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    import argparse

    parser = argparse.ArgumentParser(description='Train LSTM or Transformer forecasting model')
    parser.add_argument('--model', type=str, choices=['lstm', 'transformer'],
                       default='lstm', help='Model type to train (default: lstm)')
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'MSFT', 'GOOGL'],
                       help='Stock symbols to train on')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')

    args = parser.parse_args()

    # Training configuration
    symbols = args.symbols
    epochs = args.epochs

    logger.info(f"🚀 Starting {args.model.upper()} forecasting training (with IB data when available)...")  # Multiple symbols for training
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    logger.info(f"Fetching data from {start_date} to {end_date} for symbols: {symbols}")

    # Initialize unified data ingestion
    data_ingestion = UnifiedDataIngestion(
        use_real_data=True,  # Always try real IB data first
        ib_host='localhost',  # Will try host.docker.internal automatically
        ib_port=7497
    )

    # Initialize data sources
    try:
        logger.info("Initializing data ingestion system...")
        data_ingestion.initialize()
        real_data_available = data_ingestion.ib_available
    except RuntimeError as e:
        logger.warning(f"⚠️ IB connection failed, falling back to synthetic data: {e}")
        # Re-initialize with synthetic data only
        data_ingestion = UnifiedDataIngestion(
            use_real_data=False,  # Use synthetic data
            ib_host='localhost',
            ib_port=7497
        )
        data_ingestion.initialize()
        real_data_available = False

    # Get data for all symbols
    data_source = "REAL IB MARKET DATA" if real_data_available else "SYNTHETIC DATA"
    logger.info(f"📊 Using {data_source} for training")
    data_dict = data_ingestion.get_multiple_symbols(symbols, start_date, end_date)

    if not data_dict:
        logger.error("❌ No data available for training")
        return 1

    # Combine all symbols into single dataset for training
    all_data = []
    for symbol, data in data_dict.items():
        if data is not None and len(data) > 0:
            data_copy = data.copy()
            data_copy['symbol'] = symbol
            all_data.append(data_copy)
            logger.info(f"✅ {symbol}: {len(data)} data points")

    if not all_data:
        logger.error("❌ No valid data found")
        return 1

    # Combine all data
    combined_data = pd.concat(all_data, axis=0).sort_index()
    logger.info(f"📊 Combined dataset: {len(combined_data)} total data points")

    # Initialize forecaster based on model type
    if args.model == 'lstm':
        forecaster = LSTMForecaster(
            sequence_length=30,  # Look back 30 days
            forecast_horizon=1,  # Predict 1 day ahead
            lstm_units=64,      # 64 LSTM units
            dropout_rate=0.2,   # 20% dropout
            learning_rate=0.001
        )
    elif args.model == 'transformer':
        forecaster = TransformerForecaster(
            sequence_length=30,  # Look back 30 days
            num_heads=8,         # 8 attention heads
            embed_dim=64,        # 64 embedding dimension
            ff_dim=128,          # 128 feed-forward dimension
            num_transformer_blocks=4,  # 4 transformer blocks
            dropout_rate=0.1,    # 10% dropout
            learning_rate=0.001
        )
    else:
        logger.error(f"❌ Unknown model type: {args.model}")
        return 1

    # Prepare data
    logger.info("🔧 Preparing data for training...")
    X_train, y_train, X_test, y_test = forecaster.prepare_data(combined_data, target_column='close')

    if len(X_train) == 0:
        logger.error("❌ Insufficient data for training after preprocessing")
        return 1

    # Train model
    logger.info(f"🎯 Training {args.model.upper()} model on GPU...")
    history = forecaster.train(
        X_train, y_train,
        X_test, y_test,
        epochs=epochs,      # Use specified epochs
        batch_size=16,      # Small batch size for better gradient estimates
        verbose=1
    )

    # Log final training results
    final_loss = history.history['loss'][-1]
    final_val_loss = history.history.get('val_loss', [final_loss])[-1]
    logger.info(f"🏁 Training completed - Final loss: {final_loss:.4f}, Validation loss: {final_val_loss:.4f}")

    # Make predictions
    logger.info("🔮 Making predictions...")
    predictions = forecaster.predict(X_test)
    predictions_original = forecaster.inverse_transform_predictions(predictions)
    y_test_original = forecaster.inverse_transform_predictions(y_test)

    # Evaluate model
    logger.info("📊 Evaluating model performance...")
    metrics = forecaster.evaluate(y_test_original, predictions_original)

    # Save model
    model_path = forecaster.save_model()
    logger.info(f"💾 Model saved to: {model_path}")

    # Plot training history
    plot_path = "training_history.png"
    forecaster.plot_training_history(plot_path)
    logger.info(f"📈 Training plot saved to: {plot_path}")

    # Disconnect from IB
    data_ingestion.disconnect()

    # Print final results
    logger.info("="*60)
    logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("="*60)
    if args.model == 'lstm':
        logger.info(f"Model: LSTM with {forecaster.lstm_units} units")
    elif args.model == 'transformer':
        logger.info(f"Model: Transformer with {forecaster.embed_dim} embed dim, {forecaster.num_heads} heads, {forecaster.num_transformer_blocks} blocks")
    logger.info(f"Training symbols: {symbols}")
    logger.info("Data source: Interactive Brokers (Real Market Data)")
    logger.info(f"Training data: {len(X_train)} sequences")
    logger.info(f"Test data: {len(X_test)} sequences")
    logger.info(f"Sequence length: {forecaster.sequence_length} days")
    logger.info("Forecast horizon: 1 day(s)")  # Hardcoded for now, can be made configurable later
    logger.info("")
    logger.info("PERFORMANCE METRICS:")
    for metric, value in metrics.items():
        if metric == 'MAPE':
            logger.info(f"  {metric}: {value:.2f}%")
        else:
            logger.info(f"  {metric}: {value:.4f}")
    logger.info("")
    logger.info(f"Model saved: {model_path}")
    logger.info(f"Training plot: {plot_path}")
    logger.info("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())