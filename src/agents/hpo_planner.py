import os
import sys
import pandas as pd
import logging
from typing import List, Dict, Any

from models.model_zoo import ModelZoo, DataSpec, HPOConfig, ModelTrainingResult

logger = logging.getLogger(__name__)

class HPOAgent:
    """
    Hyperparameter Optimization Agent to orchestrate model training and tuning.
    """
    def __init__(self, symbols: List[str], data_store: Dict[str, pd.DataFrame] = None, data_path: str = 'data/reference'):
        """
        Initializes the HPOAgent.

        Args:
            symbols (List[str]): A list of stock symbols to process.
            data_store (Dict[str, pd.DataFrame]): Dictionary containing data for each symbol.
            data_path (str): The path to the reference data (fallback).
        """
        self.symbols = symbols
        self.data_store = data_store
        self.data_path = data_path
        self.model_zoo = ModelZoo()
        self.results: Dict[str, Dict[str, ModelTrainingResult]] = {}

    def _load_data(self, symbol: str) -> pd.DataFrame:
        """Loads and prepares reference data for a given symbol."""
        if self.data_store and symbol in self.data_store:
            df = self.data_store[symbol].copy()
            # Handle dict format if passed from state
            if isinstance(df, dict):
                df = pd.DataFrame.from_dict(df, orient='index')
                df.index = pd.to_datetime(df.index)
            
            # Ensure required columns exist
            if 'close' in df.columns and 'y' not in df.columns:
                df = df.rename(columns={'close': 'y'})
            
            if 'ds' not in df.columns:
                df['ds'] = df.index
                
            df['unique_id'] = symbol
            return df

        file_path = os.path.join(self.data_path, f"{symbol}_reference_data.parquet")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found for symbol {symbol} at {file_path}")
        
        df = pd.read_parquet(file_path)
        
        # Transform data for NeuralForecast
        df = df.rename(columns={'date': 'ds', 'symbol': 'unique_id', 'close': 'y'})
        df['ds'] = pd.to_datetime(df['ds'])
        
        return df[['unique_id', 'ds', 'y']]

    def run_hpo_session(self):
        """
        Runs the HPO session for all symbols and model families.
        """
        logger.info("--- Starting HPO Session ---")

        for symbol in self.symbols:
            logger.info(f"--- Processing Symbol: {symbol} ---")
            try:
                df = self._load_data(symbol)
                
                if df.empty:
                    logger.warning(f"No data found for {symbol}, skipping.")
                    continue

                if len(df) < 60: # Ensure enough data for train/val
                     logger.warning(f"Insufficient data for {symbol} (len={len(df)}), skipping.")
                     continue

                train_df = df.iloc[:-30]
                val_df = df.iloc[-30:]

                data_spec = DataSpec(
                    target_col='y',
                    train_df=train_df,
                    val_df=val_df,
                    symbol_scope=symbol,
                    horizon=30
                )

                self.results[symbol] = {}

                # Define HPO configs
                hpo_config_small = HPOConfig(max_trials=1, max_epochs=2)
                hpo_config_large = HPOConfig(max_trials=1, max_epochs=3)

                # Train models
                logger.info(f"Training BaselineLinear for {symbol}...")
                self.results[symbol]['BaselineLinear'] = self.model_zoo.train_baseline_linear(data_spec)

                logger.info(f"Training AutoNHITS for {symbol}...")
                self.results[symbol]['AutoNHITS'] = self.model_zoo.train_autonhits(data_spec, hpo_config_small)

                logger.info(f"Training AutoTFT for {symbol}...")
                self.results[symbol]['AutoTFT'] = self.model_zoo.train_autotft(data_spec, hpo_config_large)

            except Exception as e:
                logger.error(f"Error processing symbol {symbol}: {e}", exc_info=True)

        self.print_summary()

    def print_summary(self):
        """Prints a summary of the HPO session results."""
        logger.info("--- HPO Session Summary ---")
        for symbol, results in self.results.items():
            logger.info(f"--- Results for {symbol} ---")
            for model_family, result in results.items():
                if result:
                    logger.info(f"  - {model_family}:")
                    logger.info(f"    - Best MAPE: {result.best_val_mape:.4f}")
                    logger.info(f"    - Best MAE:  {result.best_val_mae:.4f}")
                    logger.info(f"    - Best Model ID: {result.best_model_id}")
        logger.info("--------------------------")

if __name__ == '__main__':
    # Example usage
    agent = HPOAgent(symbols=["AAPL", "NVDA", "TSLA"])
    agent.run_hpo_session()
