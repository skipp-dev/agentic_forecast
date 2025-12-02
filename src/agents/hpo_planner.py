import os
import sys
import pandas as pd
from typing import List, Dict, Any

from models.model_zoo import ModelZoo, DataSpec, HPOConfig, ModelTrainingResult

class HPOAgent:
    """
    Hyperparameter Optimization Agent to orchestrate model training and tuning.
    """
    def __init__(self, symbols: List[str], data_path: str = 'data/reference'):
        """
        Initializes the HPOAgent.

        Args:
            symbols (List[str]): A list of stock symbols to process.
            data_path (str): The path to the reference data.
        """
        self.symbols = symbols
        self.data_path = data_path
        self.model_zoo = ModelZoo()
        self.results: Dict[str, Dict[str, ModelTrainingResult]] = {}

    def _load_data(self, symbol: str) -> pd.DataFrame:
        """Loads and prepares reference data for a given symbol."""
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
        print("--- Starting HPO Session ---")

        for symbol in self.symbols:
            print(f"\n--- Processing Symbol: {symbol} ---")
            try:
                df = self._load_data(symbol)
                
                train_df = df.iloc[:-30]
                val_df = df.iloc[-30:]

                data_spec = DataSpec(
                    train_df=train_df,
                    val_df=val_df,
                    symbol=symbol,
                )

                self.results[symbol] = {}

                # Define HPO configs
                hpo_config_small = HPOConfig(max_trials=1, max_epochs=2)
                hpo_config_large = HPOConfig(max_trials=1, max_epochs=3)

                # Train models
                print("Training BaselineLinear...")
                self.results[symbol]['BaselineLinear'] = self.model_zoo.train_baseline_linear(data_spec)

                print("Training AutoNHITS...")
                self.results[symbol]['AutoNHITS'] = self.model_zoo.train_autonhits(data_spec, hpo_config_small)

                print("Training AutoTFT...")
                self.results[symbol]['AutoTFT'] = self.model_zoo.train_autotft(data_spec, hpo_config_large)

            except Exception as e:
                print(f"[ERROR] Error processing symbol {symbol}: {e}")

        self.print_summary()

    def print_summary(self):
        """Prints a summary of the HPO session results."""
        print("\n--- HPO Session Summary ---")
        for symbol, results in self.results.items():
            print(f"\n--- Results for {symbol} ---")
            for model_family, result in results.items():
                if result:
                    print(f"  - {model_family}:")
                    print(f"    - Best MAPE: {result.best_val_mape:.4f}")
                    print(f"    - Best MAE:  {result.best_val_mae:.4f}")
                    print(f"    - Best Model ID: {result.best_model_id}")
        print("--------------------------")

if __name__ == '__main__':
    # Example usage
    agent = HPOAgent(symbols=["AAPL", "NVDA", "TSLA"])
    agent.run_hpo_session()
