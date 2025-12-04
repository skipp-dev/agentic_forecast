import os
import sys
import pandas as pd
import logging
import time
from typing import List, Dict, Any

from models.model_zoo import ModelZoo, DataSpec, HPOConfig, ModelTrainingResult
from src.utils.champion_selection import select_champion_model, ModelMetrics
from src.data.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

class HPOAgent:
    """
    Hyperparameter Optimization Agent to orchestrate model training and tuning.
    """
    def __init__(self, symbols: List[str], data_store: Dict[str, pd.DataFrame] = None, data_path: str = 'data/reference', config: Dict = None, hpo_plan: Any = None):
        """
        Initializes the HPOAgent.

        Args:
            symbols (List[str]): A list of stock symbols to process.
            data_store (Dict[str, pd.DataFrame]): Dictionary containing data for each symbol.
            data_path (str): The path to the reference data (fallback).
            config (Dict): Configuration dictionary.
            hpo_plan (Any): Optional HPO plan from LLM.
        """
        self.symbols = symbols
        self.data_store = data_store
        self.data_path = data_path
        self.config = config or {}
        self.hpo_plan = hpo_plan
        self.model_zoo = ModelZoo()
        self.results: Dict[str, Dict[str, ModelTrainingResult]] = {}
        self.model_registry = ModelRegistry()

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
        
        # Determine run type
        run_type = os.environ.get('RUN_TYPE', 'DAILY')
        logger.info(f"HPO Run Type: {run_type}")

        # Apply Plan Filtering
        target_symbols = self.symbols
        if self.hpo_plan and hasattr(self.hpo_plan, 'symbols_to_focus') and self.hpo_plan.symbols_to_focus:
            logger.info(f"Applying HPO Plan: Focusing on symbols {self.hpo_plan.symbols_to_focus}")
            # Intersect with available symbols
            target_symbols = [s for s in self.symbols if s in self.hpo_plan.symbols_to_focus]
            if not target_symbols:
                logger.warning("Plan symbols do not match any available symbols. Falling back to all.")
                target_symbols = self.symbols

        for symbol in target_symbols:
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
                hpo_settings = self.config.get('hpo', {})
                
                # Select settings based on run_type
                if run_type == 'WEEKEND_HPO':
                    run_settings = hpo_settings.get('weekend_hpo', {})
                    if not run_settings:
                        logger.warning("WEEKEND_HPO settings not found in config, falling back to defaults.")
                else:
                    run_settings = hpo_settings.get('daily', {})
                    if not run_settings:
                        logger.warning("DAILY settings not found in config, falling back to defaults.")

                # Get early stopping patience
                early_stopping_patience = hpo_settings.get('early_stopping_patience', 3)

                # Determine models to run
                models_to_run = ['BaselineLinear', 'AutoDLinear', 'AutoNHITS', 'AutoTFT']
                
                # Plan Overrides
                if self.hpo_plan and hasattr(self.hpo_plan, 'families_to_prioritize') and self.hpo_plan.families_to_prioritize:
                     # Always keep BaselineLinear
                     models_to_run = ['BaselineLinear'] + [m for m in models_to_run if m in self.hpo_plan.families_to_prioritize]

                # Helper to get trials
                def get_trials(family, default_trials, default_epochs):
                    trials = default_trials
                    epochs = default_epochs
                    if self.hpo_plan and hasattr(self.hpo_plan, 'budget_allocation'):
                        if family in self.hpo_plan.budget_allocation:
                            trials = self.hpo_plan.budget_allocation[family]
                            logger.info(f"Plan Override: {family} trials set to {trials}")
                    return trials, epochs

                # Small config (for lighter models)
                small_settings = run_settings.get('small_models', {}) if run_settings else {}
                small_trials_def = small_settings.get('trials', 10)
                small_epochs_def = small_settings.get('max_epochs', 10)

                # Large config (for heavier models)
                large_settings = run_settings.get('large_models', {}) if run_settings else {}
                large_trials_def = large_settings.get('trials', 20)
                large_epochs_def = large_settings.get('max_epochs', 20)

                # Train models
                if 'BaselineLinear' in models_to_run:
                    logger.info(f"Training BaselineLinear for {symbol}...")
                    self.results[symbol]['BaselineLinear'] = self.model_zoo.train_baseline_linear(data_spec)

                if 'AutoDLinear' in models_to_run:
                    trials, epochs = get_trials('AutoDLinear', small_trials_def, small_epochs_def)
                    hpo_config = HPOConfig(max_trials=trials, max_epochs=epochs, early_stopping_patience=early_stopping_patience)
                    logger.info(f"Training AutoDLinear for {symbol} (trials={trials}, epochs={epochs})...")
                    self.results[symbol]['AutoDLinear'] = self.model_zoo.train_autodlinear(data_spec, hpo_config)

                if 'AutoNHITS' in models_to_run:
                    trials, epochs = get_trials('AutoNHITS', small_trials_def, small_epochs_def)
                    hpo_config = HPOConfig(max_trials=trials, max_epochs=epochs, early_stopping_patience=early_stopping_patience)
                    logger.info(f"Training AutoNHITS for {symbol} (trials={trials}, epochs={epochs})...")
                    self.results[symbol]['AutoNHITS'] = self.model_zoo.train_autonhits(data_spec, hpo_config)

                if 'AutoTFT' in models_to_run:
                    trials, epochs = get_trials('AutoTFT', large_trials_def, large_epochs_def)
                    hpo_config = HPOConfig(max_trials=trials, max_epochs=epochs, early_stopping_patience=early_stopping_patience)
                    logger.info(f"Training AutoTFT for {symbol} (trials={trials}, epochs={epochs})...")
                    self.results[symbol]['AutoTFT'] = self.model_zoo.train_autotft(data_spec, hpo_config)

            except Exception as e:
                logger.error(f"Error processing symbol {symbol}: {e}", exc_info=True)

        self.print_summary()

    def print_summary(self):
        """Prints a summary of the HPO session results and selects champions."""
        logger.info("--- HPO Session Summary ---")
        for symbol, results in self.results.items():
            logger.info(f"--- Results for {symbol} ---")
            
            model_metrics_list = []
            
            for model_family, result in results.items():
                if result:
                    logger.info(f"  - {model_family}:")
                    logger.info(f"    - Best MAPE: {result.best_val_mape:.4f}")
                    logger.info(f"    - Best MAE:  {result.best_val_mae:.4f}")
                    logger.info(f"    - Best Model ID: {result.best_model_id}")
                    
                    # Create ModelMetrics for champion selection
                    # Note: SMAPE and Directional Accuracy are not currently returned by ModelZoo
                    # We will use MAPE as proxy for SMAPE and 0.5 for DA if missing
                    # In a real implementation, ModelZoo should return these metrics
                    mape = result.best_val_mape if result.best_val_mape is not None else 1.0
                    smape = mape # Proxy
                    da = 0.5 # Default
                    
                    metrics = ModelMetrics(
                        family=model_family,
                        mape=mape,
                        smape=smape,
                        directional_accuracy=da
                    )
                    model_metrics_list.append(metrics)
            
            if model_metrics_list:
                try:
                    selection = select_champion_model(model_metrics_list)
                    logger.info(f"  >>> CHAMPION: {selection.champion.family}")
                    logger.info(f"  >>> Reason: {selection.reason}")
                    
                    # Update Registry
                    self.model_registry.set_champion_model(
                        symbol, 
                        selection.champion.family, 
                        selection.reason
                    )
                    self.model_registry.set_last_hpo_run(symbol, time.time())
                    
                except Exception as e:
                    logger.error(f"  >>> Error selecting champion: {e}")

        logger.info("--------------------------")

if __name__ == '__main__':
    # Example usage
    agent = HPOAgent(symbols=["AAPL", "NVDA", "TSLA"])
    agent.run_hpo_session()
