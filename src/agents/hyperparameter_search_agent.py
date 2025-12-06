"""
Hyperparameter Search Agent

Autonomous hyperparameter optimization using Optuna with GPU acceleration.
Integrates with the existing GPU services and data pipeline.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
import traceback
from datetime import datetime
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Add paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.gpu_services import get_gpu_services
from src.data_pipeline import DataPipeline
from src.services.training_service import GPUTrainingService
from src.services.model_registry_service import ModelRegistryService
from src.data.types import DataSpec

logger = logging.getLogger(__name__)

class HyperparameterSearchAgent:
    """
    Autonomous hyperparameter optimization agent.

    Uses Optuna for Bayesian optimization with GPU acceleration.
    Integrates with existing data pipeline and GPU services.
    """

    def __init__(self, gpu_services=None, data_pipeline=None, risk_mode=False, model_zoo=None):
        """
        Initialize hyperparameter search agent.

        Args:
            gpu_services: GPU services instance (auto-created if None)
            data_pipeline: Data pipeline instance (auto-created if None)
            risk_mode: Whether to enable risk mode (default: False)
            model_zoo: ModelZoo instance (optional)
        """
        self.gpu_services = gpu_services or get_gpu_services()
        self.data_pipeline = data_pipeline or DataPipeline()
        self.risk_mode = risk_mode
        self.model_zoo = model_zoo
        
        # Initialize new services
        self.training_service = GPUTrainingService(gpu_services=self.gpu_services)
        self.model_registry = ModelRegistryService()

        # Optuna configuration
        self.sampler = TPESampler(seed=42)
        self.pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

        # Search configuration
        self.max_trials = 50
        self.timeout_minutes = 30
        self.n_jobs = 1  # Sequential for GPU memory management
        
        # Budget Enforcement
        self.budget_config = {
            'DAILY': {
                'max_symbols': 10,
                'max_trials_per_symbol': 5,
                'max_gpu_hours': 1.0
            },
            'WEEKEND_HPO': {
                'max_symbols': 200,
                'max_trials_per_symbol': 50,
                'max_gpu_hours': 24.0
            }
        }
        self.current_gpu_usage_hours = 0.0

        # Results storage
        self.search_history = []
        self.best_trials = {}

        logger.info("Hyperparameter Search Agent initialized")

    def set_budget(self, run_type: str):
        """Set HPO budget based on run type."""
        if run_type in self.budget_config:
            cfg = self.budget_config[run_type]
            self.max_trials = cfg['max_trials_per_symbol']
            # Note: max_gpu_hours and max_symbols need to be enforced at the orchestration level
            # or by tracking usage across calls.
            logger.info(f"Set HPO budget for {run_type}: Max Trials={self.max_trials}")
        else:
            logger.warning(f"Unknown run type {run_type}, using default budget.")

    @property
    def model_families(self) -> List[str]:
        """List of supported model families."""
        return ['NLinear', 'NHITS', 'NBEATS', 'TFT', 'AutoDLinear', 'AutoNHITS', 'AutoNBEATS', 'AutoTFT', 'BaselineLinear']

    def define_search_space(self, model_type: str) -> Dict[str, Any]:
        """
        Define hyperparameter search space for different model types.

        Args:
            model_type: Type of model

        Returns:
            Search space configuration
        """
        if model_type == 'BaselineLinear':
            return {
                'fit_intercept': {'type': 'categorical', 'choices': [True, False]}
            }
        
        elif model_type == 'AutoDLinear':
             return {
                'input_size': {'type': 'categorical', 'choices': [24, 48, 96]},
                'start_padding_enabled': {'type': 'categorical', 'choices': [True, False]},
                'horizon': {'type': 'fixed', 'value': 24}
            }

        elif model_type in ['NLinear', 'NHITS', 'NBEATS', 'TFT']:
            return {
                'learning_rate': {'type': 'float', 'low': 1e-4, 'high': 1e-2, 'log': True},
                'batch_size': {'type': 'categorical', 'choices': [16, 32, 64, 128]},
                'max_steps': {'type': 'int', 'low': 500, 'high': 2000},
                'input_size': {'type': 'categorical', 'choices': [24, 48, 96]}, # Multiples of horizon usually
                'horizon': {'type': 'fixed', 'value': 24} # Default horizon, can be overridden
            }

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def sample_hyperparameters(self, trial: optuna.Trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sample hyperparameters for a trial.

        Args:
            trial: Optuna trial object
            search_space: Search space configuration

        Returns:
            Sampled hyperparameters
        """
        params = {}

        for param_name, config in search_space.items():
            if config['type'] == 'int':
                params[param_name] = trial.suggest_int(
                    param_name, config['low'], config['high']
                )
            elif config['type'] == 'float':
                params[param_name] = trial.suggest_float(
                    param_name, config['low'], config['high'],
                    log=config.get('log', False)
                )
            elif config['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name, config['choices']
                )
            elif config['type'] == 'fixed':
                params[param_name] = config['value']

        return params

    def objective_function(self, trial: optuna.Trial, symbol: str, model_type: str, data: Optional[pd.DataFrame] = None) -> float:
        """
        Objective function for hyperparameter optimization.

        Args:
            trial: Optuna trial object
            symbol: Stock symbol to optimize for
            model_type: Type of model to optimize
            data: Optional DataFrame to use instead of fetching

        Returns:
            Validation metric (negative MAPE for minimization)
        """
        try:
            # Get search space
            search_space = self.define_search_space(model_type)

            # Sample hyperparameters
            hyperparams = self.sample_hyperparameters(trial, search_space)

            # Optimize GPU for training
            if self.gpu_services:
                self.gpu_services.optimize_for_training()

            # Fetch data
            if data is not None:
                df = data
            else:
                # We need to fetch data here because HPO agent runs independently
                # Ideally, data should be passed in, but for now we fetch it
                df = self.data_pipeline.fetch_stock_data(symbol, period='2y')
            
            if df is None or df.empty:
                raise ValueError(f"No data found for {symbol}")

            # Split data for validation
            train_size = int(len(df) * 0.8)
            train_df = df.iloc[:train_size]
            val_df = df.iloc[train_size:]
            
            # Create DataSpec
            data_spec = DataSpec(train_df=train_df, val_df=val_df, target_col='y')
            
            # Train model using GPUTrainingService
            
            # Train using service
            result = self.training_service.train_model(
                symbol=symbol,
                model_type=model_type,
                data=data_spec,
                hyperparams=hyperparams,
                job_id=f"hpo_{trial.number}"
            )
            
            if result['status'] == 'failed':
                raise Exception(result.get('error', 'Unknown error'))
            
            metrics = result.get('metrics', {})
            # Use MAE as primary metric for optimization
            metric = metrics.get('mae', 999.0)
            
            trial.set_user_attr('model_id', result.get('model_id'))
            results = {'test_metrics': metrics}

            # Store trial results
            trial.set_user_attr('hyperparams', hyperparams)
            trial.set_user_attr('test_metrics', results.get('test_metrics', {}))
            trial.set_user_attr('cv_metric', metric)

            logger.info(f"Trial {trial.number}: {model_type} on {symbol} - CV Metric: {metric:.4f}")

            return metric

        except Exception as e:
            logger.error(f"Trial {trial.number} failed for {symbol} ({model_type}): {e}")
            logger.error(f"Failed params: {hyperparams if 'hyperparams' in locals() else 'Unknown'}")
            logger.error(traceback.format_exc())
            # Return high penalty for failed trials
            return 999.0

    def run_search(self, symbol: str, model_type: str,
                   max_trials: Optional[int] = None,
                   timeout_minutes: Optional[int] = None,
                   n_trials: Optional[int] = None,
                   data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Run hyperparameter search for a symbol and model type.

        Args:
            symbol: Stock symbol to optimize for
            model_type: Type of model to optimize
            max_trials: Maximum number of trials (overrides default)
            timeout_minutes: Timeout in minutes (overrides default)
            n_trials: Alias for max_trials
            data: Optional DataFrame to use

        Returns:
            Search results dictionary
        """
        max_trials = max_trials or n_trials or self.max_trials
        timeout_minutes = timeout_minutes or self.timeout_minutes

        logger.info(f"Starting hyperparameter search for {symbol} {model_type}")
        logger.info(f"Max trials: {max_trials}, Timeout: {timeout_minutes} minutes")

        # Create study
        study_name = f"{symbol}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study = optuna.create_study(
            study_name=study_name,
            sampler=self.sampler,
            pruner=self.pruner,
            direction='minimize'  # Minimize MAE
        )

        # Optimize
        try:
            study.optimize(
                lambda trial: self.objective_function(trial, symbol, model_type, data),
                n_trials=max_trials,
                timeout=timeout_minutes * 60,
                n_jobs=self.n_jobs
            )
        except Exception as e:
            logger.error(f"Hyperparameter search failed: {e}")
            return {'error': str(e)}

        # Extract best results
        best_trial = study.best_trial
        best_params = best_trial.user_attrs.get('hyperparams', {})
        test_metrics = best_trial.user_attrs.get('test_metrics', {})
        best_model_id = best_trial.user_attrs.get('model_id')

        results = {
            'symbol': symbol,
            'model_type': model_type,
            'study_name': study_name,
            'best_value': best_trial.value,
            'best_params': best_params,
            'best_model_id': best_model_id,
            'test_metrics': test_metrics,
            'n_trials': len(study.trials),
            'completed_at': datetime.now().isoformat(),
            'search_duration_minutes': (datetime.now() - datetime.strptime(
                "_".join(study_name.split('_')[-2:]), '%Y%m%d_%H%M%S'
            )).total_seconds() / 60
        }

        # Store in history
        self.search_history.append(results)
        self.best_trials[f"{symbol}_{model_type}"] = results

        logger.info(f"Hyperparameter search completed for {symbol} {model_type}")
        logger.info(f"Best MAE: {best_trial.value:.4f}")
        logger.info(f"Best params: {best_params}")

        return results

    def get_best_hyperparams(self, symbol: str, model_type: str) -> Optional[Dict[str, Any]]:
        """
        Get best hyperparameters for a symbol and model type.

        Args:
            symbol: Stock symbol
            model_type: Type of model

        Returns:
            Best hyperparameters or None if not found
        """
        key = f"{symbol}_{model_type}"
        if key in self.best_trials:
            return self.best_trials[key]['best_params']
        return None

    def save_search_results(self, filepath: str):
        """Save search history to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.search_history, f, indent=2, default=str)

    def load_search_results(self, filepath: str):
        """Load search history from JSON file."""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.search_history = json.load(f)

    def get_search_summary(self) -> pd.DataFrame:
        """Get summary of all hyperparameter searches."""
        if not self.search_history:
            return pd.DataFrame()

        summary_data = []
        for result in self.search_history:
            summary_data.append({
                'symbol': result['symbol'],
                'model_type': result['model_type'],
                'best_mae': result['best_value'],
                'n_trials': result['n_trials'],
                'duration_minutes': result['search_duration_minutes'],
                'completed_at': result['completed_at']
            })

        return pd.DataFrame(summary_data)

    # Alias for compatibility
    search_hyperparameters = run_search

    def run_model_zoo_hpo(self, symbol: str, data_spec: Any, model_families: List[str]) -> Dict[str, Any]:
        """
        Run HPO using ModelZoo.

        Args:
            symbol: Stock symbol to optimize for
            data_spec: Data specification for training
            model_families: List of model families to include in HPO

        Returns:
            Results of the HPO process
        """
        results = []
        
        # Inject baseline if risk mode is enabled
        if self.risk_mode and 'BaselineLinear' not in model_families:
            model_families.append('BaselineLinear')

        for family in model_families:
            try:
                if family == 'AutoNHITS':
                    res = self.model_zoo.train_autonhits(data_spec)
                elif family == 'AutoNBEATS':
                    res = self.model_zoo.train_autonbeats(data_spec)
                elif family == 'AutoDLinear':
                    res = self.model_zoo.train_autodlinear(data_spec)
                elif family == 'BaselineLinear':
                    res = self.model_zoo.train_baseline_linear(data_spec)
                else:
                    continue
                
                results.append({
                    'model_family': res.model_family,
                    'best_val_mape': res.best_val_mape
                })
            except Exception as e:
                logger.error(f"Failed to train {family} in ModelZoo HPO for {symbol}: {e}")
                logger.error(traceback.format_exc())
        
        # Find best family
        best_family = None
        best_mape = float('inf')
        
        for res in results:
            if res['best_val_mape'] is not None and res['best_val_mape'] < best_mape:
                best_mape = res['best_val_mape']
                best_family = res['model_family']

        return {
            'success': True,
            'all_results': results,
            'best_family': best_family,
            'best_val_mape': best_mape if best_family else None
        }

    def test_ensemble_combinations(self, symbol: str, data_spec: Any, families: List[str]) -> Dict[str, Any]:
        """
        Test ensemble of multiple model families.

        Args:
            symbol: Stock symbol
            data_spec: Data specification
            families: List of model families to ensemble

        Returns:
            Dictionary with ensemble results
        """
        preds_list = []
        
        for family in families:
            try:
                if family == 'AutoNHITS':
                    res = self.model_zoo.train_autonhits(data_spec)
                elif family == 'AutoNBEATS':
                    res = self.model_zoo.train_autonbeats(data_spec)
                elif family == 'BaselineLinear':
                    res = self.model_zoo.train_baseline_linear(data_spec)
                else:
                    continue
                
                if res.val_preds is not None:
                    # Extract prediction column (excluding ds, unique_id, y)
                    pred_cols = [c for c in res.val_preds.columns if c not in ['ds', 'unique_id', 'y']]
                    if pred_cols:
                        preds_list.append(res.val_preds[pred_cols[0]].values)
            except Exception as e:
                logger.error(f"Failed to train {family} for ensemble: {e}")
                logger.error(traceback.format_exc())

        if not preds_list:
            return {'success': False, 'error': 'No models trained successfully'}

        # Average predictions
        # Ensure all predictions have same length
        min_len = min(len(p) for p in preds_list)
        preds_list = [p[-min_len:] for p in preds_list]
        
        ensemble_preds = np.mean(preds_list, axis=0)
        
        # Calculate MAPE
        # Handle DataSpec being either object or dict (though it should be object)
        try:
            y_true = data_spec.val_df['y'].values
        except:
            y_true = data_spec['val_df']['y'].values
            
        y_true = y_true[-len(ensemble_preds):]
        
        # Simple MAPE implementation to avoid extra imports if possible, 
        # or import from neuralforecast if available
        try:
            from neuralforecast.losses.numpy import mape
            ensemble_mape = mape(y_true, ensemble_preds)
        except ImportError:
            mask = y_true != 0
            ensemble_mape = np.mean(np.abs((y_true[mask] - ensemble_preds[mask]) / y_true[mask])) * 100
        
        return {
            'success': True,
            'ensemble_mape': ensemble_mape
        }