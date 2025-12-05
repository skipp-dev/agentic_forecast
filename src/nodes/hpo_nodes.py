import pandas as pd
from typing import Dict, Any
from types import SimpleNamespace
from ..agents.hyperparameter_search_agent import HyperparameterSearchAgent
from ..graphs.state import GraphState
import logging

logger = logging.getLogger(__name__)

def hpo_node(state: GraphState) -> GraphState:
    """
    Runs the hyperparameter optimization process using HyperparameterSearchAgent.
    """
    logger.info("--- Node: Hyperparameter Optimization ---")

    symbols = state.get("symbols", [])
    features = state.get("features", {})
    
    if not symbols:
        logger.warning("No symbols found in state. Skipping HPO.")
        return state

    try:
        # Initialize Agent
        hpo_agent = HyperparameterSearchAgent()
        
        run_type = state.get('run_type', 'DAILY')
        hpo_results = {}

        # Determine models to search
        model_families = ['BaselineLinear']
        if run_type == 'WEEKEND_HPO':
            model_families = hpo_agent.model_families # All supported families
        else:
            # Add NLinear for daily if not already there
            if 'NLinear' in hpo_agent.model_families:
                model_families.append('NLinear')

        for symbol in symbols:
            logger.info(f"Running HPO for {symbol}...")
            
            # Prepare data
            data = features.get(symbol)
            if data is None:
                logger.warning(f"No features found for {symbol}, skipping HPO.")
                continue
                
            # Convert dict to DataFrame if needed
            if isinstance(data, dict):
                data = pd.DataFrame.from_dict(data, orient='index')
                data.index = pd.to_datetime(data.index)
            
            # Ensure 'y' column exists
            if 'y' not in data.columns:
                # Try to create it from close price
                if 'close' in data.columns:
                    data = data.copy()
                    data['y'] = data['close'].pct_change().shift(-1)
                    data = data.dropna()
                else:
                    logger.warning(f"No 'y' or 'close' column for {symbol}, skipping HPO.")
                    continue

            symbol_results = {}
            
            for family in model_families:
                logger.info(f"  - Optimizing {family}...")
                res = hpo_agent.run_search(
                    symbol=symbol, 
                    model_type=family,
                    n_trials=10 if run_type == 'DAILY' else 20,
                    data=data
                )
                
                if 'error' not in res:
                    # Wrap result in an object compatible with execution_nodes expectation
                    result_obj = SimpleNamespace(
                        best_model_id=res.get('best_model_id'),
                        best_val_mape=res.get('best_value'), # best_value is MAE/MAPE from objective
                        best_params=res.get('best_params'),
                        model_family=family
                    )
                    symbol_results[family] = result_obj
                else:
                    logger.error(f"HPO failed for {symbol} - {family}: {res['error']}")
            
            hpo_results[symbol] = symbol_results

        if not hpo_results:
            logger.warning("HPO session completed but returned NO results.")
            state['errors'].append("HPO session returned no results.")

        state['best_models'] = hpo_results
        state['hpo_results'] = hpo_results
        # Reset the trigger flag to prevent loops
        state['hpo_triggered'] = False
        logger.info("HPO session finished.")
        return state

    except Exception as e:
        logger.error(f"Error during HPO: {e}")
        state['errors'].append(f"HPO Error: {e}")
        # Reset flag even on error
        state['hpo_triggered'] = False
        return state
