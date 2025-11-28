#!/usr/bin/env python3
"""
Individual Component Runner: Model Training and Forecasting

This script runs the model training and forecasting components independently.
"""

import os
import sys
import yaml
import logging
import argparse
import pickle
import pandas as pd

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graphs.state import GraphState
from src.nodes.execution_nodes import forecasting_node

def setup_logging(log_level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/model_training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config():
    """Load configuration from config.yaml"""
    config_path = "config.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

def run_model_training(features_data=None, input_file=None, config=None):
    """
    Run model training and forecasting independently

    Args:
        features_data: Dict of features DataFrames. If None, loads from input_file
        input_file: Path to pickled features file
        config: Configuration dict. If None, loads from config.yaml

    Returns:
        GraphState with forecasts populated
    """
    logger = setup_logging()

    if config is None:
        config = load_config()

    # Load input data
    if features_data is None:
        if input_file is None:
            logger.error("Either features_data or input_file must be provided")
            return None

        try:
            with open(input_file, 'rb') as f:
                features_data = pickle.load(f)
            logger.info(f"Loaded features from {input_file}")
        except Exception as e:
            logger.error(f"Failed to load input file {input_file}: {e}")
            return None

    if not features_data:
        logger.error("No features data provided")
        return None

    symbols = list(features_data.keys())
    logger.info(f"Starting model training and forecasting for {len(symbols)} symbols")

    # Initialize state
    state = GraphState(
        symbols=symbols,
        config=config,
        raw_data={},  # Not needed for this component
        features=features_data,
        forecasts={},
        performance_summary=pd.DataFrame(),
        drift_metrics=pd.DataFrame(),
        risk_kpis=pd.DataFrame(),
        anomalies={},
        recommended_actions=[],
        executed_actions=[],
        retrained_models={},
        best_models={},
        errors=[],
        hpo_results={},
        shap_results={},
        analytics_summary=pd.DataFrame(),
        hpo_decision={},
        retraining_history=[],
        guardrail_log=[],
        hpo_triggered=False,
        drift_detected=False,
        edge_index=None,
        node_features=None,
        symbol_to_idx={}
    )

    try:
        # Run forecasting node (includes model training)
        logger.info("Running forecasting node...")
        state = forecasting_node(state)

        # Summary
        successful_symbols = len(state['forecasts'])
        logger.info(f"Model training and forecasting completed. Generated forecasts for {successful_symbols} symbols")

        for symbol, forecast_data in state['forecasts'].items():
            logger.info(f"  {symbol}: {len(forecast_data)} forecast types")

    except Exception as e:
        error_msg = f"Model training/forecasting failed: {e}"
        state['errors'].append(error_msg)
        logger.error(error_msg)
        return state

    return state

def main():
    parser = argparse.ArgumentParser(description='Run model training and forecasting component')
    parser.add_argument('--input', required=True, help='Input file with features (pickled dict)')
    parser.add_argument('--output', help='Output file to save forecasts (optional)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')

    args = parser.parse_args()

    # Setup logging
    log_level = getattr(logging, args.log_level)
    logger = setup_logging(log_level)

    try:
        # Run model training and forecasting
        state = run_model_training(input_file=args.input)

        if state is None:
            sys.exit(1)

        # Save output if requested
        if args.output:
            import pickle
            with open(args.output, 'wb') as f:
                pickle.dump(state['forecasts'], f)
            logger.info(f"Forecasts saved to {args.output}")

        # Print summary
        print(f"\nModel Training & Forecasting Summary:")
        print(f"Symbols processed: {len(state['symbols'])}")
        print(f"Forecasts generated: {len(state['forecasts'])}")
        print(f"Errors: {len(state['errors'])}")

        if state['forecasts']:
            print("\nForecast details:")
            for symbol, forecasts in state['forecasts'].items():
                print(f"  {symbol}: {len(forecasts)} forecast types")

        if state['errors']:
            print("\nErrors:")
            for error in state['errors']:
                print(f"  - {error}")

    except Exception as e:
        logger.error(f"Model training/forecasting failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()</content>
<parameter name="filePath">c:\Users\spreu\Documents\agentic_forecast\scripts\run_model_training.py