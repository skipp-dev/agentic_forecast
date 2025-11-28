#!/usr/bin/env python3
"""
Individual Component Runner: Feature Engineering

This script runs the feature engineering component independently with proper error handling and logging.
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
from src.agents.feature_agent import FeatureAgent

def setup_logging(log_level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/feature_engineering.log'),
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

def run_feature_engineering(input_data=None, input_file=None, config=None):
    """
    Run feature engineering independently

    Args:
        input_data: Dict of raw data DataFrames. If None, loads from input_file
        input_file: Path to pickled raw data file
        config: Configuration dict. If None, loads from config.yaml

    Returns:
        GraphState with features populated
    """
    logger = setup_logging()

    if config is None:
        config = load_config()

    # Load input data
    if input_data is None:
        if input_file is None:
            logger.error("Either input_data or input_file must be provided")
            return None

        try:
            with open(input_file, 'rb') as f:
                input_data = pickle.load(f)
            logger.info(f"Loaded raw data from {input_file}")
        except Exception as e:
            logger.error(f"Failed to load input file {input_file}: {e}")
            return None

    if not input_data:
        logger.error("No input data provided")
        return None

    symbols = list(input_data.keys())
    logger.info(f"Starting feature engineering for {len(symbols)} symbols")

    # Initialize state
    state = GraphState(
        symbols=symbols,
        config=config,
        raw_data=input_data,
        features={},
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
        # Initialize feature agent
        logger.info("Initializing feature engineering agent...")
        agent = FeatureAgent()

        # Generate features
        logger.info("Generating features...")
        features = agent.generate_features(state['raw_data'])

        state['features'] = features

        # Summary
        successful_symbols = len(features)
        logger.info(f"Feature engineering completed. Generated features for {successful_symbols} symbols")

        for symbol, feature_df in features.items():
            logger.info(f"  {symbol}: {feature_df.shape[1]} features, {feature_df.shape[0]} rows")

    except Exception as e:
        error_msg = f"Feature engineering failed: {e}"
        state['errors'].append(error_msg)
        logger.error(error_msg)
        return state

    return state

def main():
    parser = argparse.ArgumentParser(description='Run feature engineering component')
    parser.add_argument('--input', required=True, help='Input file with raw data (pickled dict)')
    parser.add_argument('--output', help='Output file to save features (optional)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')

    args = parser.parse_args()

    # Setup logging
    log_level = getattr(logging, args.log_level)
    logger = setup_logging(log_level)

    try:
        # Run feature engineering
        state = run_feature_engineering(input_file=args.input)

        if state is None:
            sys.exit(1)

        # Save output if requested
        if args.output:
            import pickle
            with open(args.output, 'wb') as f:
                pickle.dump(state['features'], f)
            logger.info(f"Features saved to {args.output}")

        # Print summary
        print(f"\nFeature Engineering Summary:")
        print(f"Symbols processed: {len(state['symbols'])}")
        print(f"Features generated: {len(state['features'])}")
        print(f"Errors: {len(state['errors'])}")

        if state['features']:
            print("\nFeature details:")
            for symbol, features_df in state['features'].items():
                print(f"  {symbol}: {features_df.shape[1]} features, {features_df.shape[0]} rows")

        if state['errors']:
            print("\nErrors:")
            for error in state['errors']:
                print(f"  - {error}")

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
