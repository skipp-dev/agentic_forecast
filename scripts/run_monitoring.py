#!/usr/bin/env python3
"""
Individual Component Runner: Monitoring

This script runs the monitoring components (drift detection, risk assessment) independently.
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
from src.nodes.monitoring_nodes import drift_detection_node, risk_assessment_node

def setup_logging(log_level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/monitoring.log'),
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

def run_monitoring(raw_data=None, input_file=None, config=None):
    """
    Run monitoring components independently

    Args:
        raw_data: Dict of raw data DataFrames. If None, loads from input_file
        input_file: Path to pickled raw data file
        config: Configuration dict. If None, loads from config.yaml

    Returns:
        GraphState with monitoring results populated
    """
    logger = setup_logging()

    if config is None:
        config = load_config()

    # Load input data
    if raw_data is None:
        if input_file is None:
            logger.error("Either raw_data or input_file must be provided")
            return None

        try:
            with open(input_file, 'rb') as f:
                raw_data = pickle.load(f)
            logger.info(f"Loaded raw data from {input_file}")
        except Exception as e:
            logger.error(f"Failed to load input file {input_file}: {e}")
            return None

    if not raw_data:
        logger.error("No raw data provided")
        return None

    symbols = list(raw_data.keys())
    logger.info(f"Starting monitoring for {len(symbols)} symbols")

    # Initialize state
    state = GraphState(
        symbols=symbols,
        config=config,
        raw_data=raw_data,
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
        # Run drift detection
        logger.info("Running drift detection...")
        state = drift_detection_node(state)

        # Run risk assessment
        logger.info("Running risk assessment...")
        state = risk_assessment_node(state)

        # Summary
        logger.info("Monitoring completed successfully")

        if state['drift_detected']:
            logger.warning("Drift detected in data")
        else:
            logger.info("No significant drift detected")

    except Exception as e:
        error_msg = f"Monitoring failed: {e}"
        state['errors'].append(error_msg)
        logger.error(error_msg)
        return state

    return state

def main():
    parser = argparse.ArgumentParser(description='Run monitoring component')
    parser.add_argument('--input', required=True, help='Input file with raw data (pickled dict)')
    parser.add_argument('--output', help='Output file to save monitoring results (optional)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')

    args = parser.parse_args()

    # Setup logging
    log_level = getattr(logging, args.log_level)
    logger = setup_logging(log_level)

    try:
        # Run monitoring
        state = run_monitoring(input_file=args.input)

        if state is None:
            sys.exit(1)

        # Save output if requested
        if args.output:
            import pickle
            monitoring_results = {
                'drift_metrics': state['drift_metrics'],
                'risk_kpis': state['risk_kpis'],
                'drift_detected': state['drift_detected'],
                'errors': state['errors']
            }
            with open(args.output, 'wb') as f:
                pickle.dump(monitoring_results, f)
            logger.info(f"Monitoring results saved to {args.output}")

        # Print summary
        print(f"\nMonitoring Summary:")
        print(f"Symbols processed: {len(state['symbols'])}")
        print(f"Drift detected: {state['drift_detected']}")
        print(f"Errors: {len(state['errors'])}")

        if not state['drift_metrics'].empty:
            print(f"\nDrift Metrics: {len(state['drift_metrics'])} records")

        if not state['risk_kpis'].empty:
            print(f"Risk KPIs: {len(state['risk_kpis'])} records")

        if state['errors']:
            print("\nErrors:")
            for error in state['errors']:
                print(f"  - {error}")

    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()</content>
<parameter name="filePath">c:\Users\spreu\Documents\agentic_forecast\scripts\run_monitoring.py