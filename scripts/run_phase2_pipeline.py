#!/usr/bin/env python3
"""
Phase 2 Pipeline Runner: Macro Data, Regimes, and Strategies

This script runs the Phase 2 components of the agentic forecasting system.
"""

import os
import sys
import yaml
import logging
import argparse
import pickle
import pandas as pd
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.macro_data_agent import MacroDataAgent
from src.agents.regime_detection_agent import RegimeDetectionAgent
from src.agents.strategy_selection_agent import StrategySelectionAgent

def setup_logging(log_level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/phase2_pipeline.log'),
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

def run_phase2_pipeline(start_date=None, end_date=None, config=None, risk_tolerance='medium'):
    """
    Run the complete Phase 2 pipeline: macro data -> regime detection -> strategy selection

    Args:
        start_date: Start date for macro data (YYYY-MM-DD)
        end_date: End date for macro data (YYYY-MM-DD)
        config: Configuration dict
        risk_tolerance: Risk tolerance for strategy selection

    Returns:
        Dictionary with Phase 2 results
    """
    logger = setup_logging()

    if config is None:
        config = load_config()

    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')

    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    logger.info("Starting Phase 2 pipeline")
    logger.info(f"Date range: {start_date} to {end_date}")

    results = {}

    try:
        # Step 1: Collect macro data
        logger.info("Step 1: Collecting macro economic data")
        macro_agent = MacroDataAgent(config.get('macro', {}))
        macro_data = macro_agent.get_macro_data(start_date, end_date)

        results['macro_data'] = macro_data
        logger.info(f"Collected {len(macro_data['raw_data'])} macro indicators")

        # Step 2: Detect regimes
        logger.info("Step 2: Detecting market regimes")
        regime_agent = RegimeDetectionAgent(config.get('regime', {}))
        regime_analysis = regime_agent.get_regime_summary(macro_data['processed_features'])

        results['regime_analysis'] = regime_analysis
        logger.info(f"Detected {len(regime_analysis['regimes'])} regime types")

        # Step 3: Strategy selection
        logger.info("Step 3: Selecting optimal strategies")

        # Get current regimes (most recent)
        current_regimes = {}
        for regime_type, regime_series in regime_analysis['regimes'].items():
            current_regimes[regime_type] = regime_series.iloc[-1] if not regime_series.empty else 'unknown'

        strategy_agent = StrategySelectionAgent(config.get('strategy', {}))
        strategy_recommendations = strategy_agent.get_strategy_recommendations(
            current_regimes, risk_tolerance=risk_tolerance
        )

        results['strategy_recommendations'] = strategy_recommendations
        logger.info(f"Selected {len(strategy_recommendations['selected_strategies'])} strategies")

        # Step 4: Create regime playbook
        logger.info("Step 4: Creating strategy playbook")
        # This would require historical data - placeholder for now
        playbook = strategy_agent.create_regime_playbook(
            regime_analysis['regimes'],
            macro_data['processed_features']  # Using macro data as proxy
        )

        results['strategy_playbook'] = playbook
        logger.info("Strategy playbook created")

        results['pipeline_status'] = 'completed'
        results['execution_timestamp'] = datetime.now()

    except Exception as e:
        error_msg = f"Phase 2 pipeline failed: {e}"
        results['pipeline_status'] = 'failed'
        results['error'] = error_msg
        logger.error(error_msg)

    return results

def main():
    parser = argparse.ArgumentParser(description='Run Phase 2 pipeline components')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD, default: 3 years ago)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD, default: today)')
    parser.add_argument('--risk-tolerance', choices=['low', 'medium', 'high'],
                       default='medium', help='Risk tolerance for strategy selection')
    parser.add_argument('--output', help='Output file to save Phase 2 results (optional)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')

    args = parser.parse_args()

    # Setup logging
    log_level = getattr(logging, args.log_level)
    logger = setup_logging(log_level)

    try:
        # Run Phase 2 pipeline
        results = run_phase2_pipeline(
            start_date=args.start_date,
            end_date=args.end_date,
            risk_tolerance=args.risk_tolerance
        )

        # Save output if requested
        if args.output:
            import pickle
            with open(args.output, 'wb') as f:
                pickle.dump(results, f)
            logger.info(f"Phase 2 results saved to {args.output}")

        # Print summary
        print(f"\nPhase 2 Pipeline Summary:")
        print(f"Status: {results.get('pipeline_status', 'unknown')}")

        if results.get('macro_data'):
            macro_data = results['macro_data']
            print(f"Macro indicators collected: {len(macro_data.get('raw_data', {}))}")

        if results.get('regime_analysis'):
            regime_analysis = results['regime_analysis']
            print(f"Regime types detected: {len(regime_analysis.get('regimes', {}))}")

        if results.get('strategy_recommendations'):
            strategy_rec = results['strategy_recommendations']
            print(f"Strategies selected: {len(strategy_rec.get('selected_strategies', []))}")

        if results.get('error'):
            print(f"Error: {results['error']}")

    except Exception as e:
        logger.error(f"Phase 2 pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()