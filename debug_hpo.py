import os
import sys
import logging
import yaml
import pandas as pd
from src.core.run_context import RunType, RunContext
from src.pipelines import run_weekend_hpo_pipeline

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Force DEBUG level for specific loggers
logging.getLogger('src.agents.llm_hpo_planner_agent').setLevel(logging.DEBUG)
logging.getLogger('src.agents.hpo_agent').setLevel(logging.DEBUG)
logging.getLogger('src.nodes.hpo_nodes').setLevel(logging.DEBUG)

def load_config():
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)

def main():
    logger.info("Starting DEBUG HPO run...")
    
    # Load config
    config = load_config()
    
    # Override config for debugging
    config['hpo']['weekend_hpo']['small_models']['trials'] = 1
    config['hpo']['weekend_hpo']['small_models']['max_epochs'] = 1
    config['hpo']['weekend_hpo']['large_models']['trials'] = 1
    config['hpo']['weekend_hpo']['large_models']['max_epochs'] = 1
    
    # Use a single symbol for debugging
    symbols = ['AAPL']
    
    # Create context
    ctx = RunContext.create(run_type=RunType.WEEKEND_HPO)
    
    try:
        run_weekend_hpo_pipeline(ctx, symbols, config)
        logger.info("DEBUG HPO run completed successfully.")
    except Exception as e:
        logger.error(f"DEBUG HPO run failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
