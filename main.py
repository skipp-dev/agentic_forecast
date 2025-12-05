# Entry point for the agentic_forecast application
import os
import yaml
import signal
import sys
import logging
import tensorflow as tf
import argparse
import re
import asyncio
import pandas as pd
import torch
from langgraph.graph import StateGraph, END

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on system env vars

# Suppress TensorFlow warnings but allow GPU detection
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations to prevent numerical warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Reduce TensorFlow logging (0=INFO, 1=WARNING, 2=ERROR, 3=FATAL)

from src.core.run_context import RunType, RunContext
from src.core.state import PipelineGraphState
from src.nodes.execution_nodes import (
    data_ingestion_node,
    feature_engineering_node,
    forecasting_node
)
from src.nodes.macro_nodes import macro_data_node, regime_detection_node
from src.nodes.hpo_nodes import hpo_node
from src.nodes.agent_nodes import analytics_node
from src.nodes.monitoring_nodes import monitoring_node
from src.nodes.retraining_nodes import retraining_node
from src.nodes.reporting_nodes import generate_report_node
from src.nodes.strategy_nodes import strategy_node
from src.nodes.portfolio_nodes import portfolio_construction_node
from src.agents.orchestrator_agent import OrchestratorAgent

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Agentic Forecast System')
    parser.add_argument('--task', choices=['full', 'llm_analytics_only'], default='full',
                       help='Task to run (default: full)')
    parser.add_argument('--run_type', choices=[rt.value for rt in RunType],
                       default=RunType.DAILY.value, help='Type of run (default: DAILY)')
    return parser.parse_args()

# Parse args early to set environment variables before imports
args = parse_args()

# For BACKTEST mode, skip neuralforecast imports to avoid hangs
if args.run_type == "BACKTEST":
    os.environ['SKIP_NEURALFORECAST'] = 'true'
    os.environ['RUN_TYPE'] = 'BACKTEST'
    print("Skipping NeuralForecast imports for BACKTEST mode")

def load_config():
    """Load configuration from config.yaml with environment variable substitution"""
    config_path = "config.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Helper to substitute env vars
        def substitute_env_vars(item):
            if isinstance(item, dict):
                return {k: substitute_env_vars(v) for k, v in item.items()}
            elif isinstance(item, list):
                return [substitute_env_vars(v) for v in item]
            elif isinstance(item, str):
                # Replace ${VAR} with value from os.environ
                pattern = re.compile(r'\$\{([^}]+)\}')
                return pattern.sub(lambda m: os.environ.get(m.group(1), m.group(0)), item)
            return item
            
        config = substitute_env_vars(config)
        return config
    return {}

def load_symbols_from_csv(csv_path="watchlist_main.csv", max_symbols=None):
    """Load symbols from main watchlist CSV file"""
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if 'Symbol' in df.columns:
                # Process symbols from the watchlist
                all_symbols = df['Symbol'].tolist()
                if max_symbols is not None and len(all_symbols) > max_symbols:
                    print(f"Limiting to {max_symbols} symbols for testing (from {len(all_symbols)} total)")
                    return all_symbols[:max_symbols]
                return all_symbols
            else:
                logger.warning(f"'Symbol' column not found in {csv_path}")
                return []
        except Exception as e:
            logger.error(f"Error loading symbols from {csv_path}: {e}")
            return []
    else:
        logger.warning(f"{csv_path} not found")
        return []

def setup_gpu():
    """Setup GPU configuration for optimal utilization in production"""
    try:
        print("Detecting GPU devices...")

        # Check TensorFlow GPU support
        tf_gpus = tf.config.list_physical_devices('GPU')
        tf_cuda_available = tf.test.is_built_with_cuda()
        
        # Check PyTorch GPU support (used by NeuralForecast)
        torch_cuda_available = False
        try:
            import torch
            torch_cuda_available = torch.cuda.is_available()
            torch_device_count = torch.cuda.device_count() if torch_cuda_available else 0
        except ImportError:
            torch_device_count = 0

        print(f"Available devices:")
        print(f"   TensorFlow GPUs: {len(tf_gpus)} (CUDA built-in: {tf_cuda_available})")
        print(f"   PyTorch GPUs: {torch_device_count} (CUDA available: {torch_cuda_available})")

        # Use PyTorch GPU detection for NeuralForecast compatibility
        gpu_available = torch_cuda_available
        
        if gpu_available:
            print(f"   GPU Details: PyTorch detected {torch_device_count} GPU(s)")
            print("   NeuralForecast models will use GPU acceleration via PyTorch/Ray")

            # Configure TensorFlow if GPUs are available (for any TF components)
            if tf_gpus:
                for i, gpu in enumerate(tf_gpus):
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                        print(f"   TensorFlow GPU {i}: Memory growth enabled")
                    except RuntimeError as e:
                        print(f"   TensorFlow GPU {i} configuration error: {e}")

            print("GPU setup complete - NeuralForecast training will use GPU acceleration!")
            return True

        else:
            print("No GPU devices found - training will use CPU")
            # Suppress CUDA warnings only when no GPU is available
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            return False

    except Exception as e:
        print(f"GPU setup failed: {e}")
        print("Falling back to CPU training")
        # Suppress CUDA warnings on error
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        return False

def setup_langsmith(config):
    """Setup LangSmith tracing"""
    from dotenv import load_dotenv
    import os
    
    # Load environment variables from .env file
    load_dotenv()
    
    langsmith_config = config.get('langsmith', {})
    llm_config = config.get('llm', {})
    
    tracing_enabled = langsmith_config.get('tracing_enabled', False) or llm_config.get('enabled', False)
    
    # Check for API key in environment variables (try both LANGSMITH_API_KEY and LANGCHAIN_API_KEY)
    api_key = os.getenv('LANGCHAIN_API_KEY') or os.getenv('LANGSMITH_API_KEY')
    
    if api_key and tracing_enabled:
        os.environ['LANGCHAIN_TRACING_V2'] = 'true'
        os.environ['LANGCHAIN_API_KEY'] = api_key
        os.environ['LANGCHAIN_PROJECT'] = langsmith_config.get('project', 'agentic_forecast')
        print("LangSmith tracing enabled")
    else:
        os.environ['LANGCHAIN_TRACING_V2'] = 'false'
        print(" LangSmith tracing disabled")

def main():
    """Main entry point for the agentic_forecast application"""

    # Configure logging to reduce verbosity during evaluation
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s:%(message)s',
        handlers=[
            logging.FileHandler('logs/daily_pipeline.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    # Setup GPU
    setup_gpu()

    def signal_handler(signum, frame):
        """Handle graceful shutdown on SIGINT (Ctrl+C)"""
        print("\nReceived interrupt signal. Shutting down gracefully...")
        print("Cleaning up connections...")
        sys.exit(0)

    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    print("Initializing agentic_forecast Agentic Framework...")
    
    config = load_config()
    
    # Setup LangSmith tracing
    setup_langsmith(config)
    
    # Load symbols from main watchlist CSV
    max_symbols = config.get('scaling', {}).get('max_symbols', None)
    symbols = load_symbols_from_csv(max_symbols=max_symbols)
    if not symbols:
        print("Error: Could not load symbols from watchlist_main.csv")
        print("   Please ensure the CSV file exists with a 'Symbol' column")
        return
    
    print(f"Loaded {len(symbols)} symbols from main watchlist for processing")
    
    # Handle different tasks
    if args.task == "full":
        run_type = RunType(args.run_type)
        ctx = RunContext.create(run_type=run_type)
        
        logging.info("Starting run", extra={"run_type": ctx.run_type.value, "run_id": ctx.run_id})

        # Initialize Orchestrator
        orchestrator = OrchestratorAgent(config=config)

        # Define Routing Logic
        def route_after_features(state: PipelineGraphState):
            decision = orchestrator.coordinate_workflow(state)
            logger.info(f"Orchestrator decision after features: {decision}")
            if decision == "hpo":
                return "hpo"
            return "forecasting"

        def route_after_analytics(state: PipelineGraphState):
            decision = orchestrator.coordinate_workflow(state)
            logger.info(f"Orchestrator decision after analytics: {decision}")
            if decision == "retrain":
                return "strategy"
            elif decision == "end":
                return END
            return "strategy" # Default continue to strategy node

        # --- Build the Graph ---
        workflow = StateGraph(PipelineGraphState)

        # Add nodes
        workflow.add_node("data_ingestion", data_ingestion_node)
        workflow.add_node("macro_data", macro_data_node)
        workflow.add_node("regime_detection", regime_detection_node)
        workflow.add_node("feature_engineering", feature_engineering_node)
        workflow.add_node("hpo", hpo_node)
        workflow.add_node("forecasting", forecasting_node)
        workflow.add_node("analytics", analytics_node)
        workflow.add_node("strategy", strategy_node)
        workflow.add_node("portfolio_construction", portfolio_construction_node)
        workflow.add_node("monitoring", monitoring_node)
        workflow.add_node("retraining", retraining_node)
        workflow.add_node("reporting", generate_report_node)

        # Define edges
        workflow.set_entry_point("data_ingestion")
        workflow.add_edge("data_ingestion", "macro_data")
        workflow.add_edge("macro_data", "regime_detection")
        workflow.add_edge("regime_detection", "feature_engineering")
        
        # Conditional routing after feature engineering
        workflow.add_conditional_edges(
            "feature_engineering",
            route_after_features,
            {
                "hpo": "hpo",
                "forecasting": "forecasting"
            }
        )
        
        workflow.add_edge("hpo", "forecasting")
        workflow.add_edge("forecasting", "analytics")
        
        # Conditional routing after analytics
        workflow.add_conditional_edges(
            "analytics",
            route_after_analytics,
            {
                "strategy": "strategy",
                END: END
            }
        )
        
        workflow.add_edge("strategy", "portfolio_construction")
        workflow.add_edge("portfolio_construction", "monitoring")
        workflow.add_edge("monitoring", "retraining")
        workflow.add_edge("retraining", "reporting")
        workflow.add_edge("reporting", END)

        # Compile
        app = workflow.compile()

        # Initial state
        initial_state = PipelineGraphState(
            symbols=symbols,
            start_date="2020-01-01",
            end_date="2023-01-01",
            run_id=ctx.run_id,
            config=config,
            run_type=ctx.run_type.value,
            hpo_triggered=True, # Enable HPO by default for now to verify flow
            drift_detected=[],
            retrained_models=[],
            hpo_results={},
            errors=[],
            run_status="ACTIVE",
            next_step="",
            deep_research_conducted=False,
            horizon_forecasts={},
            interpreted_forecasts=False
        )

        # Run
        print("Starting Graph Execution...")
        final_state = app.invoke(initial_state)
        print("Graph Execution Completed.")
        
    elif args.task == "llm_analytics_only":
        from src.analytics.llm_analytics_orchestrator import run_llm_analytics_explainer
        print("Running LLM analytics explainer only...")
        explanation = run_llm_analytics_explainer()
        print("LLM analytics explainer completed.")
        print(f"Summary keys: {list(explanation.keys())}")
    else:
        raise ValueError(f"Unknown task: {args.task}")

if __name__ == "__main__":
    main()
