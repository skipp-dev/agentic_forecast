# Entry point for the agentic_forecast application
import os
import yaml
import signal
import sys
import logging
import tensorflow as tf

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
# Note: CUDA_VISIBLE_DEVICES is set in .env file to allow GPU access

import asyncio
import pandas as pd
import torch
import argparse

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Agentic Forecast System')
    parser.add_argument('--task', choices=['full'], default='full',
                       help='Task to run (default: full)')
    parser.add_argument('--run_type', choices=['DAILY', 'WEEKEND_HPO', 'BACKTEST'],
                       default='DAILY', help='Type of run (default: DAILY)')
    return parser.parse_args()

def build_initial_state(symbols, config, run_type='DAILY'):
    """Build the initial GraphState with run_type"""
    return GraphState(
        symbols=symbols,
        config=config,
        run_type=run_type,  # Add run_type to state
        raw_data={},
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

def load_config():
    """Load configuration from config.yaml"""
    config_path = "config.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

def load_symbols_from_csv(csv_path="watchlist_ibkr.csv", max_symbols=None):
    """Load symbols from IBKR watchlist CSV file"""
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
            print("For GPU support:")
            print("   1. Ensure NVIDIA GPU is installed")
            print("   2. Install NVIDIA drivers: https://www.nvidia.com/Download/index.aspx")
            print("   3. Install CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit")
            print("   4. Install PyTorch with CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            print("   See GPU_SETUP_GUIDE.md for detailed instructions")

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
    langsmith_config = config.get('langsmith', {})
    llm_config = config.get('llm', {})
    
    tracing_enabled = langsmith_config.get('tracing_enabled', False) or llm_config.get('enabled', False)
    
    if langsmith_config.get('api_key') and tracing_enabled:
        os.environ['LANGCHAIN_TRACING_V2'] = 'true'
        os.environ['LANGCHAIN_API_KEY'] = langsmith_config['api_key']
        os.environ['LANGCHAIN_PROJECT'] = langsmith_config.get('project', 'agentic_forecast')
        print("✅ LangSmith tracing enabled")
    else:
        os.environ['LANGCHAIN_TRACING_V2'] = 'false'
        print("⚠️ LangSmith tracing disabled")

from src.graphs.main_graph import create_main_graph
from src.graphs.state import GraphState

def main():
    """Main entry point for the agentic_forecast application"""

    # Parse command line arguments
    args = parse_args()

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
        # Force exit to avoid IB cleanup issues during shutdown
        sys.exit(0)

    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    print("Initializing agentic_forecast Agentic Framework...")
    
    config = load_config()
    app = create_main_graph(config)
    
    print(f"\nRunning the agentic workflow with run_type: {args.run_type}...")
    
    # Load symbols from IBKR watchlist CSV
    # For testing, limit to first 5 symbols to avoid long data loading
    max_symbols = 5 if args.run_type == "WEEKEND_HPO" else None
    symbols = load_symbols_from_csv(max_symbols=max_symbols)
    if not symbols:
        print("Error: Could not load symbols from watchlist_ibkr.csv")
        print("   Please ensure the CSV file exists with a 'Symbol' column")
        return
    
    print(f"Loaded {len(symbols)} symbols from IBKR watchlist for processing")
    
    # Build initial state with run_type
    initial_state = build_initial_state(symbols, config, args.run_type)
    
    for output in app.stream(initial_state):
        for key, value in output.items():
            logger.info(f"Output from node '{key}':")
            logger.info("---")
            logger.info(str(value))
        logger.info("\n---\n")

    logger.info("Agentic workflow finished.")

if __name__ == "__main__":
    main()

