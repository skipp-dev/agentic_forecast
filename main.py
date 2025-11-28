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

def load_config():
    """Load configuration from config.yaml"""
    config_path = "config.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

def load_symbols_from_csv(csv_path="watchlist_ibkr.csv"):
    """Load symbols from IBKR watchlist CSV file"""
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if 'Symbol' in df.columns:
                # Process all symbols from the watchlist
                return df['Symbol'].tolist()
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
        print("ðŸ” Detecting GPU devices...")

        # List all physical devices
        gpus = tf.config.list_physical_devices('GPU')
        cpus = tf.config.list_physical_devices('CPU')

        print(f"ðŸ“Š Available devices:")
        print(f"   GPUs: {len(gpus)}")
        print(f"   CPUs: {len(cpus)}")

        if gpus:
            print(f"   GPU Details: {[gpu.name for gpu in gpus]}")

            # Configure each GPU for optimal utilization
            for i, gpu in enumerate(gpus):
                try:
                    # Enable memory growth to prevent TensorFlow from allocating all GPU memory
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"   âœ… GPU {i}: Memory growth enabled")

                    # Set GPU device visibility
                    tf.config.set_visible_devices([gpu], 'GPU')
                    print(f"   âœ… GPU {i}: Device visibility set")

                except RuntimeError as e:
                    print(f"   âš ï¸  GPU {i} configuration error: {e}")

            # Enable mixed precision for better GPU utilization (2x-4x speedup)
            try:
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                print("   âœ… Mixed precision training enabled (float16) - 2-4x speedup expected")      
            except Exception as e:
                print(f"   âš ï¸  Mixed precision setup failed: {e}")

            # Verify GPU is accessible with actual computation
            try:
                with tf.device('/GPU:0'):
                    test_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                    result = tf.matmul(test_tensor, test_tensor)
                    print(f"   âœ… GPU computation test passed: {result.shape}")
            except Exception as e:
                print(f"   âš ï¸  GPU computation test failed: {e}")

            print("ðŸŽ‰ GPU setup complete - training will use GPU acceleration!")
            return True

        else:
            print("âš ï¸  No GPU devices found - training will use CPU")
            print("ðŸ’¡ For GPU support:")
            print("   1. Ensure NVIDIA GPU is installed")
            print("   2. Install NVIDIA drivers: https://www.nvidia.com/Download/index.aspx")
            print("   3. Install NVIDIA Container Toolkit: https://github.com/NVIDIA/nvidia-docker")    
            print("   4. Run with docker-compose: docker-compose up --build")
            print("   ðŸ“– See GPU_SETUP_GUIDE.md for detailed instructions")

            # Suppress CUDA warnings only when no GPU is available
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            return False

    except Exception as e:
        print(f"âŒ GPU setup failed: {e}")
        print("ðŸ”„ Falling back to CPU training")
        # Suppress CUDA warnings on error
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        return False

def setup_langsmith(config):
    """Setup LangSmith tracing"""
    langsmith_config = config.get('langsmith', {})
    tracing_enabled = os.getenv('LANGCHAIN_TRACING_V2', 'false').lower() == 'true'

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
        print("\n🛑 Received interrupt signal. Shutting down gracefully...")
        print("Cleaning up connections...")
        # Force exit to avoid IB cleanup issues during shutdown
        sys.exit(0)

    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    print("🚀 Initializing agentic_forecast Agentic Framework...")
    
    config = load_config()
    app = create_main_graph(config)
    
    print("\n📈 Running the agentic workflow...")
    
    # Load symbols from IBKR watchlist CSV
    symbols = load_symbols_from_csv()
    if not symbols:
        print("❌ Error: Could not load symbols from watchlist_ibkr.csv")
        print("   Please ensure the CSV file exists with a 'Symbol' column")
        return
    
    print(f"📊 Loaded {len(symbols)} symbols from IBKR watchlist for processing")
    
    initial_state = GraphState(
        symbols=symbols,
        config=config,
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
    
    for output in app.stream(initial_state):
        for key, value in output.items():
            logger.info(f"Output from node '{key}':")
            logger.info("---")
            logger.info(str(value))
        logger.info("\n---\n")

    logger.info("Agentic workflow finished.")

if __name__ == "__main__":
    main()

