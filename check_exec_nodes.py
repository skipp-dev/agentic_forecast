import sys
import os

# Add project root
sys.path.append(os.getcwd())

try:
    print("Importing src.nodes.execution_nodes...")
    from src.nodes import execution_nodes
    print(f"_HAS_HEAVY_DEPS: {execution_nodes._HAS_HEAVY_DEPS}")
    print(f"NeuralForecast: {execution_nodes.NeuralForecast}")
except Exception as e:
    print(f"Error importing execution_nodes: {e}")
    import traceback
    traceback.print_exc()
