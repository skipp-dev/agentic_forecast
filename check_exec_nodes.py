import sys
import os

# Add project root
sys.path.append(os.getcwd())

try:
    print("Importing src.nodes.core_nodes...")
    from src.nodes import core_nodes
    print(f"_HAS_HEAVY_DEPS: {core_nodes._HAS_HEAVY_DEPS}")
    print(f"NeuralForecast: {core_nodes.NeuralForecast}")
except Exception as e:
    print(f"Error importing core_nodes: {e}")
    import traceback
    traceback.print_exc()
