import os
import sys
import traceback

# Add root to path for models import
sys.path.append(os.getcwd())

print("Attempting imports...")
try:
    if os.environ.get('SKIP_NEURALFORECAST', '').lower() not in ('true', '1', 'yes'):
        print("Importing neuralforecast...")
        import neuralforecast as nf
        print("Importing NeuralForecast class...")
        from neuralforecast import NeuralForecast
        print("Importing Auto models...")
        from neuralforecast.auto import AutoNHITS, AutoNBEATS, AutoDLinear, AutoTFT
        print("Importing NLinear...")
        from neuralforecast.models import NLinear
        print("Importing torch...")
        import torch
        print("Importing GNNModel...")
        try:
            from models.gnn_model import GNNModel
            print("GNNModel imported.")
        except ImportError:
            print("GNNModel import failed (caught).")
            GNNModel = None
        _HAS_HEAVY_DEPS = True
        print("All imports successful.")
    else:
        print("Skipping NeuralForecast imports.")
except Exception as e:
    print(f"Caught exception during imports: {type(e).__name__}: {e}")
    traceback.print_exc()
