import sys
import torch
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

print("-" * 20)
print("Attempting to import NeuralForecast...")

try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS, TFT
    from neuralforecast.auto import AutoNHITS, AutoTFT
    print("SUCCESS: NeuralForecast imported.")
except ImportError as e:
    print(f"FAILURE: Could not import NeuralForecast. Error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"FAILURE: Unexpected error during import. Error: {e}")
    traceback.print_exc()
    sys.exit(1)

print("-" * 20)
print("Attempting to instantiate AutoNHITS...")

try:
    # Instantiate with minimal parameters
    # AutoNHITS(h, loss=None, valid_loss=None, config=None, search_alg=None, num_samples=10, cpus=1, gpus=0, verbose=False, alias=None, backend='ray')
    # We'll try a simple instantiation.
    model = AutoNHITS(h=24, num_samples=1, cpus=1, gpus=1 if torch.cuda.is_available() else 0)
    print("SUCCESS: AutoNHITS instantiated.")
except Exception as e:
    print(f"FAILURE: Could not instantiate AutoNHITS. Error: {e}")
    traceback.print_exc()
    sys.exit(1)

print("-" * 20)
print("NeuralForecast check passed!")
