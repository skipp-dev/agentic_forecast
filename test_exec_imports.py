import os
import sys

print("Testing execution_nodes.py imports...")

try:
    import neuralforecast as nf
    print("import neuralforecast as nf: OK")
    
    from neuralforecast import NeuralForecast
    print("from neuralforecast import NeuralForecast: OK")
    
    from neuralforecast.auto import AutoNHITS, AutoNBEATS, AutoDLinear, AutoTFT
    print("from neuralforecast.auto import AutoNHITS...: OK")
    
    from neuralforecast.models import NLinear
    print("from neuralforecast.models import NLinear: OK")
    
    import torch
    print("import torch: OK")
    
    # We can't easily test models.gnn_model without the full path context, 
    # but let's try adding the path
    sys.path.append(os.getcwd())
    try:
        from models.gnn_model import GNNModel
        print("from models.gnn_model import GNNModel: OK")
    except ImportError:
        print("from models.gnn_model import GNNModel: FAILED (might be expected if file missing)")
    except Exception as e:
        print(f"from models.gnn_model import GNNModel: ERROR {e}")

except Exception as e:
    print(f"IMPORT FAILED: {e}")
    import traceback
    traceback.print_exc()
