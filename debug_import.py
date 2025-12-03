import sys
import os
print(f"CWD: {os.getcwd()}")
print(f"sys.path: {sys.path}")
try:
    import models.model_zoo
    print("Successfully imported models.model_zoo")
except ImportError as e:
    print(f"Failed to import models.model_zoo: {e}")
