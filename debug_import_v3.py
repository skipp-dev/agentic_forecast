import sys
import os

print(f"Current working directory: {os.getcwd()}")
print(f"sys.path: {sys.path}")

try:
    import models
    print(f"Successfully imported models from: {models.__file__}")
    from models import model_zoo
    print(f"Successfully imported model_zoo from: {model_zoo.__file__}")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

if os.path.exists("models"):
    print("Directory 'models' exists.")
else:
    print("Directory 'models' DOES NOT exist.")

if os.path.exists("models/model_zoo.py"):
    print("File 'models/model_zoo.py' exists.")
else:
    print("File 'models/model_zoo.py' DOES NOT exist.")
