import sys
import pytest

def test_path_debug():
    print("\nSYS.PATH:")
    for p in sys.path:
        print(p)
    
    import pipelines
    print(f"\nIMPORTED PIPELINES: {pipelines}")
    
    try:
        import models
        print(f"IMPORTED MODELS: {models}")
        if hasattr(models, '__file__'):
            print(f"MODELS FILE: {models.__file__}")
    except ImportError as e:
        print(f"IMPORT MODELS FAILED: {e}")
