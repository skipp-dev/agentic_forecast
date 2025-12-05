import sys
import os
print("SYS.PATH:")
for p in sys.path:
    print(p)

try:
    import pipelines
    print(f"IMPORTED PIPELINES FROM: {pipelines.__file__}")
except ImportError:
    print("COULD NOT IMPORT PIPELINES")
except AttributeError:
    print(f"IMPORTED PIPELINES (Namespace): {pipelines}")
