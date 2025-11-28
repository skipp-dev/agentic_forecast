import sys, os

# Ensure project's root is importable as top-level for tests running in
# isolated pytest environments.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import sys
from unittest.mock import MagicMock

class MockTalib:
    def __getattr__(self, name):
        return MagicMock()

sys.modules['talib'] = MockTalib()
sys.modules['GPUtil'] = MagicMock()

