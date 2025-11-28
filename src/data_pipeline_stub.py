"""Isolated DataPipeline stub used exclusively for tests/HPO agent.

Avoids interference from any legacy mixed-content in data_pipeline.py.
"""

from typing import Tuple, Dict, Any
import numpy as np

class DataPipeline:
    def __init__(self):
        pass

    def train_cnn_lstm(self, symbol: str, period: str = '1y', epochs: int = 5,
                       batch_size: int = 32, verbose: int = 0) -> Tuple[None, Dict[str, Any]]:
        seed = abs(hash(symbol)) % (10**6)
        rng = np.random.default_rng(seed)
        mae = float(rng.uniform(0.5, 2.0))
        return None, {'test_metrics': {'mae': mae, 'mape': mae * 1.1}}

    def train_ensemble(self, symbol: str, period: str = '1y', verbose: int = 0) -> Tuple[None, Dict[str, Any]]:
        seed = abs(hash(symbol + 'ensemble')) % (10**6)
        rng = np.random.default_rng(seed)
        mae = float(rng.uniform(0.4, 1.5))
        return None, {'test_mae': mae, 'test_mape': mae * 1.15}
