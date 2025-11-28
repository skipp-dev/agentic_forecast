import os

from agents.hyperparameter_search_agent import HyperparameterSearchAgent
from models.model_zoo import ModelZoo, _HAS_NEURALFORECAST


def test_hpo_includes_autodlinear():
    agent = HyperparameterSearchAgent()
    assert 'AutoDLinear' in agent.model_families


def test_train_autodlinear_import_guard():
    mz = ModelZoo()
    # Train autodlinear should raise ImportError if NF not installed
    if not _HAS_NEURALFORECAST:
        try:
            mz.train_autodlinear(None, None)
        except ImportError:
            pass
        else:
            raise AssertionError("train_autodlinear should raise ImportError when NeuralForecast not available")
    else:
        # If NeuralForecast is installed, calling should not raise ImportError - but we can't assert full training in tests
        assert hasattr(mz, 'train_autodlinear')
