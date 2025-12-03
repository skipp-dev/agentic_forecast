import os
import sys
import pandas as pd
import numpy as np
import unittest
from unittest.mock import patch, MagicMock

# Ensure imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.model_zoo import ModelZoo, DataSpec, HPOConfig, ModelTrainingResult
from src.agents.hyperparameter_search_agent import HyperparameterSearchAgent

def _make_sample_frames():
    dates = pd.date_range('2023-01-01', periods=10, freq='D')
    df = pd.DataFrame({'close': np.arange(10).astype(float), 'volume': np.arange(10)*10}, index=dates)  
    split = int(len(df) * 0.8)
    train = df.iloc[:split]
    val = df.iloc[split:]
    return train, val

class TestModelZooUnit(unittest.TestCase):
    def test_prepare_nf_frames_attaches_temporal_attrs(self):
        train, val = _make_sample_frames()
        spec = DataSpec(job_id='j', symbol_scope='TEST:SYM', train_df=train, val_df=val, feature_cols=['volume'], target_col='close', horizon=1)
        mz = ModelZoo()
        uid, train_nf, val_nf, full_df = mz._prepare_nf_frames(spec)

        assert 'temporal_cols' in dir(train_nf)
        assert 'temporal_index' in dir(train_nf)
        assert isinstance(getattr(train_nf, 'temporal_cols'), np.ndarray)
        assert isinstance(getattr(train_nf, 'temporal_index'), np.ndarray)

    @patch('models.model_zoo.NeuralForecast')
    def test_baseline_val_preds_format(self, mock_nf_cls):
        train, val = _make_sample_frames()
        spec = DataSpec(job_id='jid', symbol_scope='SYM:TEST', train_df=train, val_df=val, feature_cols=['volume'], target_col='close', horizon=1)
        
        # Mock NeuralForecast instance and predict return
        mock_nf = mock_nf_cls.return_value
        mock_nf.predict.return_value = pd.DataFrame({
            'ds': val.index,
            'unique_id': 'SYM:TEST',
            'NLinear': [100.0] * len(val)
        })

        mz = ModelZoo()
        res = mz.train_baseline_linear(spec)

        assert hasattr(res, 'val_preds')
        preds = res.val_preds
        assert isinstance(preds, pd.DataFrame)
        assert 'unique_id' in preds.columns
        assert 'ds' in preds.columns
        # One prediction per validation row
        assert len(preds) == len(val)

    def test_hpo_families_include_autodlinear_and_nbeats(self):
        agent = HyperparameterSearchAgent()
        # Ensure expected families are present
        families = getattr(agent, 'model_families', None)
        assert families is not None
        assert 'AutoNBEATS' in families
        assert 'AutoDLinear' in families

if __name__ == '__main__':
    unittest.main()
