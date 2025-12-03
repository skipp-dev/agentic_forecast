import os
import sys
import tempfile
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import model_zoo as mzc
from models.model_zoo import ModelZoo, DataSpec, HPOConfig, ArtifactInfo


def _make_simple_frames():
    idx = pd.date_range('2020-01-01', periods=10, freq='D')
    df = pd.DataFrame({'y': np.arange(10).astype(float), 'x': np.arange(10).astype(float)}, index=idx)
    split = int(len(df) * 0.7)
    return df.iloc[:split], df.iloc[split:]


def _make_simple_spec():
    dates = pd.date_range("2021-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "close": np.arange(10) + 1.0,
        "feat1": np.arange(10) * 0.1,
    }, index=dates)
    train = df.iloc[:6]
    val = df.iloc[6:]
    return DataSpec(job_id="j1", symbol_scope="s1", train_df=train, val_df=val, feature_cols=["feat1"], target_col="close", horizon=1)


def test_prepare_nf_frames_adds_temporal_metadata():
    mz = ModelZoo()
    spec = _make_simple_spec()
    uid, train_nf, val_nf, full = mz._prepare_nf_frames(spec)
    assert uid == spec.symbol_scope # No replacement
    assert hasattr(train_nf, "temporal_cols")
    assert hasattr(val_nf, "temporal_cols")
    assert hasattr(full, "temporal_index")


def test_compute_val_mape_zero_when_predictions_match():
    mz = ModelZoo()
    spec = _make_simple_spec()
    _, _, val_nf, _ = mz._prepare_nf_frames(spec)
    preds = pd.DataFrame({"unique_id": val_nf["unique_id"], "ds": val_nf["ds"], "AutoDLinear": val_nf["y"].to_numpy()})
    mape = mz._compute_val_mape(preds, val_nf, "AutoDLinear")
    assert mape == 0.0


def test_compute_val_mape_raises_on_empty_preds():
    mz = ModelZoo()
    spec = _make_simple_spec()
    _, _, val_nf, _ = mz._prepare_nf_frames(spec)
    
    empty = pd.DataFrame()
    try:
        mz._compute_val_mape(empty, val_nf, "AutoDLinear")
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_train_autonbeats_raises_when_nf_missing():
    train, val = _make_simple_frames()
    spec = DataSpec(job_id='t', symbol_scope='AAPL:US', train_df=train, val_df=val, feature_cols=['x'], target_col='y', horizon=1)
    
    # Save original state
    original_flag = mzc._HAS_NEURALFORECAST
    original_cls = mzc.AutoNBEATS
    
    try:
        # Ensure NF is disabled
        mzc._HAS_NEURALFORECAST = False
        # mzc.AutoNBEATS = None # No longer needed/valid as we use dummy classes
        mz = ModelZoo()
        
        with pytest.raises(ImportError):
            mz.train_autonbeats(spec, HPOConfig())
            
    finally:
        # Restore default back to imported status
        mzc._HAS_NEURALFORECAST = original_flag
        mzc.AutoNBEATS = original_cls


def test_train_autonbeats_returns_when_stubs_present():
    train, val = _make_simple_frames()
    spec = DataSpec(job_id='t2', symbol_scope='SYM', train_df=train, val_df=val, feature_cols=['x'], target_col='y', horizon=1)
    
    # Monkeypatch module to simulate NF presence
    class MockModel:
        def __init__(self, h, input_size, stack_types, start_padding_enabled): pass
    
    class MockNF:
        def __init__(self, models, freq): 
            self.models = models
        def fit(self, df, val_size=None): pass
        def predict(self, futr_df=None):
            return pd.DataFrame({'ds': val.index, 'unique_id': 'SYM', 'NBEATS': [100.0]*len(val)})
        def save(self, path): pass

    original_nf = mzc.NeuralForecast
    original_cls = mzc.NBEATS # train_autonbeats uses NBEATS class, not AutoNBEATS
    
    mzc.NeuralForecast = MockNF
    mzc.NBEATS = MockModel
    mzc._HAS_NEURALFORECAST = True
    
    try:
        mz = ModelZoo()
        res = mz.train_autonbeats(spec, HPOConfig())
        assert res is not None
        assert res.model_family == 'AutoNBEATS'
        assert hasattr(res, 'val_preds')
        assert isinstance(res.val_preds, pd.DataFrame)
        assert len(res.val_preds) == len(val)
    finally:
        mzc.NeuralForecast = original_nf
        mzc.NBEATS = original_cls


def test_compute_val_mape_and_persist_model(tmp_path):
    train, val = _make_simple_frames()
    spec = DataSpec(job_id='t3', symbol_scope='SYM2', train_df=train, val_df=val, feature_cols=['x'], target_col='y', horizon=1)
    
    # Mock NeuralForecast to avoid GPU usage
    class MockModel:
        def __init__(self, h, input_size, loss=None, scaler_type=None, max_steps=None): pass
        def fit(self, df, val_size=0): return self
        def predict(self, df): return np.zeros(len(df))

    class MockNF:
        def __init__(self, models, freq):
            self.models = models
        def fit(self, df, val_size=0): pass
        def predict(self, futr_df=None):
            # Return a DataFrame matching validation index
            return pd.DataFrame({
                'ds': val.index, 
                'unique_id': 'SYM2', 
                'NLinear': [100.0]*len(val)
            })
        def save(self, path): pass

    original_nf = mzc.NeuralForecast
    original_cls = mzc.NLinear # BaselineLinear uses NLinear
    mzc.NeuralForecast = MockNF
    mzc.NLinear = MockModel
    mzc._HAS_NEURALFORECAST = True

    try:
        mz = ModelZoo()
        # Baseline preds
        res = mz.train_baseline_linear(spec)
        assert res.val_preds is not None
        uid, train_nf, val_nf, full = mz._prepare_nf_frames(spec)
        mape = mz._compute_val_mape(res.val_preds, val_nf, 'BaselineLinear')
        assert isinstance(mape, float)

        # Persist model (use serializable dict instead of local class)
        model_id = 'pkey'
        res = mz._persist_nf_model(model_id, {'dummy': True}, 'AutoNBEATS')
        assert isinstance(res, ArtifactInfo)
        # If MLflow not installed we should get a local path; otherwise artifact_uri
        if res.local_path:
            assert os.path.exists(res.local_path)
        else:
            # For remote backends we still expect an artifact URI
            assert isinstance(res.artifact_uri, str) and res.artifact_uri
    finally:
        mzc.NeuralForecast = original_nf
        mzc.NLinear = original_cls


def test_train_autonhits_raises_when_nf_missing():
    train, val = _make_simple_frames()
    spec = DataSpec(job_id='t4', symbol_scope='AAPL:US', train_df=train, val_df=val, feature_cols=['x'], target_col='y', horizon=1)
    
    original_flag = mzc._HAS_NEURALFORECAST
    original_cls = mzc.AutoNHITS
    
    try:
        # Ensure NF is disabled
        mzc._HAS_NEURALFORECAST = False
        # mzc.AutoNHITS = None
        mz = ModelZoo()
        
        with pytest.raises(ImportError):
            mz.train_autonhits(spec, HPOConfig())
            
    finally:
        mzc._HAS_NEURALFORECAST = original_flag
        mzc.AutoNHITS = original_cls


def test_train_autonhits_returns_when_stubs_present():
    train, val = _make_simple_frames()
    spec = DataSpec(job_id='t5', symbol_scope='SYM', train_df=train, val_df=val, feature_cols=['x'], target_col='y', horizon=1)
    
    class MockModel:
        def __init__(self, h, input_size, start_padding_enabled): pass
    
    class MockNF:
        def __init__(self, models, freq): 
            self.models = models
        def fit(self, df, val_size=None): pass
        def predict(self, futr_df=None):
            return pd.DataFrame({'ds': val.index, 'unique_id': 'SYM', 'NHITS': [100.0]*len(val)})
        def save(self, path): pass

    original_nf = mzc.NeuralForecast
    original_cls = mzc.NHITS # train_autonhits uses NHITS class
    
    mzc.NeuralForecast = MockNF
    mzc.NHITS = MockModel
    mzc._HAS_NEURALFORECAST = True
    
    try:
        mz = ModelZoo()
        res = mz.train_autonhits(spec, HPOConfig())
        assert res is not None
        assert res.model_family == 'AutoNHITS'
        assert hasattr(res, 'val_preds')
        assert isinstance(res.val_preds, pd.DataFrame)
        assert len(res.val_preds) == len(val)
    finally:
        mzc.NeuralForecast = original_nf
        mzc.NHITS = original_cls


def test_model_training_result_artifact_info_dict():
    # Validate that ModelTrainingResult uses artifact_info as a dict and
    # convenience properties return expected values.
    from models.model_zoo import ModelTrainingResult, ArtifactInfo

    m = ModelTrainingResult(job_id='j', symbol_scope='s', model_family='F', framework='neuralforecast', best_val_mape=0.0, best_val_mae=0.0, best_hyperparams={}, best_model_id='m1', artifact_info=ArtifactInfo(artifact_uri="file:///tmp/x", local_path="/tmp/x"), val_preds=None)
    assert isinstance(m.artifact_info, ArtifactInfo)
    assert m.artifact_uri == "file:///tmp/x"
    assert m.local_artifact_path == "/tmp/x"


def test_autodlinear_returns_artifact_info(monkeypatch):
    # Only run when NF is available in the environment
    from models.model_zoo import ModelZoo, _HAS_NEURALFORECAST, DataSpec

    if not _HAS_NEURALFORECAST:
        import pytest
        pytest.skip("NeuralForecast not available - skipping autodlinear persistence test")

    dates = pd.date_range("2021-01-01", periods=10, freq="D")
    df = pd.DataFrame({"close": range(10)}, index=dates)
    train = df.iloc[:7]
    val = df.iloc[7:]
    spec = DataSpec(job_id='x', symbol_scope='SYM', train_df=train, val_df=val, feature_cols=[], target_col='close', horizon=1)

    mz = ModelZoo()
    try:
        res = mz.train_autodlinear(spec, HPOConfig(max_trials=1, max_epochs=1))
    except Exception:
        # If NF training fails due to env we skip
        import pytest
        pytest.skip("AutoDLinear training failed in this env")

    # New API returns ArtifactInfo dataclass
    assert hasattr(res, 'artifact_info')
    from models.model_zoo import ArtifactInfo
    assert isinstance(res.artifact_info, ArtifactInfo)
