import pandas as pd

from models.model_zoo import ModelZoo, DataSpec


def _make_simple_data():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    df = pd.DataFrame({"y": range(10), "feature": range(10)}, index=idx)
    return df


def test_nf_helpers_present_and_work():
    mz = ModelZoo()
    train = _make_simple_data()
    val = _make_simple_data()
    ds = DataSpec(job_id="j1", symbol_scope="S:1", train_df=train, val_df=val, feature_cols=["feature"], target_col="y", horizon=3)
    unique_id, train_nf, val_nf, full = mz._prepare_nf_frames(ds)
    assert "ds" in train_nf.columns
    assert unique_id == "S:1"
    preds = val_nf[ ["unique_id", "ds"] ].copy()
    preds["AutoDLinear"] = val_nf["y"].to_numpy()
    mape = mz._compute_val_mape(preds, val_nf, "AutoDLinear")
    assert isinstance(mape, float)
