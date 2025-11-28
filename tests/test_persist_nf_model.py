import os
import pickle
from pathlib import Path

from models.model_zoo import ModelZoo, ArtifactInfo


def test_persist_nf_model_writes_file(tmp_path):
    mz = ModelZoo()
    model_id = "abc123"
    model_family = "TestFamily"
    model_obj = {"foo": "bar"}

    # temporarily override models directory to tmp_path
    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        res = mz._persist_nf_model(model_id, model_obj, model_family)
        # Should return ArtifactInfo
        assert type(res).__name__ == 'ArtifactInfo'
        lp = res.local_path
        if lp is not None:
            assert Path(lp).exists()
            if Path(lp).is_dir():
                # It's a directory (standard for NF models)
                # Check if we have content inside
                assert any(Path(lp).iterdir())
            else:
                # It's a file (legacy or specific cases)
                with open(lp, "rb") as fh:
                    unpickled = pickle.load(fh)
                assert unpickled == model_obj
        else:
            # If local_path is None, ensure we at least have an artifact URI
            assert isinstance(res.artifact_uri, str) and res.artifact_uri
    finally:
        os.chdir(old_cwd)
