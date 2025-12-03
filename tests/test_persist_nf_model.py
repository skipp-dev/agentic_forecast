import os
import pickle
from pathlib import Path

from models.model_zoo import ModelZoo


def test_persist_nf_model_writes_file(tmp_path):
    mz = ModelZoo()
    model_id = "abc123"
    model_family = "TestFamily"
    
    class MockModel:
        def save(self, path):
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "model.pkl"), "wb") as f:
                pickle.dump({"foo": "bar"}, f)

    model_obj = MockModel()

    # temporarily override models directory to tmp_path
    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        res = mz._persist_nf_model(model_id, model_obj, model_family)
        # Should return ArtifactInfo
        from models.model_zoo import ArtifactInfo
        assert isinstance(res, ArtifactInfo)
        lp = res.local_path
        assert Path(lp).exists()
        assert Path(lp).is_dir()
        # Check if we have content inside
        assert any(Path(lp).iterdir())
        
        # Check content
        with open(os.path.join(lp, "model.pkl"), "rb") as fh:
            unpickled = pickle.load(fh)
        assert unpickled == {"foo": "bar"}
    finally:
        os.chdir(old_cwd)

