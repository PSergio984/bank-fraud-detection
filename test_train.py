import os
import pickle
import subprocess
import sys


def test_artifact_contract(tmp_path):
    out = tmp_path / "model.bin"
    env = {**os.environ, "TRAIN_SAMPLE": "50000", "OUTPUT": str(out)}
    r = subprocess.run([sys.executable, "train.py"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-2000:]

    art = pickle.loads(out.read_bytes())
    assert {"model", "dv", "threshold", "features", "medians",
            "metrics"} <= set(art)
    assert 0 < art["threshold"] < 1
    assert out.stat().st_size < 100 * 1024**2
    assert isinstance(art["medians"], dict) and art["medians"]
    assert isinstance(art["features"], list) and art["features"]
