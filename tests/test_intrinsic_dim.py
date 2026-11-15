"""intrinsic_dim.py with a stub DADApy (no jax on the host): cap, cosine normalisation, duplicate counting, small sets, output schema; the real-DADApy benchmarks run in the image (dadapy_benchmarks.py via build.sh test)."""
import json, os, subprocess, sys
import numpy as np
import pytest
from conftest import ROOT

STUB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stub_dadapy")

def run(tmp_path, arrays, **settings):
    S = {**dict(max_points=10000, mu_fraction=0.9, n_min=20, metric="euclidean"), **settings}
    tasks = []
    for i, X in enumerate(arrays):
        np.save(tmp_path / f"t{i}.npy", X); tasks.append(dict(name=f"t{i}", path=str(tmp_path / f"t{i}.npy")))
    (tmp_path / "m.json").write_text(json.dumps(dict(settings=S, tasks=tasks)))
    r = subprocess.run([sys.executable, f"{ROOT}/src/intrinsic_dim.py", "--manifest", str(tmp_path / "m.json"), "--out", str(tmp_path / "o.json")], capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=STUB))
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads((tmp_path / "o.json").read_text()), r.stdout

def test_schema_scale_curve_and_engine(tmp_path):
    out, log = run(tmp_path, [np.random.default_rng(0).random((400, 5))])
    r = out[0]
    assert r["engine"] == "dadapy 0.3.4" and "random (not disjoint)" in r["differences"] and "Facco" in log
    assert [l["points"] for l in r["decimation"]] == [400, 200, 100, 50, 25] and r["decimation"][0]["spread"] is None and r["error"] == 0.1

def test_duplicates_are_removed_and_counted(tmp_path):
    X = np.random.default_rng(1).random((100, 4)); Y = np.vstack([X, X[:30]])
    out, log = run(tmp_path, [Y, np.vstack([X, X[:5]])])
    assert out[0]["dropped"] == 30 and out[0]["dropped_warning"] and "unreliable" in log
    assert out[1]["dropped"] == 5 and not out[1]["dropped_warning"]

def test_cap_and_cosine_normalisation(tmp_path):
    X = 7 * np.random.default_rng(2).random((500, 6)) + 1
    out, _ = run(tmp_path, [X], max_points=150, metric="cosine")
    assert out[0]["points"] == 150 and out[0]["d"] == pytest.approx(1.0, abs=1e-3)

def test_too_few_points_give_no_estimate(tmp_path):
    out, _ = run(tmp_path, [np.random.default_rng(3).random((10, 3))])
    assert out[0]["d"] is None and out[0]["decimation"] == []
