"""summarize.py on a small fake analysis: tables, aggregates, TWO-NN per layer and sample, PCA hexbins, nothing that holds raw vectors."""
import json, os, subprocess, sys
import numpy as np
import yaml
from conftest import ROOT, load
store = load("store")

def fake_analysis(a, n=150, dim=12):
    rng = np.random.default_rng(0)
    for si, s in enumerate(("s1", "s2")):
        (a / "fasta").mkdir(parents=True, exist_ok=True)
        ids, lens = [f"{s}_{i}" for i in range(n)], rng.integers(30, 600, n)
        (a / "fasta" / f"{s}.tsv").write_text("record\tlength\tstatus\n" + "".join(f"{i}\t{l}\tkept\n" for i, l in zip(ids, lens)) + f"{s}_bad\t0\tinvalid:empty\n")
        (a / "fasta" / f"{s}.json").write_text(json.dumps(dict(source=f"/x/{s}.fasta", records=n + 1, kept=n, status={"kept": n, "invalid:empty": 1}, fasta_sha256="0" * 64, uniprot=None)))
        q, _ = np.linalg.qr(rng.normal(size=(dim, 3)))
        X1, X2 = (rng.random((n, 3)) @ q.T + si).astype(np.float32), (rng.random((n, 3)) @ q.T * 2 + si).astype(np.float32)
        d = a / "chunks" / s
        store.write_chunk(d, 0, {"layer_1/mean": X1, "layer_2/mean": X2, "lengths": lens.astype(np.int32), "truncated": np.zeros(n, np.uint8)}, ids, 0, 1)
        b = a / "embeddings" / "fake_12"
        store.merge(store.chunks(d), b, s, 10**9, dict(backend="fake_12"))
        (b / f"{s}.json").write_text(json.dumps(dict(dim=dim, store_dtype="float32", compute_dtype="fp32", truncated=0, revision="a" * 40, input_sha256="1" * 64, shards=[dict(proteins=n, elapsed_s=10.0)])))

def test_summary_tables_twonn_and_pca(tmp_path):
    a = tmp_path / "results" / "x"; fake_analysis(a)
    cfg = tmp_path / "p.yaml"; cfg.write_text(yaml.safe_dump(dict(summarize=dict(pairs=200, twonn=dict(enabled=True, max_points=200, n_min=20), pca=dict(enabled=True, per_sample_cap=50, hex_bins=10, per_sample_pca=True)))))
    env = dict(os.environ, MP_DADAPY_PYTHON=sys.executable, PYTHONPATH=os.path.join(ROOT, "tests", "stub_dadapy"))
    r = subprocess.run([sys.executable, f"{ROOT}/src/summarize.py", "--analysis_dir", str(a), "--yaml-in", str(cfg)], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "DADApy 0.3.4" in r.stdout and not list(a.glob(".twonn_*"))
    agg = json.loads((a / "aggregates.json").read_text())
    rows = agg["twonn"]["rows"]
    assert {(x["layer"], x["sample"]) for x in rows} == {(l, s) for l in (1, 2) for s in ("s1", "s2", "all samples")}
    assert all(x["engine"] == "dadapy 0.3.4" and x["points"] in (150, 200) and x["decimation"][0]["points"] == x["points"] for x in rows)
    P = agg["pca"]["fake_12"]["layer_2/mean"]
    assert P["components_90"] <= 4 and sum(c[3] for c in P["hex"]["s1"]) + P["clipped"]["s1"] >= 150 and len(P["confusion"]) == 2 and "per_sample" in P
    assert agg["throughput"]["fake_12"]["proteins"] == 300 and (a / "intrinsic_dimension.tsv").read_text().count("\n") == 7
    text = (a / "aggregates.json").read_text()
    assert len(text) < 200_000 and "layer_1/tokens" not in text

def test_missing_dadapy_env_stops_early(tmp_path):
    a = tmp_path / "results" / "x"; fake_analysis(a)
    cfg = tmp_path / "p.yaml"; cfg.write_text(yaml.safe_dump(dict(summarize=dict(twonn=dict(enabled=True)))))
    r = subprocess.run([sys.executable, f"{ROOT}/src/summarize.py", "--analysis_dir", str(a), "--yaml-in", str(cfg)], capture_output=True, text=True, env=dict(os.environ, MP_DADAPY_PYTHON=str(tmp_path / "missing")))
    assert r.returncode != 0 and "needs the dadapy env" in r.stdout + r.stderr and not (a / "aggregates.json").exists()

def run_summary(tmp_path, n, cap):
    a = tmp_path / "results" / "x"; fake_analysis(a, n=n)
    cfg = tmp_path / "p.yaml"; cfg.write_text(yaml.safe_dump(dict(summarize=dict(pairs=50, pca=dict(enabled=True, per_sample_cap=50, hex_bins=10), embed2d=dict(enabled=True, max_points=cap)))))
    env = dict(os.environ, MP_DADAPY_PYTHON=sys.executable, PYTHONPATH=os.path.join(ROOT, "tests", "stub_dadapy"))
    r = subprocess.run([sys.executable, f"{ROOT}/src/summarize.py", "--analysis_dir", str(a), "--yaml-in", str(cfg)], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads((a / "aggregates.json").read_text())

def test_embed2d_below_limit_shows_every_point(tmp_path):
    agg = run_summary(tmp_path, 100, 1000)
    E, P = agg["embed2d"]["fake_12"]["layer_2/mean"], agg["pca"]["fake_12"]["layer_2/mean"]
    assert E["points"] == 200 == len(E["tsne"]) == len(E["umap"]) == len(E["labels"]) and E["samples"] == ["s1", "s2"]
    assert sum(len(v) for v in P["points"].values()) == 200

def test_embed2d_at_limit_is_skipped(tmp_path):
    agg = run_summary(tmp_path, 100, 200)
    assert agg["embed2d"]["fake_12"]["layer_2/mean"] == {"skipped": "n >= 200"}
    assert "points" not in agg["pca"]["fake_12"]["layer_2/mean"]

def test_embed2d_agreement_across_backends(tmp_path):
    import shutil
    a = tmp_path / "results" / "x"; fake_analysis(a, n=60)
    shutil.copytree(a / "embeddings" / "fake_12", a / "embeddings" / "fake_13")
    cfg = tmp_path / "p.yaml"; cfg.write_text(yaml.safe_dump(dict(summarize=dict(pairs=20, embed2d=dict(enabled=True, max_points=1000)))))
    env = dict(os.environ, MP_DADAPY_PYTHON=sys.executable, PYTHONPATH=os.path.join(ROOT, "tests", "stub_dadapy"))
    r = subprocess.run([sys.executable, f"{ROOT}/src/summarize.py", "--analysis_dir", str(a), "--yaml-in", str(cfg)], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    g = json.loads((a / "aggregates.json").read_text())["embed2d_agreement"]
    assert g["backends"] == ["fake_12", "fake_13"] and g["proteins"] == 120 and g["rsa"][0][1] > 0.999 and g["knn"][0][1] == 1.0
