"""Shard job end to end with the fake backend (no torch, no GPU): a killed shard resumes from its last complete chunk and the merged result equals an uninterrupted run."""
import json, os, subprocess, sys
import numpy as np
import pytest
from conftest import ROOT, load
store = load("store")

FASTA = "".join(f">p{i}\n{'MKVLAGHWYRTE'[: 3 + i % 9] * (1 + i % 3)}\n" for i in range(23))

@pytest.fixture
def world(tmp_path):
    tsv = tmp_path / "tools.tsv"
    head = open(f"{ROOT}/configs/tools.tsv").readline()
    tsv.write_text(head + "fake_4\tweights\tfake\tmp\thttps://example.invalid/fake\t" + "a" * 40 + "\tw.bin\t0\t4\t3\tall\ttest only\n")
    m = tmp_path / "models" / "fake_4"; (m / "files").mkdir(parents=True); (m / "files" / "w.bin").write_bytes(b"fake weights")
    import hashlib
    (m / "CHECKSUMS").write_text(f"{hashlib.sha256(b'fake weights').hexdigest()}  files/w.bin\n"); (m / "SOURCE.json").write_text(json.dumps(dict(revision="a" * 40)))
    fa = tmp_path / "in.fasta"; fa.write_text(FASTA)
    return tmp_path, dict(os.environ, MP_TOOLS_TSV=str(tsv))

def shard(world, out, k, K, fail=0, extra=(), **env_extra):
    tmp, env = world
    env = dict(env, MP_FAKE_FAIL_AFTER=str(fail), **env_extra) if fail else dict(env, **env_extra)
    return subprocess.run([sys.executable, f"{ROOT}/src/embed.py", "--backend", "fake_4", "--fasta", str(tmp / "in.fasta"), "--shard", str(k), "--shards", str(K), "--chunk_dir", str(out / f"shard_{k}"), "--done", str(out / f"shard_{k}.done.json"),
                           "--models_dir", str(tmp / "models"), "--batch_tokens", "30", "--flush_proteins", "1", "--layers", "[-1, 1]", "--pooling", "[mean, tokens]", "--yaml-in", str(tmp / "none.yaml")] + list(extra), capture_output=True, text=True, env=env)

def merged(world, out, K):
    tmp, env = world
    r = subprocess.run([sys.executable, f"{ROOT}/src/merge.py", "--sample", "s", "--shards", str(K), "--shard_root", str(out), "--out_dir", str(out / "final"), "--yaml-in", str(tmp / "none.yaml")], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    return out / "final" / "s.index.json"

def test_killed_shard_resumes_and_matches_uninterrupted_run(world):
    tmp, _ = world
    a, b = tmp / "a", tmp / "b"
    for k in (1, 2): assert shard(world, a, k, 2).returncode == 0
    r = shard(world, b, 1, 2, fail=2)
    assert r.returncode != 0 and "simulated kill" in r.stderr
    assert len(store.chunks(b / "shard_1")) == 2 and not (b / "shard_1.done.json").exists()
    r = shard(world, b, 1, 2)
    assert r.returncode == 0 and "resuming shard 1/2" in r.stdout, r.stdout + r.stderr
    assert shard(world, b, 2, 2).returncode == 0
    ia, ib = merged(world, a, 2), merged(world, b, 2)
    ids_a, ids_b = json.loads((ia.parent / "s.ids.json").read_text()), json.loads((ib.parent / "s.ids.json").read_text())
    assert ids_a == ids_b and sorted(ids_a) == sorted(f"p{i}" for i in range(23))
    for k in ("layer_3/mean", "layer_1/mean", "layer_3/tokens"):
        assert all(np.array_equal(x, y) for x, y in zip(store.read_rows(ia, list(range(23)), k), store.read_rows(ib, list(range(23)), k)))
    row = ids_a["p5"]; seq = FASTA.split("\n")[11]
    assert store.read_rows(ia, [row], "layer_3/tokens")[0].shape == (len(seq), 4)
    assert np.allclose(store.read_rows(ia, [row], "layer_3/mean")[0], store.read_rows(ia, [row], "layer_3/tokens")[0].mean(0))
    prov = json.loads((ia.parent / "s.json").read_text())
    assert prov["n"] == 23 and prov["layers"] == [1, 3] and prov["weights_sha256"] and not store.chunks(a / "shard_1")

def test_changed_settings_discard_old_chunks(world):
    tmp, _ = world
    out = tmp / "c"
    assert shard(world, out, 1, 1, fail=1).returncode != 0 and store.chunks(out / "shard_1")
    r = shard(world, out, 1, 1, extra=["--store_dtype", "float16"])
    assert r.returncode == 0 and "discarded" in r.stdout
    idx = merged(world, out, 1)
    assert json.loads(idx.read_text())["keys"]["layer_3/mean"]["dtype"] == "F16"

def test_tampered_weights_stop_the_job(world):
    tmp, _ = world
    (tmp / "models" / "fake_4" / "files" / "w.bin").write_bytes(b"other bytes")
    r = shard(world, tmp / "d", 1, 1)
    assert r.returncode != 0 and "weights check failed" in r.stdout + r.stderr

def test_non_finite_values_stop_the_shard(world):
    tmp, _ = world
    r = shard(world, tmp / "n", 1, 1, MP_FAKE_NAN="1")
    assert r.returncode != 0 and "non-finite" in r.stderr and not store.chunks(tmp / "n" / "shard_1")

def test_backend_shape_must_match_tools_tsv(world):
    tmp, _ = world
    r = shard(world, tmp / "s", 1, 1, MP_FAKE_DIM="5")
    assert r.returncode != 0 and "fix the pins table" in r.stdout + r.stderr

def test_merge_refuses_shards_with_different_compute_dtype(world):
    tmp, _ = world
    out = tmp / "m"
    for k in (1, 2): assert shard(world, out, k, 2).returncode == 0
    m = json.loads((out / "shard_2.done.json").read_text()); m["compute_dtype"] = "bf16"; (out / "shard_2.done.json").write_text(json.dumps(m))
    r = subprocess.run([sys.executable, f"{ROOT}/src/merge.py", "--sample", "s", "--shards", "2", "--shard_root", str(out), "--out_dir", str(out / "final"), "--yaml-in", str(tmp / "none.yaml")], capture_output=True, text=True, env=world[1])
    assert r.returncode != 0 and "different inputs or settings" in r.stdout + r.stderr

def test_writer_failure_does_not_hang(monkeypatch):
    import queue
    monkeypatch.setattr(sys, "argv", ["embed.py"])
    embed = load("embed")
    q, state = queue.Queue(1), dict(error=None)
    q.put("full")
    state["error"] = RuntimeError("writer died")
    assert embed.put(q, "next", state) is False

def test_bf16_detection_ignores_emulation(monkeypatch):
    import types
    seen = {}
    fake = types.SimpleNamespace(float32="f32", float16="f16", bfloat16="bf16", cuda=types.SimpleNamespace(is_bf16_supported=lambda including_emulation=True: seen.setdefault("emu", including_emulation) and False))
    monkeypatch.setitem(sys.modules, "torch", fake)
    import backends
    assert backends.torch_dtype("auto", types.SimpleNamespace(type="cuda")) == ("f32", "fp32") and seen["emu"] is False
