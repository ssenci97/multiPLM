"""Pins, weight checksums, batching, the cache key and the FASTA validation of prepare.py (host python, no torch)."""
import json, os, subprocess
import pytest
from conftest import ROOT
import backends

def test_every_pin_is_a_full_sha():
    rows = backends.registry()
    assert {"esmc_300m", "esmc_600m", "esmc_6b", "esm3_sm_open_v1", "esm2_t33_650M", "prott5_xl_half", "prostt5"} <= set(rows)
    import csv
    for r in csv.DictReader(open(backends.TOOLS_TSV), delimiter="\t"):
        c = r["commit"]
        assert (len(c) == 40 and int(c, 16) >= 0) or (c.startswith("sha256:") and len(c) == 71 and int(c[7:], 16) >= 0), c

def test_esm3_is_local_weights_only():
    r = backends.registry()["esm3_sm_open_v1"]
    assert r["repo"].startswith("https://huggingface.co/") and r["env"] == "esm-evo"
    assert not [n for n, x in backends.registry().items() if "forge" in x["repo"].lower() or "api" in x["family"].lower()]

def fake_model(tmp_path, spec, content=b"weights"):
    d = tmp_path / spec["name"]; (d / "files" / "data").mkdir(parents=True)
    (d / "files" / "data" / "w.pth").write_bytes(content); (d / "files" / ".cache").mkdir(); (d / "files" / ".cache" / "x").write_text("meta")
    backends.write_checksums(d, {str(p.relative_to(d)): backends.sha256(p) for p in backends.weight_files(d)})
    (d / "SOURCE.json").write_text(json.dumps(dict(revision=spec["commit"])))
    return d

def test_verify_accepts_matching_and_rejects_changed_weights(tmp_path):
    spec = backends.registry()["esmc_300m"]; d = fake_model(tmp_path, spec)
    rec, problems = backends.verify(d, spec)
    assert problems == [] and list(rec) == ["files/data/w.pth"]
    (d / "files" / "data" / "w.pth").write_bytes(b"tampered")
    assert any("differs" in p for p in backends.verify(d, spec)[1])
    (d / "files" / "data" / "extra.bin").write_bytes(b"x")
    assert any("not listed" in p for p in backends.verify(d, spec, full=False)[1])

def test_verify_rejects_another_revision_and_missing_checksums(tmp_path):
    spec = backends.registry()["esmc_300m"]; d = fake_model(tmp_path, spec)
    (d / "SOURCE.json").write_text(json.dumps(dict(revision="0" * 40)))
    assert "configs/tools.tsv pins" in backends.verify(d, spec)[1][0]
    (d / "CHECKSUMS").unlink()
    assert "CHECKSUMS missing" in backends.verify(d, spec)[1][0]

def test_batches_respect_the_padded_token_budget():
    lengths = [10, 10, 30, 30, 100, 500]
    groups = backends.batches(list(range(6)), lengths, 64, False)
    assert sum(groups, []) == list(range(6))
    assert all(len(g) * max(lengths[i] for i in g) <= 64 or len(g) == 1 for g in groups)
    assert backends.batches([0, 1, 2], lengths, 10000, True) == [[0], [1], [2]]

def test_layers_and_pooling_are_validated():
    reg = backends.registry()
    assert backends.resolve_layers(reg["esmc_300m"], [-1]) == [30] and backends.resolve_layers(reg["esmc_300m"], [1, -1, 30]) == [1, 30]
    assert backends.resolve_layers(reg["esm2_t6_8M"], "all") == [1, 2, 3, 4, 5, 6]
    for bad in ([0], [31], [-31]):
        with pytest.raises(ValueError, match="outside"): backends.resolve_layers(reg["esmc_300m"], bad)
    with pytest.raises(ValueError, match="only the last layer"): backends.resolve_layers(reg["esm3_sm_open_v1"], [10])
    assert backends.poolings(["mean", "tokens"], 5, 30) == ["mean", "tokens"] and backends.poolings({-1: ["tokens"]}, 30, 30) == ["tokens"] and backends.poolings({"5": "tokens"}, 5, 30) == ["tokens"]
    with pytest.raises(ValueError, match="pooling"): backends.poolings(["max"], 1, 30)

def test_shards_take_every_batch_once():
    groups = [[i] for i in range(10)]
    got = [backends.shard_batches(groups, k, 3) for k in (1, 2, 3)]
    assert sorted(sum(sum(got, []), [])) == list(range(10)) and [len(g) for g in got] == [4, 3, 3]

def test_prepare_validates_records(tmp_path):
    fa = tmp_path / "in.fasta"
    fa.write_text(">a desc\nMKV\nLL*\n>b\nMK1\n>a\nMKV\n>c\n\n>d\nmkvl\n")
    out = tmp_path / "o.fasta"
    r = subprocess.run(["python3", f"{ROOT}/src/prepare.py", "--sample", "s", "--source", str(fa), "--out_fasta", str(out), "--out_tsv", str(tmp_path / "o.tsv"), "--out_json", str(tmp_path / "o.json"), "--yaml-in", str(tmp_path / "none.yaml")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert out.read_text() == ">a desc\nMKVLL\n>d\nMKVL\n"
    s = json.loads((tmp_path / "o.json").read_text())
    assert s["kept"] == 2 and s["records"] == 5 and s["status"] == {"invalid:duplicate_id": 1, "invalid:empty": 1, "invalid:letters": 1, "kept": 2}

def test_all_layers_means_last_for_last_layer_only_backends():
    from backends import registry, resolve_layers
    reg = registry(f"{ROOT}/configs/tools.tsv")
    assert resolve_layers(reg["esm3_sm_open_v1"], "all") == [48] and resolve_layers(reg["esm2_t6_8M"], "all") == list(range(1, 7))
