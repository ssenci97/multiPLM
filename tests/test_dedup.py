"""dedup.py wiring with a fake mmseqs (identical sequences form a cluster): per-sample and joint scope, every protein traceable, version recorded."""
import json, subprocess, sys
import pytest
from conftest import ROOT

FAKE = '''#!/usr/bin/env python3
import sys
if sys.argv[1] == "version": print("18.8cc5c-fake"); sys.exit(0)
mode, inp, out = sys.argv[1:4]
lines = open(inp).read().split()
recs = [(lines[i][1:], lines[i + 1]) for i in range(0, len(lines), 2)]
rep = {}
with open(out + "_cluster.tsv", "w") as f:
    for h, s in recs: f.write(f"{rep.setdefault(s, h)}\\t{h}\\n")
open(out + "_rep_seq.fasta", "w").write("".join(f">{h}\\n{s}\\n" for s, h in rep.items()))
'''

@pytest.fixture
def setup(tmp_path):
    mm = tmp_path / "mmseqs"; mm.write_text(FAKE); mm.chmod(0o755)
    (tmp_path / "a.fasta").write_text(">a1 x\nMKV\n>a2\nMKV\n>a3\nGGG\n"); (tmp_path / "b.fasta").write_text(">b1\nGGG\n>b2\nWWW\n")
    return tmp_path, mm

def run(tmp, mm, scope):
    r = subprocess.run([sys.executable, f"{ROOT}/src/dedup.py", "--scope", scope, "--out_dir", str(tmp / scope), "--mmseqs", str(mm), "--fasta", str(tmp / "a.fasta"), str(tmp / "b.fasta"), "--yaml-in", str(tmp / "none.yaml")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return tmp / scope

def test_per_sample(setup):
    tmp, mm = setup; d = run(tmp, mm, "sample")
    assert (d / "a.fasta").read_text() == ">a1 x\nMKV\n>a3\nGGG\n" and (d / "b.fasta").read_text() == ">b1\nGGG\n>b2\nWWW\n"
    rows = (d / "a.clusters.tsv").read_text().splitlines()
    assert rows[0] == "representative\tmember_sample\tmember" and sorted(rows[1:]) == ["a1\ta\ta1", "a1\ta\ta2", "a3\ta\ta3"]
    j = json.loads((d / "a.dedup.json").read_text())
    assert j["proteins"] == 3 and j["representatives"] == 2 and j["mmseqs_version"] == "18.8cc5c-fake" and j["threads"] == 1

def test_joint_scope_maps_members_across_samples(setup):
    tmp, mm = setup; d = run(tmp, mm, "joint")
    assert (d / "b.fasta").read_text() == ">b2\nWWW\n"
    assert "a3\tb\tb1" in (d / "a.clusters.tsv").read_text()
    assert json.loads((d / "b.dedup.json").read_text())["represented_in_other_samples"] == 1
