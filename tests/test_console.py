"""Console: start gui_serve.py on a free port in a repo copy with fake SLURM commands and exercise every API endpoint."""
import json, os, socket, subprocess, sys, time, urllib.error, urllib.request
import pytest
import yaml

@pytest.fixture
def server(repo, fakebin):
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    P = yaml.safe_load((repo / "configs" / "param_configs.yaml").read_text()); P["gui_serve"]["port"] = port
    (repo / "configs" / "param_configs.yaml").write_text(yaml.safe_dump(P, sort_keys=False))
    p = subprocess.Popen([sys.executable, str(repo / "src" / "gui_serve.py"), "--yaml-in"], cwd=repo, env=dict(os.environ), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try: urllib.request.urlopen(base + "/api/schema", timeout=1); break
        except OSError: time.sleep(0.1)
    yield base, repo, fakebin
    p.terminate(); p.wait(5)

def get(base, path):
    with urllib.request.urlopen(base + path, timeout=30) as r: return r.status, r.read()

def post(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r: return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e: return e.code, json.loads(e.read())

def test_every_endpoint(server):
    base, repo, fb = server
    for path in ("/", "/submit.css", "/submit.js"): assert get(base, path)[0] == 200
    js = lambda p: json.loads(get(base, p)[1])
    assert js("/api/schema")["sheet_columns"][:3] == ["sample_id", "fasta_path", "uniprot_query"]
    assert [e["name"] for e in js("/api/examples")["examples"]] == ["example_kinase_gpcr", "example_toy"]
    b = js("/api/backends"); assert b["selected"] == ["esmc_300m"] and all(x["weights"] == "missing" for x in b["backends"])
    assert {x["name"]: x["partition"] for x in b["backends"]}["esmc_6b"] == "DGX"
    assert js("/api/params")["embed"]["backends"] == ["esmc_300m"]
    assert js("/api/path?p=" + str(repo / "configs" / "examples" / "example_toy" / "globins.fasta"))["file"]
    assert js("/api/samples")["samples"] == []
    assert js("/api/status?analysis=")["outcome"] == "idle"
    assert js("/api/results?analysis=")["embeddings"] == []
    assert js("/api/perf")["cluster"]["available"] is False
    assert post(base, "/api/samples", dict(samples=[dict(sample_id="bad id")]))[0] == 400
    fa = str(repo / "configs" / "examples" / "example_toy" / "globins.fasta")
    assert post(base, "/api/samples", dict(samples=[dict(sample_id="g", fasta_path=fa)]))[0] == 200
    assert post(base, "/api/backends", dict(backends=["nope"]))[0] == 400
    assert post(base, "/api/backends", dict(backends=["esmc_300m", "prott5_xl_half"]))[1]["selected"] == ["esmc_300m", "prott5_xl_half"]
    code, r = post(base, "/api/run", dict(analysis="mine"))
    assert code == 200 and r["analysis"] == "mine", r
    st = js("/api/status?analysis=mine")
    assert st["running"] and [s["step"] for s in st["steps"]] == ["00_weights", "01_prepare", "02_dedup", "03_embed", "04_merge", "05_summary"] and st["steps"][3]["total"] == 2 and st["size"]["level"] == "ok"
    code, r = post(base, "/api/run", dict(analysis="ex", examples=["example_toy"]))
    assert code == 400 and "still running" in r["errors"][0]
    assert len(post(base, "/api/stop", dict(analysis="mine"))[1]["cancelled"]) == 8
    code, r = post(base, "/api/run", dict(analysis="ex", examples=["example_toy", "example_kinase_gpcr"]))
    assert code == 200 and r["members"] == ["ex_toy"] and "same weights" in r["errors"]["ex_kinase_gpcr"][0], r
    assert post(base, "/api/stop_all", {})[0] == 200
    assert js("/api/results?analysis=ex_toy")["backends"] == ["esmc_300m"] and js("/api/results?analysis=ex_kinase_gpcr")["backends"] == ["esm2_t6_8M", "esmc_300m"]
    assert "throughput" in js("/api/perf")
