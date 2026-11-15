"""Console: start gui_serve.py on a free port in a repo copy with fake SLURM commands and exercise every API endpoint."""
import json, os, socket, subprocess, sys, threading, time, urllib.error, urllib.parse, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
    assert [e["name"] for e in js("/api/examples")["examples"]] == ["example_chaperones", "example_kinase_gpcr", "example_toy"]
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

def fake_uniprot(repo, counts):
    """A local stand-in for rest.uniprot.org (one page per query, count chosen by Pfam id); the repo copy's prefetch.py is pointed at it."""
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)["query"][0]
            pf, n = next((k, v) for k, v in counts.items() if k in q)
            body = "".join(f">sp|Q{i:05d}|{pf}_{i} chaperone\nMKV{'AGLE' * 30}\n" for i in range(min(n, 500))).encode()
            self.send_response(200)
            for k, v in (("x-total-results", str(n)), ("x-uniprot-release", "2026_04"), ("x-uniprot-release-date", "04-November-2026"), ("Content-Length", str(len(body)))): self.send_header(k, v)
            self.end_headers(); self.wfile.write(body)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
    p = repo / "src" / "prefetch.py"; p.write_text(p.read_text().replace("https://rest.uniprot.org/uniprotkb/search", f"http://127.0.0.1:{srv.server_port}/search"))
    return srv

def test_uniprot_example_downloads_on_run_and_plans_every_step(server):
    base, repo, fb = server
    srv = fake_uniprot(repo, {"PF00011": 381, "PF00012": 319, "PF00183": 101})
    try:
        code, r = post(base, "/api/run", dict(analysis="ch", examples=["example_chaperones"]))
        assert code == 200 and r["members"] == ["ch_chaperones"], r
        a = repo / "results" / "ch_chaperones"
        assert sorted(p.name for p in (a / "inputs" / "uniprot").glob("*.fasta")) == ["HSP70.fasta", "HSP90.fasta", "sHSP.fasta"]
        assert (a / "inputs" / "uniprot" / "sHSP.fasta").read_text().count(">") == 381 and "prefetch.py" in (a / "run.log").read_text()
        st = json.loads(get(base, "/api/status?analysis=ch_chaperones")[1])
        assert {s["step"]: s["total"] for s in st["steps"]} == {"00_weights": 1, "01_prepare": 3, "02_dedup": 3, "03_embed": 3, "04_merge": 3, "05_summary": 1}
        assert st["size"]["proteins"] == {"sHSP": 381, "HSP70": 319, "HSP90": 101} and st["size"]["hard"] == []
        assert len((fb / "sbatch.log").read_text().splitlines()) == 14
    finally: srv.shutdown()

def test_uniprot_example_above_the_count_cap_saves_and_submits_nothing(server):
    base, repo, fb = server
    srv = fake_uniprot(repo, {"PF00011": 381, "PF00012": 501, "PF00183": 101})
    try:
        code, r = post(base, "/api/run", dict(analysis="ch", examples=["example_chaperones"]))
        assert code == 400 and "HSP70: 501 entries exceed the prefetch.max_entries of 500" in r["errors"][0] and "nothing was saved" in r["errors"][0], r
        assert not (repo / "data" / "uniprot" / "HSP70").exists() and not (repo / "results" / "ch_chaperones" / "inputs").exists() and not (fb / "sbatch.log").exists()
    finally: srv.shutdown()
