#!/usr/bin/env python3
"""Serve the MultiPLM console on localhost (stdlib demo): Samples, Run and Results tabs plus a Performance page, with the logic in plain functions and all state in files."""
###################################### DEPENDENCIES_AND_CONSTANTS
from __future__ import annotations
import argparse
import csv
import datetime
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis
import backends as backends_mod
import pipeline
T0 = time.time()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAM_YAML, NAME = f"{ROOT}/configs/param_configs.yaml", "gui_serve"
GUI, SCHEMA = f"{ROOT}/gui", f"{ROOT}/configs/sample_schema.json"
SAMPLES_TSV, SAMPLES_DIR, EXAMPLES, RESULTS = f"{ROOT}/configs/samples.tsv", f"{ROOT}/configs/samples", f"{ROOT}/configs/examples", f"{ROOT}/results"
STATIC = {"/": ("submit.html", "text/html"), "/submit.css": ("submit.css", "text/css"), "/submit.js": ("submit.js", "application/javascript")}
DEFAULT_SIF, MP_BIN, PIPELINE = f"{ROOT}/containers/multiplm.sif", f"{ROOT}/.mp_bin", os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline.py")
PREFETCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prefetch.py")
FAILED = {"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "PREEMPTED"}
RUNNING = {"RUNNING", "CONFIGURING", "COMPLETING"}
FASTA_EXT = analysis.FASTA_EXT
PARAMS = dict(host="127.0.0.1", port=8000)
LAUNCH = threading.Lock()
LOGIN_SKIP = {"PWD", "OLDPWD", "SHLVL", "_", "PS1"}
NODE_FMT = "NodeList:40,Partition:24,StateCompact:12,CPUsState:24,Memory:14,AllocMem:14,Gres:60,GresUsed:80"
PERF_FIELDS = "JobID,State,Elapsed,TotalCPU,MaxRSS,AllocCPUS,ReqMem,Submit,Start,End,NodeList"
GPU_RE = re.compile(r"gpu(?::[^:(,]+)?:(\d+)")
###################################### HELPER_FUNCTIONS
def log(m: str) -> None:
    """Print a tagged log line."""
    print(f"[PYTHON-INFO] {m}", flush=True)
def read_yaml(p: str) -> dict:
    """YAML file as a dict ({} when missing or unreadable)."""
    try: return analysis.load_yaml(p)
    except Exception: return {}
def write_yaml(p: str, d: dict) -> None:
    """Write a YAML file atomically, keeping the previous one as .bak."""
    import yaml
    if os.path.exists(p): shutil.copy(p, f"{p}.bak")
    open(f"{p}.tmp", "w").write(yaml.safe_dump(d, sort_keys=False)); os.replace(f"{p}.tmp", p)
def yaml_io(p: dict, a: argparse.Namespace) -> dict:
    """Apply --yaml-in / --yaml-out to the server parameters."""
    c = read_yaml(PARAM_YAML)
    if a.yaml_in: p.update(c.get(NAME) or {}); log(f"params read from {PARAM_YAML} [{NAME}]")
    if a.yaml_out: c[NAME] = p; write_yaml(PARAM_YAML, c); log(f"params written to {PARAM_YAML} [{NAME}]")
    return p
def tsv_rows(p: str) -> list[dict]:
    """Rows of a TSV file ([] when missing)."""
    return list(csv.DictReader(open(p), delimiter="\t")) if os.path.isfile(p) else []
def tail(path: str, n: int = 40) -> str:
    """Last n lines of a text file."""
    if not os.path.exists(path): return ""
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END); fh.seek(max(0, fh.tell() - 65536))
        return "\n".join(fh.read().decode("utf-8", "replace").splitlines()[-n:])
def num(v: str, cast: type = float) -> object:
    """Cast or None."""
    try: return cast(v)
    except (TypeError, ValueError): return None
###################################### SAMPLES
def columns() -> list[str]:
    """Sheet columns of the schema."""
    return json.load(open(SCHEMA))["sheet_columns"]
def load_samples() -> list[dict]:
    """Saved configs/samples.tsv merged with configs/samples/<id>.json."""
    return [{**{k: v for k, v in analysis.submitted(ROOT, r["sample_id"]).items() if k not in r}, **r} for r in tsv_rows(SAMPLES_TSV) if r.get("sample_id")]
def save(samples: list[dict]) -> dict:
    """Write configs/samples.tsv and one json per sample."""
    cols = columns(); os.makedirs(SAMPLES_DIR, exist_ok=True)
    open(SAMPLES_TSV, "w").write("\t".join(cols) + "\n" + "".join("\t".join(str(s.get(c, "") or "").replace("\t", " ") for c in cols) + "\n" for s in samples))
    for s in samples: json.dump({**{k: v for k, v in s.items() if v not in ("", None)}, "submitted_at": f"{datetime.datetime.now():%F %T}"}, open(f"{SAMPLES_DIR}/{s['sample_id']}.json", "w"), indent=1)
    log(f"wrote {os.path.relpath(SAMPLES_TSV, ROOT)} ({len(samples)} samples)")
    return dict(samples_tsv=os.path.relpath(SAMPLES_TSV, ROOT), n=len(samples))
def examples() -> list[dict]:
    """configs/examples/* with their title, credit and samples."""
    out = []
    for n in sorted(os.listdir(EXAMPLES)) if os.path.isdir(EXAMPLES) else []:
        d = f"{EXAMPLES}/{n}"
        if not (os.path.exists(f"{d}/samples.tsv") and os.path.exists(f"{d}/param_configs.yaml")): continue
        m = read_yaml(f"{d}/meta.yaml")
        out.append(dict(name=n, short=n.removeprefix("example_"), title=m.get("title", n), credit=m.get("credit", ""), backends=(read_yaml(f"{d}/param_configs.yaml").get("embed") or {}).get("backends", []), options=options_of(read_yaml(f"{d}/param_configs.yaml")), samples=[{k: v for k, v in r.items() if v} for r in tsv_rows(f"{d}/samples.tsv")]))
    return out
def backends() -> dict:
    """Backends of configs/tools.tsv with their default GPU ask, partition and weights state; the ones selected in param_configs.yaml."""
    P = read_yaml(PARAM_YAML); S = {**pipeline.SLURM, **(P.get("slurm") or {})}
    out = []
    for name, r in backends_mod.registry(f"{ROOT}/configs/tools.tsv").items():
        src = f"{ROOT}/data/models/{name}/SOURCE.json"; got = json.load(open(src)) if os.path.isfile(src) else {}
        c, m, t = pipeline.GPU_RES.get(name, (2, 8000, 240))
        out.append(dict(name=name, family=r["family"], env=r["env"], repo=r["repo"], revision=r["commit"], size_gb=r["size_gb"], licence=r["licence"], cpus=c, mem_mb=m, minutes=t,
                        partition=S["dgx_partition"] if name in S["dgx_backends"] else S["gpu_partition"], weights="ready" if got.get("revision") == r["commit"] and os.path.isfile(f"{ROOT}/data/models/{name}/CHECKSUMS") else "missing"))
    return dict(backends=out, selected=(P.get("embed") or {}).get("backends", []), why=pipeline.GPU_WHY, gpus_max=S["gpus_max"])
def options_of(P: dict) -> dict:
    """The analysis switches of a parameter file: dedup, TWO-NN, PCA, t-SNE, UMAP and whether every layer is stored."""
    S, e2 = P.get("summarize") or {}, (P.get("summarize") or {}).get("embed2d") or {}
    return dict(dedup=bool((P.get("dedup") or {}).get("enabled")), twonn=bool((S.get("twonn") or {}).get("enabled")), pca=bool((S.get("pca") or {}).get("enabled")),
                tsne=bool(e2.get("enabled") and e2.get("tsne", True)), umap=bool(e2.get("enabled") and e2.get("umap", True)), all_layers=(P.get("embed") or {}).get("layers") == "all")
def set_options(o: dict) -> dict:
    """Store the analysis switches of the Samples tab in configs/param_configs.yaml (other settings stay)."""
    P = read_yaml(PARAM_YAML); S = P.setdefault("summarize", {})
    P.setdefault("dedup", {})["enabled"] = bool(o.get("dedup"))
    S.setdefault("twonn", {})["enabled"] = bool(o.get("twonn"))
    S.setdefault("pca", {})["enabled"] = bool(o.get("pca"))
    e2 = S.setdefault("embed2d", {}); e2["tsne"], e2["umap"] = bool(o.get("tsne")), bool(o.get("umap")); e2["enabled"] = e2["tsne"] or e2["umap"]
    P.setdefault("embed", {})["layers"] = "all" if o.get("all_layers") else [-1]
    write_yaml(PARAM_YAML, P)
    return options_of(P)
def set_backends(names: list[str]) -> dict:
    """Store the backends ticked in the Run tab under embed.backends of configs/param_configs.yaml."""
    known = backends_mod.registry(f"{ROOT}/configs/tools.tsv")
    bad = [n for n in names if n not in known]
    if bad or not names: return dict(errors=[f"unknown backend {b}" for b in bad] or ["tick at least one backend"])
    P = read_yaml(PARAM_YAML); P.setdefault("embed", {})["backends"] = list(dict.fromkeys(names)); write_yaml(PARAM_YAML, P)
    return dict(selected=P["embed"]["backends"])
def path_info(p: str) -> dict:
    """Existence, size and folder entries (folders and FASTA files) of a path, for autocomplete."""
    p = os.path.expanduser(p or "") or os.path.expanduser("~") + "/"
    out = dict(path=p, exists=os.path.exists(p), file=os.path.isfile(p), size=os.path.getsize(p) if os.path.isfile(p) else None, entries=[])
    d = p if p.endswith("/") else os.path.dirname(p)
    try: out["entries"] = [os.path.join(d, e.name) + ("/" if e.is_dir() else "") for e in sorted(os.scandir(d), key=lambda e: e.name)[:500] if not e.name.startswith(".") and (e.is_dir() or e.name.lower().endswith(FASTA_EXT))]
    except OSError: pass
    return out
###################################### RUN
def login_env(cmd: str) -> dict:
    """Environment of a login shell after cmd."""
    try: r = subprocess.run(["bash", "-lc", f"{cmd} >/dev/null 2>&1; env -0"], capture_output=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired): return {}
    return dict(kv.split("=", 1) for kv in r.stdout.decode("utf-8", "replace").split("\0") if "=" in kv)
def ensure_singularity() -> None:
    """Load singularity (module) into this process, or alias apptainer as singularity."""
    if shutil.which("singularity"): return
    log("loading singularity (module load singularity)")
    for k, v in login_env("module load singularity").items():
        if k not in LOGIN_SKIP and not k.startswith("BASH_FUNC_"): os.environ[k] = v
    if not shutil.which("singularity") and shutil.which("apptainer"):
        os.makedirs(MP_BIN, exist_ok=True)
        if not os.path.exists(f"{MP_BIN}/singularity"): os.symlink(shutil.which("apptainer"), f"{MP_BIN}/singularity")
        os.environ["PATH"] = f"{MP_BIN}:{os.environ.get('PATH', '')}"
def runtime() -> dict:
    """What a launch needs: the image, sbatch and singularity."""
    sif = os.environ.get("MP_SIF") or (DEFAULT_SIF if os.path.exists(DEFAULT_SIF) else None)
    return dict(sif=sif, sbatch=bool(shutil.which("sbatch")), apptainer=shutil.which("singularity"))
def analyses() -> list[dict]:
    """Every frozen analysis under results/."""
    out = []
    for n in sorted(os.listdir(RESULTS)) if os.path.isdir(RESULTS) else []:
        p = f"{RESULTS}/{n}/inputs/analysis.json"
        if not os.path.exists(p): continue
        try: d = json.load(open(p))
        except ValueError: continue
        st = read_state(n)
        out.append(dict(name=n, created_at=d.get("created_at"), samples=d.get("samples", []), started_at=st.get("started_at"), done=os.path.exists(f"{RESULTS}/{n}/summary.tsv"), group=st.get("group")))
    return out
def read_state(name: str) -> dict:
    """results/<name>/run_state.json."""
    p = f"{RESULTS}/{name}/run_state.json"
    return json.load(open(p)) if os.path.exists(p) else {}
def job_states(name: str) -> tuple[list[dict], dict]:
    """jobs.tsv rows and their SLURM states."""
    rows = pipeline.read_jobs(name, ROOT)
    return rows, pipeline.slurm_states([r["jobid"] for r in rows])
def live_rows(name: str) -> list[dict]:
    """Rows of jobs still queued or running."""
    rows, st = job_states(name)
    return [r for r in rows if st.get(r["jobid"]) in pipeline.LIVE]
def preflight(need_sheet: bool) -> str | None:
    """Why a launch cannot start now, or None."""
    for a in analyses():
        if live_rows(a["name"]): return f"analysis {a['name']} is still running"
    if need_sheet and not tsv_rows(SAMPLES_TSV): return "no samples in configs/samples.tsv: save samples first"
    rt = runtime()
    if not rt["sbatch"]: return "sbatch not found: start the console on the cluster login node"
    if not rt["apptainer"]: return "singularity not on PATH (module load singularity before starting the console)"
    return None
def last_section(name: str) -> str:
    """The last launch section of results/<name>/run.log."""
    p = f"{RESULTS}/{name}/run.log"; t = open(p, errors="replace").read() if os.path.exists(p) else ""
    return t[t.rfind("\n===== ") + 1:] if "\n===== " in t else t
def prefetch(name: str, base: str) -> list[str]:
    """Download the missing UniProt samples of a new analysis; return the !! errors."""
    a_dir = analysis.analysis_dir(name, ROOT)
    if os.path.exists(f"{a_dir}/inputs/samples.tsv") or not any((r.get("uniprot_query") or "").strip() for r in tsv_rows(f"{base}/samples.tsv")): return []
    cmd = ["python3", PREFETCH, "--samples", f"{base}/samples.tsv", "--yaml-in", f"{base}/param_configs.yaml"]
    os.makedirs(a_dir, exist_ok=True)
    with open(f"{a_dir}/run.log", "a") as fh:
        fh.write(f"\n===== {datetime.datetime.now():%F %T} {' '.join(cmd)} =====\n"); fh.flush()
        try: rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, timeout=600).returncode
        except subprocess.TimeoutExpired: rc = -1; fh.write("[PYTHON-INFO] !! prefetch.py timed out\n")
    if not rc: return []
    return [l.split("!!", 1)[1].strip() for l in last_section(name).splitlines() if "!!" in l] or [f"prefetch.py exited with code {rc}"]
def launch(name: str, src: str | None = None, group: str | None = None, online_here: bool = False, allow_large: bool = False) -> tuple[dict | None, object]:
    """Prefetch, freeze and submit one analysis; errors come from the !! lines of run.log."""
    try: errs = prefetch(name, src or f"{ROOT}/configs")
    except ValueError as e: return None, str(e)
    if errs: return None, errs
    try: a_dir, new, errs = analysis.create(name, ROOT, src)
    except ValueError as e: return None, str(e)
    if errs: return None, errs
    cmd = ["python3", PIPELINE, name, "--root", ROOT] + (["--online-here"] if online_here else []) + (["--allow-large"] if allow_large else [])
    with open(f"{a_dir}/run.log", "a") as fh:
        fh.write(f"\n===== {datetime.datetime.now():%F %T} {' '.join(cmd)} =====\n"); fh.flush()
        try: rc = subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, MP_ROOT=ROOT), stdout=fh, stderr=subprocess.STDOUT, timeout=3600 if online_here else 600).returncode
        except subprocess.TimeoutExpired: rc = -1; fh.write("[PYTHON-INFO] !! pipeline.py timed out\n")
    sec = last_section(name)
    if rc: return None, [l.split("!!", 1)[1].strip() for l in sec.splitlines() if "!!" in l] or [l for l in sec.splitlines() if l.strip()][-5:]
    rows = pipeline.read_jobs(name, ROOT)
    st = dict(analysis=name, started_at=f"{datetime.datetime.now():%F %T}", new=new, samples=json.load(open(f"{a_dir}/inputs/analysis.json"))["samples"], submitted=sum(bool(r["jobid"]) for r in rows), image=runtime()["sif"])
    if src: st["source"] = os.path.relpath(src, ROOT)
    if group: st["group"] = group
    json.dump(st, open(f"{a_dir}/run_state.json", "w"), indent=1)
    log(f"{'started' if new else 'resumed'} analysis {name}, {st['submitted']} job(s) submitted")
    return st, None
def start_run(name: str, wanted: list[str], online_here: bool = False, allow_large: bool = False) -> tuple[dict | None, object]:
    """One launch at a time: the saved samples, or one analysis <name>_<short> per ticked example."""
    if not LAUNCH.acquire(blocking=False): return None, "a launch is already in progress: wait for it to finish"
    try:
        if not wanted:
            err = preflight(True)
            return (None, err) if err else launch(name, online_here=online_here, allow_large=allow_large)
        known = {e["name"]: e for e in examples()}
        if [w for w in wanted if w not in known]: return None, [f"unknown example {w}" for w in wanted if w not in known]
        if not analysis.NAME_RE.match(name or ""): return None, f"analysis name {name!r}: use letters, digits, _ . - (max 64)"
        err = preflight(False)
        if err: return None, err
        started, errors = [], {}
        for w in wanted:
            st, e = launch(f"{name}_{known[w]['short']}", f"{EXAMPLES}/{w}", name, online_here, allow_large)
            if st: started.append(st)
            else: errors[f"{name}_{known[w]['short']}"] = e if isinstance(e, list) else [e]
        if not started: return None, [f"{n}: {m}" for n, ms in errors.items() for m in ms]
        return dict(analysis=started[0]["analysis"], group=name, members=[s["analysis"] for s in started], errors=errors), None
    finally: LAUNCH.release()
def stop_run(name: str) -> dict:
    """Cancel the jobs of one analysis."""
    try: r = pipeline.stop(name, ROOT)
    except ValueError: return dict(stopped=False, cancelled=[], remaining=[])
    return dict(r, stopped=bool(r["cancelled"]))
def job_state(job: dict, by: dict, st: dict, jobid: str) -> str:
    """done > waiting (PENDING) > running > failed > cancelled > stopped (log, no live job) > waiting."""
    if pipeline.is_done(job, by): return "done"
    s = st.get(jobid, "") if jobid else ""
    if s == "PENDING": return "waiting"
    if s in RUNNING: return "running"
    if s in FAILED: return "failed"
    if s.startswith("CANCELLED"): return "cancelled"
    return "stopped" if os.path.exists(job["log"]) else "waiting"
def step_state(name: str, rows: list[dict], st: dict) -> list[dict]:
    """Per step: counts and per-job state, with the log tail of failed or stopped jobs."""
    try: jobs = pipeline.plan(name, ROOT)
    except (ValueError, OSError, KeyError, RuntimeError): return []
    by, ids, out = {j["key"]: j for j in jobs}, {r["key"]: r["jobid"] for r in rows}, []
    for step in pipeline.STEPS:
        L = []
        for j in (j for j in jobs if j["step"] == step):
            s = job_state(j, by, st, ids.get(j["key"]))
            L.append(dict(job=j["label"], gpu=j["gpu"], state=s, log=os.path.relpath(j["log"], ROOT), tail=tail(j["log"], 15) if s in ("failed", "stopped", "cancelled") else ""))
        c = lambda k: sum(r["state"] == k for r in L)
        out.append(dict(step=step, total=len(L), done=c("done"), failed=c("failed"), cancelled=c("cancelled"), running=c("running"), jobs=L))
    return out
def status(name: str) -> dict:
    """Everything the Run tab shows for one analysis."""
    rows, sl = job_states(name) if name else ([], {})
    live = [r for r in rows if sl.get(r["jobid"]) in pipeline.LIVE]
    steps = step_state(name, rows, sl) if name else []
    total, done = sum(s["total"] for s in steps), sum(s["done"] for s in steps)
    states = {j["state"] for s in steps for j in s["jobs"]}
    outcome = "running" if live else "done" if total and done == total else "failed" if states & {"failed", "cancelled"} else "stopped" if rows else "idle"
    q = [dict(zip(("job", "name", "state", "time", "partition", "where"), l.split("|", 5))) for l in pipeline.sh(["squeue", "-h", "-j", ",".join(r["jobid"] for r in live), "-o", "%i|%j|%T|%M|%P|%R"]) if l.count("|") >= 5] if live else []
    try: size = pipeline.estimate(name, ROOT) if name else None
    except (OSError, ValueError, KeyError): size = None
    return dict(runtime=runtime(), analyses=analyses(), analysis=name, state=read_state(name) if name else {}, running=bool(live), size=size, outcome=outcome if name else "idle", progress=dict(done=done, total=total, pct=round(100 * done / total, 1)) if total else None,
                queue=q, steps=steps, node_log=tail(f"{RESULTS}/{name}/logs/node_pick.log", 20) if name else "", log=tail(f"{RESULTS}/{name}/run.log") if name else "")
###################################### RESULTS
def results(name: str) -> dict:
    """Sample x backend table, per-sample counts, cheap aggregates, provenance and weights of one analysis (never raw vectors)."""
    out = dict(analysis=name, embeddings=[], samples=[], aggregates=None, meta={}, weights={}, backends=[], sample_ids=[])
    if not name: return out
    a = f"{RESULTS}/{name}"
    out["embeddings"], out["samples"] = tsv_rows(f"{a}/summary.tsv"), tsv_rows(f"{a}/samples_summary.tsv")
    if os.path.isfile(f"{a}/aggregates.json"): out["aggregates"] = json.load(open(f"{a}/aggregates.json"))
    P = read_yaml(f"{a}/inputs/params.yaml"); out["backends"] = (P.get("embed") or {}).get("backends", []); out["settings"] = {k: (P.get("embed") or {}).get(k) for k in ("layers", "pooling", "store_dtype")}
    out["sample_ids"] = [r["sample_id"] for r in tsv_rows(f"{a}/inputs/samples.tsv")]
    for f in sorted(os.listdir(f"{a}/meta")) if os.path.isdir(f"{a}/meta") else []:
        if f.endswith(".json"):
            try: out["meta"][f[:-5]] = json.load(open(f"{a}/meta/{f}"))
            except ValueError: pass
    for b in out["backends"]:
        p = f"{ROOT}/data/models/{b}/SOURCE.json"
        out["weights"][b] = json.load(open(p)) if os.path.isfile(p) else None
    return out
###################################### PERFORMANCE
def gpus(field: str) -> int:
    """GPU count of a sinfo Gres/GresUsed field."""
    return sum(int(m) for m in GPU_RE.findall(field or ""))
def ours_running() -> list[dict]:
    """Our RUNNING jobs from squeue with cores, memory and GPU flag."""
    rows = {r["jobid"]: dict(r, analysis=a["name"]) for a in analyses() for r in pipeline.read_jobs(a["name"], ROOT) if r.get("jobid")}
    ids, out = list(rows), []
    for i in range(0, len(ids), 200):
        for l in pipeline.sh(["squeue", "-h", "-j", ",".join(ids[i:i + 200]), "-o", "%i|%T|%P|%N|%C|%m"]):
            f = [x.strip() for x in l.split("|")]
            if len(f) < 6 or f[1] != "RUNNING" or f[0] not in rows: continue
            out.append(dict(jobid=f[0], analysis=rows[f[0]]["analysis"], key=rows[f[0]]["key"], partition=f[2].rstrip("*"), node=f[3], cpus=num(f[4], int) or 0, mem_mb=round(pipeline.mb(f[5])), gpus=1 if rows[f[0]].get("gpu") == "1" else 0))
    return out
def cluster() -> dict:
    """Nodes and partitions from sinfo (counts only, GPUs included) with the share our running jobs hold."""
    nodes = {}
    for l in pipeline.sh(["sinfo", "-N", "-h", "-O", NODE_FMT]):
        f = l.split()
        if len(f) < 6 or f[3].count("/") != 3: continue
        a, i, o, t = (num(x, int) or 0 for x in f[3].split("/"))
        n = nodes.setdefault(f[0], dict(node=f[0], partitions=[], state=f[2].rstrip("*~#!%$@^-"), cpus=t, alloc=a, idle=i, mem_mb=num(f[4], int) or 0, alloc_mb=int(num(f[5]) or 0), gpus=gpus(f[6] if len(f) > 6 else ""), gpus_used=gpus(f[7] if len(f) > 7 else "")))
        if f[1].rstrip("*") not in n["partitions"]: n["partitions"].append(f[1].rstrip("*"))
    mine = ours_running()
    for n in nodes.values():
        here = [j for j in mine if j["node"] == n["node"]]
        n.update(ours_cpus=sum(j["cpus"] for j in here), ours_mb=sum(j["mem_mb"] for j in here), ours_gpus=sum(j["gpus"] for j in here), ours_jobs=[f"{j['analysis']}:{j['key']}" for j in here])
    parts = {}
    for n in nodes.values():
        for p in n["partitions"]:
            d = parts.setdefault(p, dict(partition=p, nodes=0, cpus=0, alloc=0, mem_mb=0, alloc_mb=0, gpus=0, gpus_used=0, ours_cpus=0, ours_gpus=0, ours_jobs=0))
            d["nodes"] += 1
            for k in ("cpus", "alloc", "mem_mb", "alloc_mb", "gpus", "gpus_used"): d[k] += n[k]
    for j in mine:
        d = parts.get(j["partition"])
        if d: d["ours_cpus"] += j["cpus"]; d["ours_gpus"] += j["gpus"]; d["ours_jobs"] += 1
    return dict(nodes=sorted(nodes.values(), key=lambda n: (n["partitions"][0], n["node"])), partitions=sorted(parts.values(), key=lambda d: d["partition"]), available=bool(nodes),
                ours=dict(jobs=len(mine), cpus=sum(j["cpus"] for j in mine), gpus=sum(j["gpus"] for j in mine), nodes=sorted({j["node"] for j in mine})))
def job_perf() -> dict:
    """sacct view of every job in every jobs.tsv: efficiency, waits, OOM/timeouts, core-hours and GPU-hours, per job, backend and analysis."""
    rows = [dict(r, analysis=a["name"]) for a in analyses() for r in pipeline.read_jobs(a["name"], ROOT) if r.get("jobid")]
    acct, ids = {}, [r["jobid"] for r in rows]
    for i in range(0, len(ids), 200):
        for l in pipeline.sh(["sacct", "-n", "-P", "-j", ",".join(ids[i:i + 200]), "-o", PERF_FIELDS]):
            f = l.split("|")
            if len(f) < 11: continue
            d = acct.setdefault(f[0].split(".")[0], dict(rss=0.0, cpu=0.0, elapsed=0.0))
            if "." not in f[0]: d.update(state=f[1].split()[0] if f[1].strip() else "", cpus=num(f[5], int), req=pipeline.mb(f[6]), submit=num(f[7], datetime.datetime.fromisoformat), start=num(f[8], datetime.datetime.fromisoformat), end=f[9], node=f[10])
            d.update(rss=max(d["rss"], pipeline.mb(f[4])), cpu=max(d["cpu"], pipeline.to_min(f[3])), elapsed=max(d["elapsed"], pipeline.to_min(f[2])))
    jobs = []
    for r in rows:
        a = acct.get(r["jobid"])
        if not a or "state" not in a: continue
        cpus, req, el, rq = a["cpus"] or num(r.get("cpus"), int) or 1, num(r.get("mem_mb")) or a["req"] or 0, a["elapsed"], num(r.get("minutes"))
        jobs.append(dict(analysis=r["analysis"], key=r["key"], tool=pipeline.group(r["key"]), jobid=r["jobid"], state=a["state"], node=a["node"] if a["node"] not in ("None assigned", "") else "", gpu=r.get("gpu") == "1",
                         cpus=cpus, req_mb=round(req), used_mb=round(a["rss"]), mem_eff=round(100 * a["rss"] / req, 1) if req and a["rss"] else None, elapsed_min=round(el, 1), req_min=rq,
                         time_eff=round(100 * el / rq, 1) if rq and el else None, cpu_eff=round(100 * a["cpu"] / (el * cpus), 1) if el and a["cpu"] and a["state"] not in pipeline.LIVE else None,
                         wait_min=round((a["start"] - a["submit"]).total_seconds() / 60, 1) if a["start"] and a["submit"] else None, core_h=round(el * cpus / 60, 2), gpu_h=round(el / 60, 2) if r.get("gpu") == "1" else 0))
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else None
    def agg(key: str) -> list[dict]:
        """Aggregate the jobs by one field."""
        g = {}
        for j in jobs: g.setdefault(j[key], []).append(j)
        return [dict(name=k, n=len(v), completed=sum(j["state"] == "COMPLETED" for j in v), failed=sum(j["state"] in FAILED - {"OUT_OF_MEMORY", "TIMEOUT"} for j in v), oom=sum(j["state"] == "OUT_OF_MEMORY" for j in v),
                     timeout=sum(j["state"] == "TIMEOUT" for j in v), cpu_eff=med([j["cpu_eff"] for j in v if j["cpu_eff"] is not None]), mem_eff=med([j["mem_eff"] for j in v if j["mem_eff"] is not None]),
                     wait_min=med([j["wait_min"] for j in v if j["wait_min"] is not None]), core_h=round(sum(j["core_h"] for j in v), 2), gpu_h=round(sum(j["gpu_h"] for j in v), 2)) for k, v in sorted(g.items())]
    return dict(jobs=jobs, tools=agg("tool"), analyses=agg("analysis"))
def perf() -> dict:
    """Performance page data (refreshes the sacct part of the perf metadata first)."""
    for a in analyses():
        try: pipeline.fill_perf(a["name"], ROOT)
        except (OSError, ValueError) as e: log(f"!! perf metadata of {a['name']} not refreshed: {e}")
    thr = {}
    for a in analyses():
        p = f"{RESULTS}/{a['name']}/aggregates.json"
        for b, t in (json.load(open(p)).get("throughput") or {}).items() if os.path.isfile(p) else []:
            d = thr.setdefault(b, dict(proteins=0, gpu_s=0.0, shards=0))
            for k in d: d[k] += t[k]
    for d in thr.values(): d["proteins_per_gpu_hour"] = round(d["proteins"] / (d["gpu_s"] / 3600), 1) if d["gpu_s"] else None
    return dict(at=f"{datetime.datetime.now():%F %T}", cluster=cluster(), runs=job_perf(), throughput=thr)
###################################### SERVER
class Handler(BaseHTTPRequestHandler):
    """Thin router: static files and the JSON API."""
    def log_message(self, fmt: str, *a: object) -> None:
        """Tagged access log."""
        log(f"{self.address_string()} {fmt % a}")
    def send_json(self, code: int, obj: object) -> None:
        """Send a JSON response."""
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def query(self, k: str) -> str:
        """One query-string value."""
        return (parse_qs(urlparse(self.path).query).get(k) or [""])[0]
    def do_GET(self) -> None:
        """GET routes."""
        path = urlparse(self.path).path
        if path in STATIC:
            name, ctype = STATIC[path]; b = open(f"{GUI}/{name}", "rb").read()
            self.send_response(200); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b); return
        routes = {"/api/schema": lambda: json.load(open(SCHEMA)), "/api/samples": lambda: dict(samples=load_samples()), "/api/examples": lambda: dict(examples=examples()), "/api/backends": backends, "/api/options": lambda: options_of(read_yaml(PARAM_YAML)),
                  "/api/params": lambda: read_yaml(PARAM_YAML), "/api/path": lambda: path_info(self.query("p")), "/api/status": lambda: status(self.query("analysis")),
                  "/api/results": lambda: results(self.query("analysis")), "/api/perf": perf}
        self.send_json(200, routes[path]()) if path in routes else self.send_json(404, dict(error="not found"))
    def do_POST(self) -> None:
        """POST routes."""
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path == "/api/samples":
            errs = analysis.validate(body.get("samples", []))
            return self.send_json(400, dict(errors=errs)) if errs else self.send_json(200, save(body["samples"]))
        if self.path == "/api/options": return self.send_json(200, set_options(body))
        if self.path == "/api/backends":
            r = set_backends([str(x) for x in body.get("backends") or []])
            return self.send_json(400 if "errors" in r else 200, r)
        if self.path == "/api/run":
            st, err = start_run((body.get("analysis") or "").strip(), [str(x) for x in body.get("examples") or []], bool(body.get("online_here")), bool(body.get("allow_large")))
            return self.send_json(400, dict(errors=err if isinstance(err, list) else [err])) if err else self.send_json(200, st)
        if self.path == "/api/stop": return self.send_json(200, stop_run((body.get("analysis") or "").strip()))
        if self.path == "/api/stop_all": return self.send_json(200, dict(analyses={a["name"]: stop_run(a["name"]) for a in analyses()}))
        self.send_json(404, dict(error="not found"))
###################################### MAIN
def main() -> None:
    """Start the console on localhost."""
    ap = argparse.ArgumentParser(description="MultiPLM console"); ap.add_argument("--yaml-in", action="store_true"); ap.add_argument("--yaml-out", action="store_true")
    p = yaml_io(dict(PARAMS), ap.parse_args())
    if os.environ.get("APPTAINER_CONTAINER") or os.environ.get("SINGULARITY_CONTAINER"): log("!! running inside the container: sbatch will not be reachable; start the console on the host")
    ensure_singularity()
    srv = ThreadingHTTPServer((p["host"], int(p["port"])), Handler)
    rt = runtime()
    log(f"image {rt['sif'] or 'not pulled'}, sbatch {'yes' if rt['sbatch'] else 'no'}, singularity {rt['apptainer'] or 'no'}; console at http://{p['host']}:{p['port']}/ (ctrl-c to stop)")
    try: srv.serve_forever()
    except KeyboardInterrupt: log("stopped")
    finally: srv.server_close()
    log(f"{datetime.datetime.now():%F %T} | elapsed {time.time() - T0:.1f} s | max RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576:.2f} GB")
if __name__ == "__main__":
    main()
