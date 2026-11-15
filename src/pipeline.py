#!/usr/bin/env python3
"""Plan, size, schedule (CPU lane + GPU lane) and submit the MultiPLM jobs of an analysis on SLURM; files on disk decide what runs (ported from KNOCKIN-VAL 2263299)."""
###################################### DEPENDENCIES_AND_CONSTANTS
from __future__ import annotations
import argparse
import collections
import csv
import datetime
import fcntl
import getpass
import glob
import json
import os
import re
import resource
import subprocess
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis
import backends as backends_mod
import sample_meta
T0 = time.time()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEPS = ["00_weights", "01_prepare", "02_dedup", "03_embed", "04_merge", "05_summary"]
LIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED"}
FOOTER = re.compile(r"\| elapsed [0-9]")
PEAK_RE = re.compile(r"\| elapsed ([0-9:]+) \| max RSS ([0-9.]+) GB")
COLS = ["key", "step", "label", "jobid", "log", "node", "partition", "gpu", "cpus", "mem_mb", "minutes", "in_bytes"]
RES = dict(weights=(1, 1000, 120), prepare=(1, 1000, 30), dedup=(1, 1000, 60), merge=(1, 1000, 60), summary=(1, 1000, 120))
EST = dict(weights=20, prepare=3, dedup=5, merge=5, summary=10, embed=30)
DEDUP = dict(enabled=False, scope="sample")
EMBED = dict(backends=[], layers=[-1], pooling=["mean"], store_dtype="float32", max_length=0, shard_proteins=25000, size_warn_gb=50, size_cap_gb=500)
LENGTHS = {}
TWONN_RUNTIME_MB = 5000
GPU_RES = {
    "esm2_t6_8M": (2, 4000, 60), "esm2_t12_35M": (2, 4000, 60), "esm2_t30_150M": (2, 4000, 120), "esm2_t33_650M": (2, 8000, 240), "esm2_t36_3B": (2, 16000, 480),
    "esm3_sm_open_v1": (2, 8000, 480), "esmc_300m": (2, 6000, 120), "esmc_600m": (2, 8000, 240), "esmc_6b": (2, 16000, 720),
    "prott5_xl_half": (2, 8000, 240), "prostt5": (2, 16000, 240),
}
GPU_WHY = "2 cores: reader and writer threads beside the GPU loop; memory: CUDA/torch runtime (~3 GB) + the checkpoint held in host RAM while it loads (6B: one 5 GB shard at a time)"
SLURM = dict(partition="GENOA", gpu_partition="DGX", dgx_partition="DGX", dgx_backends=["esmc_6b"], gpu_flag="--gres=gpu:A100:1", gpus_max=1, node_fraction=0.5)
BUDGET_CPUS, BUDGET_MEM_MB = 32, 64000
MEM_MIN, MEM_MAX, GPU_MEM_MAX, TIME_MIN, TIME_MAX, SAFETY, CPU_SAFETY, CPU_MAX, CPU_HOT = 1000, 32000, 64000, 15, 2880, 1.25, 1.25, 8, 0.9
MAX_JOBS, MAX_USER_JOBS, STOP_WAIT, STOP_PAUSE = 100, 200, 120, 2
SKIP_PARTS, USABLE = re.compile(r"^(DGX|GPU|H100|UPDATE)", re.I), ("idle", "mix", "mixed")
NODE_FMT = "NodeList:40,Partition:24,StateCompact:12,CPUsState:24,Memory:14,AllocMem:14"
ACCT = "JobID,State,TotalCPU,Elapsed,MaxRSS,AllocCPUS,ReqMem,Submit,Start"
###################################### HELPER_FUNCTIONS
def log(m: str) -> None:
    """Print a tagged log line."""
    print(f"[PYTHON-INFO] {m}", flush=True)
def size(p: str) -> int:
    """File size, 0 when missing."""
    return os.path.getsize(p) if os.path.isfile(p) else 0
def sh(cmd: list[str], timeout: int = 30) -> list[str]:
    """Run a command and return its stdout lines ([] when it fails)."""
    try: return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.splitlines()
    except (OSError, subprocess.SubprocessError): return []
def image_ref(root: str) -> str:
    """The pushed image digest of containers/images.env."""
    return open(f"{root}/containers/images.env").read().split("MP_IMAGE_REF=")[1].split()[0]
def image(root: str) -> str:
    """The .sif the jobs run in (MP_SIF, the pulled sif, else the registry digest)."""
    sif = f"{root}/containers/multiplm.sif"
    return os.environ.get("MP_SIF") or (sif if os.path.exists(sif) else f"docker://{image_ref(root)}")
def kind(key: str) -> str:
    """Job kind of a key (weights, prepare, embed, summary)."""
    return key.split(".")[0]
def group(key: str) -> str:
    """Sizing group: embed jobs are sized per backend, the others per kind."""
    return ".".join(key.split(".")[:2]) if kind(key) == "embed" else kind(key)
def meta_target(key: str) -> tuple[str, str]:
    """(metadata document, step) a job writes its perf record to."""
    p = key.split(".")
    return {"prepare": (p[-1], "01_prepare"), "dedup": (p[-1], "02_dedup") if len(p) > 1 else ("_analysis", "02_dedup"), "embed": (p[2], f"03_embed_{p[1]}_shard{p[3]}") if len(p) > 3 else ("_analysis", key), "merge": (p[-1], f"04_merge_{p[1]}") if len(p) > 2 else ("_analysis", key), "weights": ("_analysis", f"00_weights_{p[-1]}")}.get(p[0], ("_analysis", "05_summary"))
def outside(path: str, root: str) -> list[str]:
    """Folders of a path (and of its symlink target) that lie outside the repo, to bind into the container."""
    r = os.path.realpath(root)
    return [d for d in dict.fromkeys([os.path.dirname(path), os.path.dirname(os.path.realpath(path))]) if not (d == r or d.startswith(r + "/"))]
def slurm_cfg(P: dict) -> dict:
    """SLURM settings of the frozen params over the defaults; more than 2 GPUs is refused."""
    s = {**SLURM, **(P.get("slurm") or {})}
    if int(s["gpus_max"]) not in (1, 2): raise ValueError(f"slurm.gpus_max={s['gpus_max']}: the GPU rule allows 1 (default) or, exceptionally, 2")
    if float(s["node_fraction"]) > 0.5: raise ValueError(f"slurm.node_fraction={s['node_fraction']}: above 0.5 needs the user's explicit approval; edit SLURM in src/pipeline.py consciously")
    return s
###################################### PLAN
def lengths_of(path: str) -> list[int]:
    """Residue count of every record of a FASTA (cached by path, size and mtime, so status polling stays cheap)."""
    st = os.stat(path); k = (path, st.st_size, st.st_mtime)
    if k not in LENGTHS:
        out, n = [], None
        for line in open(path):
            if line.startswith(">"):
                if n is not None: out.append(n)
                n = 0
            elif n is not None: n += len(line.strip())
        LENGTHS[k] = out + ([n] if n is not None else [])
    return LENGTHS[k]
def plan(name: str, root: str = ROOT) -> list[dict]:
    """List every job of the analysis with its inputs, outputs, log, dependencies and default size (weights, prepare, embed shards, merge, summary)."""
    analysis.analysis_dir(name, root)
    a, inp = f"results/{name}", f"results/{name}/inputs"
    if not os.path.exists(f"{root}/{inp}/samples.tsv"): raise ValueError(f"{inp}/samples.tsv missing: create the analysis first with python3 src/analysis.py {name}")
    P = analysis.load_yaml(f"{root}/{inp}/params.yaml"); S, E, D = slurm_cfg(P), {**EMBED, **(P.get("embed") or {})}, {**DEDUP, **(P.get("dedup") or {})}
    frozen = lambda sid: json.load(open(f"{root}/{inp}/samples/{sid}.json")) if os.path.exists(f"{root}/{inp}/samples/{sid}.json") else {}
    samples = {r["sample_id"]: {**frozen(r["sample_id"]), **{k: v for k, v in r.items() if v}} for r in analysis.read_sheet(f"{root}/{inp}/samples.tsv")}
    tools = {r["name"]: r for r in csv.DictReader(open(f"{root}/{inp}/tools.tsv"), delimiter="\t") if r["kind"] == "weights"}
    backends = list(dict.fromkeys(E["backends"] or []))
    errs = analysis.validate(list(samples.values())) + [f"unknown backend {b}" for b in backends if b not in tools]
    if errs: raise ValueError("; ".join(errs))
    ab, jobs = (lambda p: p if os.path.isabs(p) else f"{root}/{p}"), []
    def add(key: str, step: str, label: str, script: str, args: list, outputs: list[str], logp: str, deps: tuple = (), shared: bool = False, binds: tuple = (), inputs: tuple = (), gpu: bool = False, online: bool = False, res: tuple = None, part: str = "", revision: str = "", why: str = "") -> None:
        """Append one job with its default size."""
        c, m, t = res or RES[kind(key)]
        jobs.append(dict(key=key, step=step, label=label, script=f"src/{script}", args=[str(x) for x in args], outputs=[ab(o) for o in outputs], log=ab(logp), deps=list(dict.fromkeys(deps)), shared=shared,
                         binds=list(binds), inputs=[ab(x) for x in inputs], gpu=gpu, online=online, cpus=c, mem_mb=m, minutes=t, partition=part, revision=revision, why=why))
    for b in backends:
        add(f"weights.{b}", STEPS[0], f"weights {b}", "00_weights.sh", [b], [f"data/models/{b}/CHECKSUMS"], f"{a}/logs/00_weights/{b}.log", shared=True, online=True, revision=tools[b]["commit"])
    src = {s: (r.get("fasta_path") or "").strip() or f"{root}/{inp}/uniprot/{s}.fasta" for s, r in samples.items()}
    for s in samples:
        add(f"prepare.{s}", STEPS[1], f"prepare {s}", "01_prepare.sh", [s, src[s], f"{a}/fasta/{s}.fasta", f"{a}/fasta/{s}.tsv", f"{a}/fasta/{s}.json"],
            [f"{a}/fasta/{s}.fasta", f"{a}/fasta/{s}.tsv", f"{a}/fasta/{s}.json"], f"{a}/logs/01_prepare/{s}.log", binds=outside(src[s], root), inputs=[src[s]])
    dd, emb_in, emb_dep = f"{a}/dedup", {s: f"{a}/fasta/{s}.fasta" for s in samples}, {s: f"prepare.{s}" for s in samples}
    if D["enabled"]:
        if D["scope"] not in ("sample", "joint"): raise ValueError(f"dedup.scope {D['scope']!r}: sample or joint")
        groups = [[s] for s in samples] if D["scope"] == "sample" else [list(samples)]
        for g in groups:
            k = f"dedup.{g[0]}" if D["scope"] == "sample" else "dedup"
            add(k, STEPS[2], f"dedup {g[0] if D['scope'] == 'sample' else 'all samples'}", "02_dedup.sh", [D["scope"], dd] + [f"{a}/fasta/{s}.fasta" for s in g], [f"{dd}/{s}.{x}" for s in g for x in ("fasta", "clusters.tsv", "dedup.json")],
                f"{a}/logs/02_dedup/{k}.log", [f"prepare.{s}" for s in g], inputs=[f"{a}/fasta/{s}.fasta" for s in g])
            for s in g: emb_in[s], emb_dep[s] = f"{dd}/{s}.fasta", k
    for b in backends:
        part = S["dgx_partition"] if b in S["dgx_backends"] else S["gpu_partition"]
        for s in samples:
            n = len(lengths_of(src[s])); K = max(1, -(-n // int(E["shard_proteins"])))
            root_s = f"{a}/embeddings/{b}/{s}.shards"
            for k in range(1, K + 1):
                add(f"embed.{b}.{s}.{k}", STEPS[3], f"embed {b} {s} {k}/{K}", "03_embed.sh", [s, b, tools[b]["env"], emb_in[s], k, K, f"{root_s}/shard_{k}", f"{root_s}/shard_{k}.done.json"],
                    [f"{root_s}/shard_{k}.done.json"], f"{a}/logs/03_embed/{b}/{s}.{k}.log", [emb_dep[s], f"weights.{b}"], inputs=[emb_in[s], f"data/models/{b}/CHECKSUMS"], gpu=True, res=GPU_RES.get(b, (2, 8000, 240)), part=part)
            add(f"merge.{b}.{s}", STEPS[4], f"merge {b} {s}", "04_merge.sh", [s, b, K, root_s, f"{a}/embeddings/{b}"], [f"{a}/embeddings/{b}/{s}.index.json", f"{a}/embeddings/{b}/{s}.ids.json", f"{a}/embeddings/{b}/{s}.json"],
                f"{a}/logs/04_merge/{b}/{s}.log", [f"embed.{b}.{s}.{k}" for k in range(1, K + 1)], inputs=[f"{root_s}/shard_{k}.done.json" for k in range(1, K + 1)])
    deps = [j["key"] for j in jobs if kind(j["key"]) in ("prepare", "dedup", "merge")]
    twonn = bool(((P.get("summarize") or {}).get("twonn") or {}).get("enabled"))
    reps_mb = -int(-estimate(name, root)["total_gb"] * 1000 // 1)
    sum_mb = max(RES["summary"][1], reps_mb) + (TWONN_RUNTIME_MB if twonn else 0)
    sum_why = f"default: total size of the reps it loads ({reps_mb} MB, minimum 1 GB)" + (f" + {TWONN_RUNTIME_MB // 1000} GB for the jax/DADApy runtime of the intrinsic dimension" if twonn else "")
    add("summary", STEPS[5], "summary", "05_summary.sh", [], [f"{a}/summary.tsv", f"{a}/samples_summary.tsv", f"{a}/aggregates.json"], f"{a}/logs/05_summary/summary.log", deps, inputs=[o for j in jobs if j["key"] in deps for o in j["outputs"]], res=(1, sum_mb, RES["summary"][2]), why=sum_why)
    check_paths(jobs, name, root)
    return jobs
def estimate(name: str, root: str = ROOT) -> dict:
    """Size of the final embeddings: proteins x (mean: dim | tokens: residues x dim) x layers x bytes, per sample and backend, against embed.size_warn_gb / size_cap_gb."""
    inp = f"{root}/results/{name}/inputs"
    P = analysis.load_yaml(f"{inp}/params.yaml"); E = {**EMBED, **(P.get("embed") or {})}
    tools = backends_mod.registry(f"{inp}/tools.tsv"); rows = []
    nbytes = 2 if E["store_dtype"] == "float16" else 4
    for r in analysis.read_sheet(f"{inp}/samples.tsv"):
        L = lengths_of((r.get("fasta_path") or "").strip() or f"{inp}/uniprot/{r['sample_id']}.fasta")
        for b in E["backends"]:
            spec = tools[b]; layers = backends_mod.resolve_layers(spec, E["layers"])
            lim = min([x for x in (1022 if spec["family"] == "esm2" else 0, int(E["max_length"])) if x] or [0])
            res = sum(min(x, lim) if lim else x for x in L)
            pools = [backends_mod.poolings(E["pooling"], k, spec["n_layers"]) for k in layers]
            gb = (len(L) * sum("mean" in p for p in pools) + res * sum("tokens" in p for p in pools)) * spec["dim"] * nbytes / 1e9
            rows.append(dict(sample=r["sample_id"], backend=b, proteins=len(L), residues=res, layers=len(layers), dim=spec["dim"], gb=round(gb, 3)))
    total = round(sum(x["gb"] for x in rows), 3)
    return dict(rows=rows, total_gb=total, warn_gb=float(E["size_warn_gb"]), cap_gb=float(E["size_cap_gb"]), level="refuse" if total > float(E["size_cap_gb"]) else "warn" if total > float(E["size_warn_gb"]) else "ok")
def check_size(name: str, root: str, allow_large: bool) -> dict:
    """Print the size estimate; refuse above the hard cap unless allow_large."""
    est = estimate(name, root)
    for x in est["rows"]: log(f"size {x['sample']} {x['backend']}: {x['proteins']:,} proteins, {x['residues']:,} residues, {x['layers']} layer(s) x {x['dim']} -> {x['gb']:.3f} GB")
    log(f"estimated embeddings: {est['total_gb']:.3f} GB (warn above {est['warn_gb']} GB, refuse above {est['cap_gb']} GB)")
    if est["level"] == "warn": log(f"!! large output: {est['total_gb']:.1f} GB")
    if est["level"] == "refuse" and not allow_large: raise ValueError(f"estimated embeddings {est['total_gb']:.1f} GB exceed embed.size_cap_gb {est['cap_gb']} GB: reduce layers/tokens, use float16, or pass --allow-large")
    return est
def check_paths(jobs: list[dict], name: str, root: str) -> None:
    """Refuse outputs or logs outside results/<name>/ (or data/models/ for the shared weights jobs)."""
    base, errs = os.path.realpath(root), []
    for j in jobs:
        ok = [f"{base}/results/{name}/"] + ([f"{base}/data/models/"] if j["shared"] else [])
        errs += [f"{os.path.relpath(p, root)} resolves to {os.path.realpath(p)}, outside the analysis results dir" for p in j["outputs"] + [j["log"]] if not os.path.realpath(p).startswith(tuple(ok))]
    if errs: raise ValueError(f"unsafe output paths ({len(errs)}): " + "; ".join(errs[:3]))
###################################### STATE
def weights_ok(job: dict) -> bool:
    """A weights folder counts only for the revision pinned now."""
    src = os.path.join(os.path.dirname(job["outputs"][0]), "SOURCE.json")
    try: return json.load(open(src)).get("revision") == job["revision"]
    except (OSError, ValueError): return False
def is_done(job: dict, jobs_by_key: dict) -> bool:
    """Outputs non-empty, and (non-shared) the log ends with the footer and no output is older than its dependencies' outputs."""
    if not job["outputs"] or any(size(o) == 0 for o in job["outputs"]): return False
    if job["shared"]: return weights_ok(job) if kind(job["key"]) == "weights" else True
    if not os.path.isfile(job["log"]): return False
    tail = [l for l in open(job["log"], errors="replace").read().splitlines() if l.strip()][-5:]
    if not any(FOOTER.search(l) for l in tail): return False
    newest = max((os.path.getmtime(o) for d in job["deps"] if not jobs_by_key[d]["shared"] for o in jobs_by_key[d]["outputs"] if os.path.exists(o)), default=0)
    return all(os.path.getmtime(o) >= newest for o in job["outputs"])
def todo(name: str, root: str = ROOT) -> tuple[list[dict], set]:
    """Plan and mark the jobs to run: not done, or downstream of a job that runs."""
    jobs = plan(name, root); by = {j["key"]: j for j in jobs}; run = set()
    for j in jobs:
        if not is_done(j, by) or (not j["shared"] and any(d in run for d in j["deps"])): run.add(j["key"])
    return jobs, run
###################################### SAFETY
def verify_inputs(name: str, root: str) -> None:
    """Refuse a launch when frozen inputs, the pins or a local FASTA changed since the analysis was created."""
    inp = f"{root}/results/{name}/inputs"
    meta = json.load(open(f"{inp}/analysis.json"))
    for rel, want in meta["checksums"].items():
        if not os.path.isfile(f"{inp}/{rel}") or analysis.sha256(f"{inp}/{rel}") != want: raise ValueError(f"frozen input {inp}/{rel} changed or is missing since the analysis was created: create a new analysis instead of editing frozen inputs")
    if analysis.sha256(f"{root}/configs/tools.tsv") != meta["checksums"]["tools.tsv"]: raise ValueError("configs/tools.tsv (pins) changed since the analysis was created: create a new analysis for the new pins")
    for p, w in meta.get("local_fasta", {}).items():
        st = os.stat(p) if os.path.isfile(p) else None
        if not st or st.st_size != w["size"] or st.st_mtime != w["mtime"]: raise ValueError(f"local FASTA {p} changed or is missing since the analysis was created: create a new analysis")
def preflight(jobs: list[dict], run: set) -> None:
    """Every input of every job to run exists or is produced in this launch; otherwise nothing is submitted."""
    made, errs = {o for j in jobs if j["key"] in run for o in j["outputs"]}, []
    for j in jobs:
        if j["key"] not in run: continue
        errs += [f"{j['key']}: missing or empty input {p}" for p in j["inputs"] if p not in made and not (size(p) and os.access(p, os.R_OK))]
    if errs: raise ValueError("pre-flight failed, nothing submitted:\n  " + "\n  ".join(errs))
def live_ids(rows: list[dict]) -> list[dict]:
    """Rows of jobs.tsv whose job is still queued or running."""
    st = slurm_states([r["jobid"] for r in rows])
    return [r for r in rows if st.get(r["jobid"]) in LIVE]
def busy(name: str, jobs: list[dict], run: set, root: str) -> None:
    """Refuse while this analysis has live jobs, or another analysis has live jobs writing the same shared weights."""
    own = live_ids(read_jobs(name, root))
    if own: raise RuntimeError(f"{name} still has live jobs ({', '.join(r['jobid'] + ' ' + r['key'] for r in own)}): stop them first (--stop)")
    keys = {j["key"] for j in jobs if j["shared"] and j["key"] in run}
    others = [r for p in glob.glob(f"{root}/results/*/jobs.tsv") if os.path.basename(os.path.dirname(p)) != name for r in read_jobs(os.path.basename(os.path.dirname(p)), root) if r["key"] in keys]
    live = live_ids(others)
    if live: raise RuntimeError(f"another analysis has live jobs downloading the same weights: {', '.join(r['jobid'] + ' ' + r['key'] for r in live)}")
###################################### SIZING
def mb(v: str) -> float:
    """sacct memory string -> MB."""
    m = re.match(r"^([0-9.]+)([KMGT]?)", v or "")
    return float(m.group(1)) * {"": 1e-6, "K": 1e-3, "M": 1, "G": 1024, "T": 1048576}[m.group(2)] if m else 0.0
def to_min(v: str) -> float:
    """SLURM time string -> minutes."""
    d, _, t = (v or "").rpartition("-"); p = [float(x) for x in t.split(":") if x.replace(".", "").isdigit()]
    if not p: return 0.0
    while len(p) < 3: p.insert(0, 0.0)
    return (float(d) if d else 0.0) * 1440 + p[0] * 60 + p[1] + p[2] / 60
def usage(ids: list[str]) -> dict:
    """sacct state, MaxRSS, Elapsed and TotalCPU per job id."""
    out = {}
    for i in range(0, len(ids), 200):
        for l in sh(["sacct", "-n", "-P", "-j", ",".join(ids[i:i + 200]), "-o", "JobID,State,MaxRSS,Elapsed,TotalCPU,AllocCPUS"]):
            f = l.split("|") + ["", ""]
            if len(f) < 6: continue
            d = out.setdefault(f[0].split(".")[0], dict(state="", rss=0.0, minutes=0.0, cpu_min=0.0, cpus=0))
            if "." not in f[0]: d.update(state=f[1].split()[0] if f[1].strip() else "", cpu_min=to_min(f[4]), cpus=int(f[5]) if f[5].isdigit() else 0)
            d.update(rss=max(d["rss"], mb(f[2])), minutes=max(d["minutes"], to_min(f[3])))
    return out
def log_usage(path: str) -> tuple[float, float]:
    """(peak MB, minutes) from a job log footer."""
    if not path or not os.path.isfile(path): return 0.0, 0.0
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END); f.seek(max(0, f.tell() - 8192)); m = PEAK_RE.findall(f.read().decode("utf-8", "replace"))
    return (float(m[-1][1]) * 1024, to_min(m[-1][0])) if m else (0.0, 0.0)
def history(root: str) -> dict[str, list[dict]]:
    """Past runs of every analysis per sizing group (sacct first, log footer as fallback)."""
    rows = [r for p in glob.glob(f"{root}/results/*/jobs.tsv") for r in read_jobs(os.path.basename(os.path.dirname(p)), root) if r.get("jobid") and r.get("mem_mb")]
    use, out = usage([r["jobid"] for r in rows]), {}
    for r in rows:
        u, (lp, lt) = use.get(r["jobid"], {}), log_usage(r.get("log"))
        state = u.get("state") or ("COMPLETED" if lp else "")
        if not state: continue
        rss = u.get("rss", 0.0)
        out.setdefault(group(r["key"]), []).append(dict(state=state, peak=rss or lp, minutes=max(lt, u.get("minutes", 0.0)), src="sacct MaxRSS" if rss else "log cgroup peak" if lp else "none",
                                                        cores_used=u.get("cpu_min", 0.0) / u["minutes"] if u.get("minutes") and u.get("cpu_min") else 0.0, cpus_req=float(r.get("cpus") or 0),
                                                        mem_mb=float(r["mem_mb"]), minutes_req=float(r.get("minutes") or 0), in_bytes=float(r.get("in_bytes") or 0)))
    return out
def size_job(j: dict, obs: list[dict], limits: dict) -> None:
    """Memory, time and cores from past runs (input-size ratio, OOM/TIMEOUT doubling, clamps); defaults when there is no history."""
    c, m0, t0 = j["cpus"], j["mem_mb"], j["minutes"]
    inb = sum(size(p) for p in j["inputs"])
    ok = [o for o in obs if o["state"] == "COMPLETED" and o["peak"] > 0]
    scale = lambda o: max(1.0, inb / o["in_bytes"]) if o["in_bytes"] and inb else 1.0
    if ok:
        ref = max(ok, key=lambda o: o["peak"] * scale(o))
        mem, why = SAFETY * ref["peak"] * scale(ref) + 512, f"peak {ref['peak'] / 1024:.1f} GB ({ref['src']}) x input ratio {scale(ref):.2f} x {SAFETY} + 0.5 GB, best of {len(ok)} run(s)"
        est = max(o["minutes"] * scale(o) for o in ok); t = 2 * est + 10
    else: mem, why, t, est = m0, j.get("why") or (("default: " + GPU_WHY) if j["gpu"] else "default 1 core / 1 GB"), t0, EST[kind(j["key"])]
    oom = [2 * o["mem_mb"] for o in obs if o["state"] == "OUT_OF_MEMORY"]
    if oom and max(oom) > mem: mem, why = max(oom), f"2 x the {max(oom) / 2048:.1f} GB of a job that ran out of memory"
    t = max([t] + [2 * o["minutes_req"] for o in obs if o["state"] == "TIMEOUT"])
    used = [(o["cores_used"], o["cpus_req"]) for o in ok if o["cores_used"] > 0]
    if used and not j["gpu"]:
        hot = [r for u, r in used if r and u >= CPU_HOT * r]
        c = min(CPU_MAX, int(2 * max(hot))) if hot else max(1, min(CPU_MAX, -int(-max(u for u, _ in used) * CPU_SAFETY // 1)))
    cap = GPU_MEM_MAX if j["gpu"] else MEM_MAX
    tmax = min(TIME_MAX, limits.get(j["partition"], TIME_MAX)) if j["partition"] else TIME_MAX
    j.update(cpus=c, mem_mb=int(min(cap, max(MEM_MIN, mem))), est=round(est, 1), minutes=int(min(tmax, max(TIME_MIN, t))), in_bytes=inb,
             sized=f"{why} = {mem / 1024:.1f} GB" + (f" !! capped at {cap // 1000} GB" if mem > cap else "") + (f" !! time capped at the {tmax:.0f} min limit of {j['partition']}" if t > tmax else ""))
###################################### SCHEDULE
def node_table() -> list[dict]:
    """CPU nodes from sinfo with idle cores, free memory and the half-node ceiling."""
    out = {}
    for l in sh(["sinfo", "-N", "-h", "-O", NODE_FMT]):
        f = l.split()
        if len(f) < 6 or f[3].count("/") != 3: continue
        _, i, _, t = (int(x) if x.isdigit() else 0 for x in f[3].split("/"))
        mem, al = int(float(f[4])) if f[4].replace(".", "").isdigit() else 0, int(float(f[5])) if f[5].replace(".", "").isdigit() else 0
        n = out.setdefault(f[0], dict(node=f[0], partitions=[], state=f[2], cpus=t, idle=i, mem_mb=mem, free_mb=max(0, mem - al)))
        if f[1].rstrip("*") not in n["partitions"]: n["partitions"].append(f[1].rstrip("*"))
    for n in out.values():
        n["parts"] = [p for p in n["partitions"] if not SKIP_PARTS.match(p)]
        n["usable"] = bool(n["parts"]) and n["state"] in USABLE and not re.match(r"^(dgx|gpu)", n["node"], re.I)
        n["score"] = round((n["idle"] / n["cpus"] if n["cpus"] else 0) + (n["free_mb"] / n["mem_mb"] if n["mem_mb"] else 0), 3)
    return sorted(out.values(), key=lambda n: (not n["usable"], -n["score"], n["node"]))
def part_limits() -> dict[str, float]:
    """Walltime limit of every partition, in minutes."""
    out = {}
    for l in sh(["sinfo", "-h", "-o", "%P %l"]):
        f = l.split()
        if len(f) == 2: out[f[0].rstrip("*")] = 1e9 if f[1] in ("infinite", "UNLIMITED") else to_min(f[1])
    return out
def cpu_node(jobs: list[dict], S: dict, root: str) -> tuple[dict, str]:
    """One CPU node for the CPU lane and its ceiling (node_fraction of the node, from sinfo)."""
    w, m = max(j["cpus"] for j in jobs), max(j["mem_mb"] for j in jobs)
    want = max(collections.Counter(levels(jobs).values()).values(), default=1)
    if os.environ.get("NODE_PICK", "1") == "0": return dict(node="", part=S["partition"], cpus=min(BUDGET_CPUS, want * w), mem=BUDGET_MEM_MB), "node pick off"
    try: r = subprocess.run([f"{root}/src/node_pick.sh", str(want * w), str(min(BUDGET_MEM_MB, want * m)), str(max(j["minutes"] for j in jobs))], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError): r = None
    out = r.stdout.split() if r else []
    if len(out) != 2: return dict(node="", part=S["partition"], cpus=min(BUDGET_CPUS, want * w), mem=BUDGET_MEM_MB), f"no CPU node has {want * w} cpus free now: CPU jobs queue on {S['partition']} without a node pin"
    t = {n["node"]: n for n in node_table()}.get(out[0])
    frac = float(S["node_fraction"])
    c, mm = (int(t["cpus"] * frac), int(t["mem_mb"] * frac)) if t else (BUDGET_CPUS, BUDGET_MEM_MB)
    return dict(node=out[0], part=out[1], cpus=max(w, min(c, want * w)), mem=mm), f"{out[0]} ({out[1]}), ceiling {frac} x node = {c} cpus {mm // 1000} GB, DAG width {want} {r.stderr.strip() if r else ''}"
def levels(jobs: list[dict]) -> dict[str, int]:
    """Depth of every job in the DAG."""
    lv = {}
    for j in jobs: lv[j["key"]] = 1 + max([lv[d] for d in j["deps"] if d in lv], default=-1)
    return lv
def schedule(jobs: list[dict], node: dict, S: dict) -> dict:
    """DAG-aware launch plan: CPU jobs hold cpu tokens of the pinned node, GPU jobs hold one of gpus_max GPU tokens (DGX jobs also the single DGX token); each job waits (afterany) only for the previous holders of its tokens, longest remaining chain first."""
    keys, kids = {j["key"] for j in jobs}, collections.defaultdict(list)
    for j in jobs:
        for d in j["deps"]:
            if d in keys: kids[d].append(j["key"])
    bl = {}
    for j in reversed(jobs): bl[j["key"]] = j.get("est", 1) + max([bl[k] for k in kids[j["key"]]], default=0)
    mbt = node["mem"] / max(1, node["cpus"])
    toks = [dict(id=f"cpu#{i}", lane="cpu", free=0.0, last="") for i in range(max(1, node["cpus"]))] + [dict(id=f"gpu#{i}", lane="gpu", free=0.0, last="") for i in range(int(S["gpus_max"]))] + [dict(id="dgx", lane="dgx", free=0.0, last="")]
    placed, order, prev, start, end, held = set(), [], {}, {}, {}, {}
    while len(placed) < len(jobs):
        best = None
        for j in jobs:
            if j["key"] in placed or any(d in keys and d not in placed for d in j["deps"]): continue
            ready = max([end[d] for d in j["deps"] if d in end], default=0.0)
            if j["gpu"]:
                pick = sorted((t for t in toks if t["lane"] == "gpu"), key=lambda t: t["free"])[:1] + ([t for t in toks if t["lane"] == "dgx"] if j["partition"] == S["dgx_partition"] else [])
            else:
                pool = sorted((t for t in toks if t["lane"] == "cpu"), key=lambda t: t["free"])
                pick = pool[:min(len(pool), max(j["cpus"], -int(-j["mem_mb"] / mbt // 1)))]
            s0 = max([ready] + [t["free"] for t in pick])
            cand = (s0, -bl[j["key"]], j["key"])
            if best is None or cand < best[0]: best = (cand, j, pick, s0)
        _, j, pick, s0 = best
        prev[j["key"]] = sorted({t["last"] for t in pick if t["last"]})
        for t in pick: t["free"], t["last"] = s0 + j.get("est", 1), j["key"]
        placed.add(j["key"]); order.append(j["key"]); start[j["key"]] = s0; end[j["key"]] = s0 + j.get("est", 1); held[j["key"]] = [t["id"] for t in pick]
    return dict(order=order, prev=prev, start=start, end=end, held=held)
###################################### GUARDS
def user_live() -> int:
    """Number of jobs of the user in the queue."""
    return len(sh(["squeue", "-h", "-u", getpass.getuser(), "-o", "%i"]))
def tagged(name: str) -> list[str]:
    """Queued job ids whose comment tags them as this analysis."""
    return [l.split("|")[0] for l in sh(["squeue", "-h", "-u", getpass.getuser(), "-o", "%i|%k"]) if l.split("|", 1)[-1].startswith(f"mp:{name}:")]
def guard(n: int) -> None:
    """At most MAX_JOBS per launch and MAX_USER_JOBS in the user's queue."""
    if n > MAX_JOBS: raise ValueError(f"this launch would submit {n} jobs, above the limit of {MAX_JOBS}: split the samples or backends into smaller analyses")
    live = user_live()
    if live + n > MAX_USER_JOBS: raise RuntimeError(f"you already have {live} jobs in the SLURM queue; {n} more would pass the limit of {MAX_USER_JOBS}: wait or stop some")
###################################### SUBMIT
def job_env(root: str, name: str, img: str, job: dict) -> dict:
    """Environment of one job: SLURM_*/SBATCH_* scrubbed, MP_* added (the HF token travels only as a file path, online jobs only)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SBATCH_", "MP_HF_TOKEN"))}
    t, st = meta_target(job["key"])
    env.update(MP_ROOT=os.path.realpath(root), MP_ANALYSIS=name, MP_IMAGE=img, MP_BINDS=",".join(job["binds"]), MP_OUTPUTS=":".join(job["outputs"]), MP_GPU="1" if job["gpu"] else "0", MP_ONLINE="1" if job["online"] else "0",
               MP_META_TARGET=t, MP_META_STEP=st, MP_IN_BYTES=str(int(job.get("in_bytes") or 0)), MP_ASK=json.dumps(dict(cpus=job["cpus"], mem_mb=job["mem_mb"], time_min=job["minutes"], gpu=job["gpu"], partition=job["partition"], why=job.get("sized", ""))))
    tok = os.environ.get("MP_HF_TOKEN_FILE") or os.path.expanduser("~/.cache/huggingface/token")
    if job["online"] and os.path.isfile(tok): env["MP_HF_TOKEN_FILE"] = tok
    return env
def run_here(name: str, root: str, jobs: list[dict]) -> None:
    """Run the online jobs (downloads) in the foreground on this host, inside the image, for clusters whose compute nodes have no internet."""
    img = image(root)
    for j in jobs:
        os.makedirs(os.path.dirname(j["log"]), exist_ok=True)
        for o in j["outputs"]: os.makedirs(os.path.dirname(o), exist_ok=True)
        log(f"{j['key']}: running here (online job), log {os.path.relpath(j['log'], root)}")
        with open(j["log"], "w") as fh: rc = subprocess.run([f"{root}/src/mp_job.sh", j["script"]] + j["args"], env=job_env(root, name, img, j), stdout=fh, stderr=subprocess.STDOUT).returncode
        if rc: raise RuntimeError(f"{j['key']} failed here (exit {rc}): see {j['log']}")
def submit(name: str, root: str = ROOT, dry_run: bool = False, online_here: bool = False, allow_large: bool = False) -> list[dict]:
    """Dry run, or launch under the analysis lock."""
    if dry_run:
        verify_inputs(name, root); jobs, run = todo(name, root); preflight(jobs, run); check_size(name, root, allow_large)
        return [dict(j, state="would-run" if j["key"] in run else "done", jobid="") for j in jobs]
    d = analysis.analysis_dir(name, root); os.makedirs(d, exist_ok=True)
    if os.path.realpath(d) != f"{os.path.realpath(root)}/results/{name}": raise ValueError(f"{d} resolves to {os.path.realpath(d)}, outside the repo results/")
    lock = open(f"{d}/.lock", "w")
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError: raise RuntimeError(f"another launch of {name} is in progress")
    try: return launch(name, root, online_here, allow_large)
    finally: lock.close()
def launch(name: str, root: str, online_here: bool = False, allow_large: bool = False) -> list[dict]:
    """Freeze, verify, plan, size, schedule and sbatch the jobs that are not done; jobs.tsv is rewritten after every sbatch."""
    _, _, errs = analysis.create(name, root)
    if errs: raise ValueError("; ".join(errs))
    verify_inputs(name, root)
    jobs, run = todo(name, root)
    preflight(jobs, run); check_size(name, root, allow_large); busy(name, jobs, run, root)
    if online_here and any(j["online"] and j["key"] in run for j in jobs):
        run_here(name, root, [j for j in jobs if j["online"] and j["key"] in run]); jobs, run = todo(name, root)
    d, img, ids, out = f"{root}/results/{name}", image(root), {}, []
    S = slurm_cfg(analysis.load_yaml(f"{d}/inputs/params.yaml"))
    if os.path.exists(f"{d}/.stop"): os.remove(f"{d}/.stop")
    todo_jobs = [j for j in jobs if j["key"] in run]
    if todo_jobs: guard(len(todo_jobs))
    hist, limits = (history(root), part_limits()) if todo_jobs else ({}, {})
    for j in todo_jobs: size_job(j, hist.get(group(j["key"]), []), limits)
    try: fill_perf(name, root)
    except (OSError, ValueError) as e: log(f"!! performance metadata not refreshed: {e}")
    cpu_jobs = [j for j in todo_jobs if not j["gpu"]]
    node, why = cpu_node(cpu_jobs, S, root) if cpu_jobs else (dict(node="", part=S["partition"], cpus=1, mem=1000), "no CPU job")
    for j in cpu_jobs: j["partition"] = node["part"]
    sch = schedule(todo_jobs, node, S) if todo_jobs else dict(order=[], prev={}, start={}, end={}, held={})
    by = {j["key"]: j for j in jobs}
    plog = [f"[PYTHON-INFO] {datetime.datetime.now():%F %T} launch {name}: {len(todo_jobs)} job(s); CPU lane {why}; GPU lane {S['gpus_max']} GPU token(s), {S['gpu_flag']} on {S['gpu_partition']} ({S['dgx_partition']} for {', '.join(S['dgx_backends'])}, one DGX job at a time)"]
    plog += [f"[PYTHON-INFO]   {k}: {by[k]['cpus']} cpus {by[k]['mem_mb']} MB {by[k]['minutes']} min on {by[k]['partition']}{' +GPU' if by[k]['gpu'] else ''}; starts ~{sch['start'][k]:.0f} min; waits for {', '.join(sch['prev'][k]) or 'its inputs only'} ({by[k]['sized']})" for k in sch["order"]]
    os.makedirs(f"{d}/logs", exist_ok=True); open(f"{d}/logs/node_pick.log", "a").write("\n".join(plog) + "\n")
    log(plog[0].split("] ", 1)[1])
    before = {r["key"]: r for r in read_jobs(name, root)}
    for j in [j for j in jobs if j["key"] not in run] + [by[k] for k in sch["order"]]:
        if j["key"] not in run: out.append(dict(j, **{k: v for k, v in before.get(j["key"], {}).items() if k in ("jobid", "node", "cpus", "mem_mb", "minutes", "in_bytes") and v}, state="done")); continue
        if os.path.exists(f"{d}/.stop"):
            if ids: subprocess.run(["scancel"] + list(ids.values()), capture_output=True)
            write_jobs(name, root, out); raise RuntimeError(f"stopped while submitting: {len(ids)} submitted job(s) cancelled")
        for p in [os.path.dirname(o) for o in j["outputs"]] + [os.path.dirname(j["log"])]: os.makedirs(p, exist_ok=True)
        dep = [ids[x] for x in j["deps"] if x in ids]; after = [ids[x] for x in sch["prev"].get(j["key"], []) if x in ids and ids[x] not in dep]
        when = ",".join(([f"afterok:{':'.join(dep)}"] if dep else []) + ([f"afterany:{':'.join(after)}"] if after else []))
        pin = ["-w", node["node"]] if not j["gpu"] and node["node"] else []
        cmd = ["sbatch", "--parsable", "-J", f"mp_{name}_{j['key']}", "-p", j["partition"]] + pin + (S["gpu_flag"].split() if j["gpu"] else []) + ["-N", "1", "-c", str(j["cpus"]), f"--mem={j['mem_mb']}", "-t", str(j["minutes"]),
            "-o", j["log"], "--kill-on-invalid-dep=yes", "--comment", f"mp:{name}:{j['key']}"] + ([f"--dependency={when}"] if when else []) + [f"{root}/src/mp_job.sh", j["script"]] + j["args"]
        r = subprocess.run(cmd, capture_output=True, text=True, env=job_env(root, name, img, j))
        jid = r.stdout.strip().split(";")[0] if r.returncode == 0 else ""
        if not jid:
            if ids: subprocess.run(["scancel"] + list(ids.values()), capture_output=True)
            raise RuntimeError(f"sbatch failed for {j['key']}: {(r.stderr or r.stdout).strip()}")
        ids[j["key"]] = jid; out.append(dict(j, state="submitted", jobid=jid, node=node["node"] if not j["gpu"] else "")); write_jobs(name, root, out)
    write_jobs(name, root, out)
    return out
def write_jobs(name: str, root: str, out: list[dict]) -> None:
    """Rewrite results/<name>/jobs.tsv atomically."""
    p = f"{root}/results/{name}/jobs.tsv"
    with open(f"{p}.tmp", "w") as f:
        f.write("\t".join(COLS) + "\n")
        for j in out: f.write("\t".join(str(int(j[c]) if c == "gpu" else j.get(c, "")) for c in COLS) + "\n")
    os.replace(f"{p}.tmp", p)
def read_jobs(name: str, root: str = ROOT) -> list[dict]:
    """Rows of results/<name>/jobs.tsv."""
    p = f"{root}/results/{name}/jobs.tsv"
    return list(csv.DictReader(open(p), delimiter="\t")) if os.path.exists(p) else []
def slurm_states(jobids: list[str]) -> dict[str, str]:
    """State per job id: squeue first, then sacct for the finished ones."""
    ids, st = [i for i in jobids if i], {}
    if not ids: return st
    for l in sh(["squeue", "-h", "-j", ",".join(ids), "-o", "%i|%T"]):
        i, _, s = l.partition("|"); st[i.strip()] = s.strip()
    rest = [i for i in ids if i not in st]
    for l in sh(["sacct", "-n", "-X", "-P", "-j", ",".join(rest), "-o", "JobID,State"]) if rest else []:
        i, _, s = l.partition("|"); st[i.strip()] = s.split()[0] if s.strip() else ""
    return st
def fill_perf(name: str, root: str = ROOT) -> None:
    """Add the sacct view (state, CPU and memory efficiency, queue wait) to each finished job's perf record in meta/*.json."""
    rows, meta, need = [r for r in read_jobs(name, root) if r.get("jobid")], f"{root}/results/{name}/meta", []
    for r in rows:
        t, st = meta_target(r["key"]); p = f"{meta}/{t}.json"
        cur = ((json.load(open(p)).get("steps") or {}).get(st) or {}).get("perf") or {} if os.path.exists(p) else {}
        if not (cur.get("sacct") and str(cur.get("jobid")) == r["jobid"]): need.append((r, t, st, cur))
    acct, ids = {}, [r["jobid"] for r, *_ in need]
    for i in range(0, len(ids), 200):
        for l in sh(["sacct", "-n", "-P", "-j", ",".join(ids[i:i + 200]), "-o", ACCT]):
            f = l.split("|")
            if len(f) < 9: continue
            d = acct.setdefault(f[0].split(".")[0], dict(rss=0.0, cpu=0.0, el=0.0))
            if "." not in f[0]: d.update(state=f[1].split()[0] if f[1].strip() else "", cpus=int(f[5]) if f[5].isdigit() else 0, req=mb(f[6]), submit=f[7], start=f[8])
            d.update(rss=max(d["rss"], mb(f[4])), cpu=max(d["cpu"], to_min(f[2])), el=max(d["el"], to_min(f[3])))
    when = lambda v: datetime.datetime.fromisoformat(v) if v and v[0].isdigit() else None
    for r, t, st, cur in need:
        a = acct.get(r["jobid"])
        if not a or not a.get("state") or a["state"] in LIVE: continue
        s0, s1, req = when(a.get("submit")), when(a.get("start")), float(r.get("mem_mb") or 0) or a.get("req") or 0
        sa = dict(state=a["state"], total_cpu_s=round(a["cpu"] * 60), elapsed_s=round(a["el"] * 60), max_rss_gb=round(a["rss"] / 1024, 2), gpu_h=round(a["el"] / 60, 2) if r.get("gpu") == "1" else 0,
                  cpu_eff=round(100 * a["cpu"] / (a["el"] * a["cpus"]), 1) if a["el"] and a["cpus"] else None, mem_eff=round(100 * a["rss"] / req, 1) if req and a["rss"] else None,
                  wait_min=round((s1 - s0).total_seconds() / 60, 1) if s0 and s1 else None)
        sample_meta.update_perf(meta, t, st, dict(jobid=r["jobid"], sacct=sa), f"{root}/results/{name}")
def stop(name: str, root: str = ROOT) -> dict:
    """Cancel every job of the analysis (ids of jobs.tsv plus tagged queue entries), up to 3 rounds."""
    d = analysis.analysis_dir(name, root)
    if not os.path.isdir(d): return dict(cancelled=[], remaining=[])
    open(f"{d}/.stop", "w").close()
    lock, t = open(f"{d}/.lock", "w"), time.time()
    while True:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); break
        except OSError:
            if time.time() - t > STOP_WAIT: break
            time.sleep(0.5)
    try:
        ids = sorted({j["jobid"] for j in read_jobs(name, root) if j["jobid"]} | set(tagged(name)))
        alive = lambda xs: (lambda st: [i for i in xs if st.get(i) in LIVE])(slurm_states(xs))
        live = first = alive(ids)
        for k in range(3):
            if not live: break
            subprocess.run(["scancel"] + live, capture_output=True)
            time.sleep(STOP_PAUSE * k)
            live = alive(live)
        return dict(cancelled=sorted(set(first) - set(live)), remaining=live)
    finally:
        if os.path.exists(f"{d}/.stop"): os.remove(f"{d}/.stop")
        lock.close()
###################################### MAIN
def main() -> None:
    """Command line: plan and submit, dry-run, or stop one analysis."""
    ap = argparse.ArgumentParser(description="Plan and submit the MultiPLM jobs of an analysis"); ap.add_argument("analysis_name"); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--stop", action="store_true")
    ap.add_argument("--online-here", action="store_true", help="run the weights download jobs on this host, inside the image, before submitting the rest (UniProt: use src/prefetch.py)"); ap.add_argument("--allow-large", action="store_true", help="submit even above embed.size_cap_gb"); ap.add_argument("--root", default=ROOT, help=argparse.SUPPRESS)
    a = ap.parse_args()
    try:
        if a.stop:
            r = stop(a.analysis_name, a.root)
            log(f"cancelled {len(r['cancelled'])} job(s)" + (f"; !! still live after 3 scancel rounds: {' '.join(r['remaining'])}" if r["remaining"] else ""))
        else:
            out = submit(a.analysis_name, a.root, a.dry_run, a.online_here, a.allow_large)
            for j in out: log(f"{j['key']}: {'submitted ' + j['jobid'] if j['state'] == 'submitted' else j['state']}{' [GPU ' + j['partition'] + ']' if j['gpu'] else ''}")
            n = lambda s: sum(j["state"] == s for j in out)
            log(f"{len(out)} jobs: {n('done')} done, {n('submitted')} submitted, {n('would-run')} would run")
    except (ValueError, RuntimeError) as e: sys.exit(f"[PYTHON-INFO] !! {e}")
    log(f"{datetime.datetime.now():%F %T} | elapsed {time.time() - T0:.1f} s | max RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576:.2f} GB")
if __name__ == "__main__":
    main()
