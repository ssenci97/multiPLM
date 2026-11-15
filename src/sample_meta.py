#!/usr/bin/env python3
"""Single writer of results/<analysis>/meta/*.json (sample and analysis records, perf records), locked and atomic; `key=@file` loads JSON or CHECKSUMS."""
###################################### DEPENDENCIES_AND_CONSTANTS
from __future__ import annotations
import csv
import datetime
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
ROOT = os.environ.get("MP_ROOT") or str(Path(__file__).resolve().parents[1])
OUT = os.environ.get("MP_OUT") or f"{ROOT}/results/default"
META_DIR = f"{OUT}/meta"
ANALYSIS = "_analysis"
PERF = dict(jobid=None, node=None, partition=None, gpu=None, asked=None, used=None, started=None, ended=None, sacct=None)
###################################### HELPER_FUNCTIONS
def scalar(v: str) -> object:
    """Parse a key=value value: @file loads JSON or a CHECKSUMS file, JSON literals are decoded, anything else stays a string."""
    if v.startswith("@"):
        p = Path(v[1:])
        if not p.is_file(): return None
        if p.suffix == ".json": return json.loads(p.read_text())
        return {rel.strip(): sha for sha, rel in (l.split(None, 1) for l in p.read_text().splitlines() if l.strip())}
    try: return json.loads(v)
    except ValueError: return v
def git_sha() -> str | None:
    """Short commit of the pipeline checkout, if it is a git repository."""
    try: return subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    except OSError: return None
def environment() -> dict:
    """Image, container flag and pinned software commits."""
    e = {k.lower(): os.environ[k] for k in ("MP_SIF", "MP_IMAGE") if os.environ.get(k)}
    e["containerized"] = bool(os.environ.get("APPTAINER_CONTAINER") or os.environ.get("SINGULARITY_CONTAINER") or os.path.isdir("/.singularity.d"))
    tsv = Path(ROOT) / "configs" / "tools.tsv"
    e["software"] = {r["name"]: r["commit"] for r in csv.DictReader(open(tsv), delimiter="\t") if r["kind"] == "software"} if tsv.is_file() else {}
    return e
def submitted(target: str, out: str = OUT) -> dict:
    """The frozen sample record of the console, if any."""
    p = Path(out) / "inputs" / "samples" / f"{target}.json"
    return json.loads(p.read_text()) if p.is_file() else {}
def new_doc(target: str, out: str) -> dict:
    """Empty document of a sample or of the analysis."""
    return {"analysis": os.path.basename(out), "steps": {}} if target == ANALYSIS else {"sample_id": target, "submitted": submitted(target, out), "steps": {}}
def write(meta_dir: str, target: str, change: object, out: str = OUT) -> None:
    """Locked read-modify-write of meta_dir/<target>.json with an atomic replace."""
    os.makedirs(meta_dir, exist_ok=True); p = f"{meta_dir}/{target}.json"
    with open(f"{p}.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        d = json.load(open(p)) if os.path.exists(p) else new_doc(target, out)
        d.setdefault("steps", {}); change(d)
        json.dump(d, open(f"{p}.tmp", "w"), indent=1)
        os.replace(f"{p}.tmp", p)
def update_perf(meta_dir: str, target: str, step: str, patch: dict, out: str = OUT) -> None:
    """Merge patch into steps.<step>.perf; a record of an older job id is replaced, not merged."""
    def change(d: dict) -> None:
        """Merge the perf patch into the step record."""
        s = d["steps"].setdefault(step, {}); old = s.get("perf") or {}
        if old.get("jobid") and patch.get("jobid") and str(old["jobid"]) != str(patch["jobid"]): old = {}
        s["perf"] = {**PERF, **old, **patch}
    write(meta_dir, target, change, out)
def cgroup_peak_gb() -> float | None:
    """Peak memory of the job cgroup (includes page cache: sacct MaxRSS is preferred later)."""
    try: rel = next(l.split(":", 2)[2].strip() for l in open("/proc/self/cgroup") if l.startswith("0::"))
    except (OSError, StopIteration): rel = ""
    for f in (f"/sys/fs/cgroup{rel}/memory.peak", "/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"):
        try: return round(int(open(f).read().strip()) / 1073741824, 2)
        except (OSError, ValueError): continue
    return None
def job_perf(kv: dict) -> dict:
    """The perf record the job wrapper writes when a job ends: what was asked and what was used."""
    ask = scalar(os.environ.get("MP_ASK", "null"))
    used = dict(in_bytes=int(os.environ.get("MP_IN_BYTES") or 0), elapsed_s=kv.get("elapsed_s"), max_rss_gb=cgroup_peak_gb(), exit=kv.get("exit"))
    return dict(jobid=os.environ.get("SLURM_JOB_ID"), node=socket.gethostname(), partition=os.environ.get("SLURM_JOB_PARTITION"), gpu=os.environ.get("MP_GPU") == "1",
                asked=ask if isinstance(ask, dict) else None, used=used, started=kv.get("started"), ended=kv.get("ended"))
###################################### MAIN
def main() -> None:
    """Record a step (or a perf record) for one sample or for the analysis."""
    if sys.argv[1:2] == ["--perf"]:
        target, step, *kv = sys.argv[2:]
        update_perf(META_DIR, target, step, job_perf({k: scalar(v) for k, v in (x.split("=", 1) for x in kv)}))
        return
    target, step, *kv = sys.argv[1:]
    def change(d: dict) -> None:
        """Update the step record with the given values and the environment."""
        if target != ANALYSIS: d["submitted"] = d.get("submitted") or submitted(target)
        d["pipeline_commit"], d["environment"] = git_sha(), environment()
        rec = {k: scalar(v) for k, v in (x.split("=", 1) for x in kv)}
        rec["recorded_at"] = f"{datetime.datetime.now():%F %T}"
        if os.environ.get("SLURM_JOB_ID"): rec["slurm_job_id"] = os.environ["SLURM_JOB_ID"]
        d["steps"].setdefault(step, {}).update(rec)
    write(META_DIR, target, change)
if __name__ == "__main__":
    main()
