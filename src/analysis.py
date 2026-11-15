#!/usr/bin/env python3
"""Create or resume an analysis: validate the inputs and settings, then freeze them (with prefetched UniProt FASTA and pins) into results/<name>/inputs with sha256."""
###################################### DEPENDENCIES_AND_CONSTANTS
from __future__ import annotations
import argparse
import csv
import datetime
import hashlib
import json
import os
import re
import resource
import shutil
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backends import poolings, registry, resolve_layers
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
ID_RE = re.compile(r"^[A-Za-z0-9_]+$")
FASTA_EXT = (".fa", ".faa", ".fasta", ".fas")
COLUMNS = ["sample_id", "fasta_path", "uniprot_query", "description"]
UNIPROT_DIR = "data/uniprot"
MAX_PROTEINS_SAMPLE = 100000
MAX_PROTEINS_ANALYSIS = 200000
###################################### HELPER_FUNCTIONS
def log(m: str) -> None:
    """Print a tagged log line."""
    print(f"[PYTHON-INFO] {m}", flush=True)
def read_sheet(p: str) -> list[dict]:
    """Rows of a samples.tsv with a non-empty sample_id."""
    return [r for r in csv.DictReader(open(p), delimiter="\t") if (r.get("sample_id") or "").strip()]
def submitted(root: str, sid: str) -> dict:
    """The console record configs/samples/<sid>.json, if any."""
    p = f"{root}/configs/samples/{sid}.json"
    return json.load(open(p)) if os.path.exists(p) else {}
def validate(samples: list[dict]) -> list[str]:
    """Return human-readable problems of a sample sheet (empty when valid)."""
    seen, errs = set(), []
    if not samples: return ["the sample sheet is empty"]
    for i, s in enumerate(samples, 1):
        sid = (s.get("sample_id") or "").strip()
        if not ID_RE.match(sid): errs.append(f"row {i}: sample_id {sid!r} must be letters, digits or underscore")
        if sid in seen: errs.append(f"row {i}: duplicate sample_id {sid!r}")
        seen.add(sid)
        fa, q = (s.get("fasta_path") or "").strip(), (s.get("uniprot_query") or "").strip()
        if bool(fa) == bool(q): errs.append(f"row {i}: give either a local FASTA path or a UniProtKB query, not both or neither")
        if fa and not (os.path.isabs(fa) and os.path.isfile(fa) and fa.lower().endswith(FASTA_EXT)): errs.append(f"row {i}: fasta_path {fa!r} must be an absolute path to an existing {'/'.join(FASTA_EXT)} file")
        if q and ("\t" in q or "\n" in q): errs.append(f"row {i}: the UniProtKB query must be one line")
    return errs
def validate_params(params: dict, root: str) -> list[str]:
    """Check embed.backends against configs/tools.tsv and embed.layers / embed.pooling against each backend's real layer count."""
    e = params.get("embed") or {}; names = e.get("backends") or []
    known = registry(f"{root}/configs/tools.tsv")
    errs = [] if names else ["param_configs.yaml: embed.backends lists no backend"]
    errs += [f"param_configs.yaml: unknown backend {b!r} (known: {', '.join(known)})" for b in names if b not in known]
    for b in [b for b in names if b in known]:
        try: [poolings(e.get("pooling", ["mean"]), k, known[b]["n_layers"]) for k in resolve_layers(known[b], e.get("layers", [-1]))]
        except ValueError as x: errs.append(f"param_configs.yaml embed: {x}")
    if e.get("store_dtype", "float32") not in ("float32", "float16"): errs.append("param_configs.yaml embed.store_dtype: float32 or float16")
    return errs
def uniprot_file(root: str, sid: str, text: str, reviewed: bool) -> str:
    """Path of the prefetched FASTA of one sample (data/uniprot/<sample>/<key of the exact query>.fasta), shared with src/prefetch.py."""
    return f"{root}/{UNIPROT_DIR}/{sid}/{hashlib.sha256(json.dumps([text.strip(), reviewed]).encode()).hexdigest()[:16]}.fasta"
def sha256(p: str) -> str:
    """sha256 of a file."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def analysis_dir(name: str, root: str = ROOT) -> str:
    """results/<name>, after checking the name."""
    if not NAME_RE.match(name or ""): raise ValueError(f"analysis name {name!r}: use letters, digits, _ . - (max 64)")
    return f"{root}/results/{name}"
def load_yaml(p: str) -> dict:
    """Read a YAML file (PyYAML is needed on the host)."""
    import yaml
    return yaml.safe_load(open(p)) or {}
def create(name: str, root: str = ROOT, src: str | None = None) -> tuple[str, bool, list[str]]:
    """Freeze configs (or an example folder with samples.tsv + param_configs.yaml) into results/<name>/inputs once; an existing analysis is resumed with its frozen inputs.

    Args:
        name (str): Analysis name.
        root (str): Repo root.
        src (str | None): Example folder instead of configs/.

    Returns:
        tuple[str, bool, list[str]]: Analysis dir, whether it was created now, and the validation errors.
    """
    a = analysis_dir(name, root); inp = f"{a}/inputs"
    if os.path.exists(f"{inp}/samples.tsv"): return a, False, []
    base = src or f"{root}/configs"
    meta = (lambda sid: {}) if src else (lambda sid: submitted(root, sid))
    rows = [dict(r, fasta_path=os.path.join(src, r["fasta_path"])) if src and (r.get("fasta_path") or "").strip() and not os.path.isabs(r["fasta_path"]) else r for r in read_sheet(f"{base}/samples.tsv")]
    P = load_yaml(f"{base}/param_configs.yaml"); reviewed = bool((P.get("prefetch") or {}).get("reviewed_only", False))
    errs = validate([{**meta(r["sample_id"]), **r} for r in rows]) + validate_params(P, root)
    up = {r["sample_id"]: uniprot_file(root, r["sample_id"], r["uniprot_query"], reviewed) for r in rows if (r.get("uniprot_query") or "").strip()}
    errs += [f"{sid}: UniProt query not downloaded yet: click Run in the console, or run python3 src/prefetch.py{' --samples ' + os.path.relpath(base, root) + '/samples.tsv --yaml-in ' + os.path.relpath(base, root) + '/param_configs.yaml' if src else ''} on a login node" for sid, p in up.items() if not (os.path.isfile(p) and os.path.isfile(p[:-6] + ".json"))]
    if errs: return a, False, errs
    tmp = f"{inp}.tmp"; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(f"{tmp}/samples")
    cols = list(dict.fromkeys(COLUMNS + [c for r in rows for c in r]))
    open(f"{tmp}/samples.tsv", "w").write("\t".join(cols) + "\n" + "".join("\t".join(str(r.get(c) or "") for c in cols) + "\n" for r in rows))
    shutil.copy(f"{base}/param_configs.yaml", f"{tmp}/params.yaml"); shutil.copy(f"{root}/configs/tools.tsv", f"{tmp}/tools.tsv")
    if up: os.makedirs(f"{tmp}/uniprot")
    for sid, p in up.items(): shutil.copy(p, f"{tmp}/uniprot/{sid}.fasta"); shutil.copy(p[:-6] + ".json", f"{tmp}/uniprot/{sid}.json")
    for r in rows if not src else []:
        if os.path.exists(f"{root}/configs/samples/{r['sample_id']}.json"): shutil.copy(f"{root}/configs/samples/{r['sample_id']}.json", f"{tmp}/samples/")
    sums = {os.path.relpath(f"{d}/{f}", tmp): sha256(f"{d}/{f}") for d, _, fs in os.walk(tmp) for f in fs}
    local = {p: dict(size=os.stat(p).st_size, mtime=os.stat(p).st_mtime, sha256=sha256(p)) for p in sorted({(r.get("fasta_path") or meta(r["sample_id"]).get("fasta_path") or "").strip() for r in rows} - {""})}
    info = dict(analysis=name, created_at=f"{datetime.datetime.now():%F %T}", samples=[r["sample_id"] for r in rows], checksums=dict(sorted(sums.items())), local_fasta=local)
    if src: info["source"] = os.path.relpath(src, root)
    json.dump(info, open(f"{tmp}/analysis.json", "w"), indent=1)
    os.replace(tmp, inp)
    return a, True, []
###################################### MAIN
def main() -> None:
    """Create or resume the analysis named on the command line."""
    t0 = time.time()
    ap = argparse.ArgumentParser(description="Create or resume an analysis"); ap.add_argument("analysis_name"); ap.add_argument("--example", default=None, help="configs/examples/<name> instead of configs/")
    a = ap.parse_args()
    try: d, new, errs = create(a.analysis_name, ROOT, f"{ROOT}/configs/examples/{a.example}" if a.example else None)
    except ValueError as e: sys.exit(f"[PYTHON-INFO] !! {e}")
    if errs:
        for e in errs: log(f"!! {e}")
        sys.exit(1)
    log(f"{'created' if new else 'resuming'} {os.path.relpath(d, ROOT)} (inputs frozen in {os.path.relpath(d, ROOT)}/inputs); run: python3 src/pipeline.py {a.analysis_name}")
    log(f"{datetime.datetime.now():%F %T} | elapsed {time.time() - t0:.1f} s | max RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576:.2f} GB")
if __name__ == "__main__":
    main()
