#!/usr/bin/env python3
"""Optional MMseqs2 sequence deduplication before embedding, per sample or across samples: representatives keep their original headers and a cluster -> member table keeps every protein traceable."""
###################################### DEPENDENCIES_AND_CONSTANTS
import argparse
import datetime
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time
from tqdm import tqdm
import yaml
YAML_PATH = "configs/param_configs.yaml"
SCRIPT_KEY = "dedup"
MODE = "easy-linclust"
MIN_SEQ_ID = 0.95
COVERAGE = 0.8
COV_MODE = 0
MMSEQS = "mmseqs"
MODES = ("easy-linclust", "easy-cluster")
SEP = "|"
###################################### CONFIG_AND_PARSER
parser = argparse.ArgumentParser(description="MMseqs2 deduplication of prepared FASTA files")
parser.add_argument("--scope", type=str, default="sample", choices=["sample", "joint"], help="Cluster each sample alone, or all samples together")
parser.add_argument("--fasta", type=str, nargs="+", default=None, help="Prepared FASTA files, one per sample (results/<a>/fasta/<sample>.fasta)")
parser.add_argument("--out_dir", type=str, default=None, help="results/<a>/dedup")
parser.add_argument("--threads", type=int, default=1, help="mmseqs --threads (the job's cores)")
parser.add_argument("--mode", type=str, default=MODE, choices=MODES)
parser.add_argument("--min_seq_id", type=float, default=MIN_SEQ_ID)
parser.add_argument("--coverage", type=float, default=COVERAGE)
parser.add_argument("--cov_mode", type=int, default=COV_MODE)
parser.add_argument("--mmseqs", type=str, default=MMSEQS, help="mmseqs executable")
parser.add_argument("--yaml-in", nargs="?", const=YAML_PATH, default=YAML_PATH)
parser.add_argument("--yaml-out", nargs="?", const=YAML_PATH, default=None)
cli_preview, _ = parser.parse_known_args()
params = dict(mode=MODE, min_seq_id=MIN_SEQ_ID, coverage=COVERAGE, cov_mode=COV_MODE, mmseqs=MMSEQS)
if cli_preview.yaml_in and Path(cli_preview.yaml_in).exists():
    config_data = yaml.safe_load(Path(cli_preview.yaml_in).read_text()) or {}
    params.update({k: v for k, v in (config_data.get(SCRIPT_KEY) or {}).items() if k in params})
parser.set_defaults(**params)
args = parser.parse_args()
if args.yaml_out:
    config_data = (yaml.safe_load(Path(args.yaml_out).read_text()) or {}) if Path(args.yaml_out).exists() else {}
    config_data[SCRIPT_KEY] = {**(config_data.get(SCRIPT_KEY) or {}), **{k: getattr(args, k) for k in params}}
    Path(args.yaml_out).write_text(yaml.safe_dump(config_data, sort_keys=False))
###################################### HELPER_FUNCTIONS
def log(msg: str) -> None:
    """Print a tagged log line."""
    print(f"[PYTHON-INFO] {msg}", flush=True)
def read(path: Path) -> list[tuple[str, str]]:
    """Single-line FASTA (01_prepare output) into (header, sequence)."""
    lines = path.read_text().splitlines()
    return [(lines[i][1:], lines[i + 1]) for i in range(0, len(lines) - 1, 2)]
def cluster(fasta: Path, work: Path) -> list[tuple[str, str]]:
    """Run mmseqs on one FASTA and return (representative, member) pairs."""
    cmd = [args.mmseqs, args.mode, str(fasta), str(work / "out"), str(work / "tmp"), "--min-seq-id", str(args.min_seq_id), "-c", str(args.coverage), "--cov-mode", str(args.cov_mode), "--threads", str(args.threads)]
    log(" ".join(cmd))
    subprocess.run(cmd, check=True, stdout=sys.stdout, stderr=sys.stderr)
    return [tuple(l.split("\t")[:2]) for l in (work / "out_cluster.tsv").read_text().splitlines() if l.strip()]
def write(path: Path, text: str) -> None:
    """Atomic text write."""
    tmp = path.with_name(path.name + ".tmp"); tmp.write_text(text); os.replace(tmp, path)
###################################### MAIN
def main() -> None:
    """Cluster, then write per-sample representative FASTA, cluster table and summary."""
    t0 = time.time()
    if not (args.fasta and args.out_dir):
        if args.yaml_out: return
        parser.error("--fasta and --out_dir are required")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    version = subprocess.run([args.mmseqs, "version"], capture_output=True, text=True, check=True).stdout.strip()
    data = {Path(f).stem: read(Path(f)) for f in args.fasta}
    groups = [[s] for s in data] if args.scope == "sample" else [list(data)]
    for g in groups:
        with tempfile.TemporaryDirectory(dir=out) as tmp:
            work = Path(tmp); joint = work / "in.fasta"
            joint.write_text("".join(f">{s}{SEP}{h.split()[0]}\n{q}\n" for s in g for h, q in data[s]))
            pairs = cluster(joint, work)
        members = {}
        for rep, mem in tqdm(pairs, desc="cluster pairs", mininterval=30): members.setdefault(rep, []).append(mem)
        seen = {m for ms in members.values() for m in ms}
        missing = [f"{s}{SEP}{h.split()[0]}" for s in g for h, _ in data[s] if f"{s}{SEP}{h.split()[0]}" not in seen]
        if missing: sys.exit(f"[PYTHON-INFO] !! mmseqs lost {len(missing)} protein(s), e.g. {missing[:3]}")
        for s in tqdm(g, desc="samples", mininterval=30):
            reps = {r.split(SEP, 1)[1] for r in members if r.split(SEP, 1)[0] == s}
            kept = [(h, q) for h, q in data[s] if h.split()[0] in reps]
            write(out / f"{s}.fasta", "".join(f">{h}\n{q}\n" for h, q in kept))
            rows = [(r.split(SEP, 1)[1], m.split(SEP, 1)[0], m.split(SEP, 1)[1]) for r in members if r.split(SEP, 1)[0] == s for m in members[r]]
            write(out / f"{s}.clusters.tsv", "representative\tmember_sample\tmember\n" + "".join(f"{a}\t{b}\t{c}\n" for a, b, c in rows))
            elsewhere = sum(1 for r, ms in members.items() if r.split(SEP, 1)[0] != s for m in ms if m.split(SEP, 1)[0] == s)
            info = dict(sample=s, scope=args.scope, proteins=len(data[s]), representatives=len(kept), represented_in_other_samples=elsewhere, mode=args.mode, min_seq_id=args.min_seq_id, coverage=args.coverage, cov_mode=args.cov_mode, mmseqs_version=version, threads=args.threads, finished=f"{datetime.datetime.now():%F %T}")
            write(out / f"{s}.dedup.json", json.dumps(info, indent=1))
            log(f"{s}: {len(data[s]):,} -> {len(kept):,} representatives ({args.mode}, id {args.min_seq_id}, cov {args.coverage}, scope {args.scope}{f', {elsewhere} represented by another sample' if elsewhere else ''})")
    log(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} | elapsed {time.time() - t0:.1f} s | max RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576:.2f} GB")
if __name__ == "__main__":
    main()
