#!/usr/bin/env python3
"""Validate the proteins of one sample (a local FASTA or a frozen UniProt prefetch) into a single-line FASTA of valid records and a per-record status table."""
###################################### DEPENDENCIES_AND_CONSTANTS
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
from tqdm import tqdm
import yaml
YAML_PATH = "configs/param_configs.yaml"
SCRIPT_KEY = "prepare"
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYXBZUO")
MIN_LENGTH = 1
MAX_LENGTH = 0
###################################### CONFIG_AND_PARSER
parser = argparse.ArgumentParser(description="Validate the proteins of one sample")
parser.add_argument("--sample", type=str, default=None)
parser.add_argument("--source", type=str, default=None, help="FASTA: local path or results/<a>/inputs/uniprot/<sample>.fasta")
parser.add_argument("--out_fasta", type=str, default=None)
parser.add_argument("--out_tsv", type=str, default=None)
parser.add_argument("--out_json", type=str, default=None, help="Summary for the sample metadata")
parser.add_argument("--min_length", type=int, default=MIN_LENGTH)
parser.add_argument("--max_length", type=int, default=MAX_LENGTH, help="0: no upper bound")
parser.add_argument("--yaml-in", nargs="?", const=YAML_PATH, default=YAML_PATH)
parser.add_argument("--yaml-out", nargs="?", const=YAML_PATH, default=None)
cli_preview, _ = parser.parse_known_args()
params = dict(min_length=MIN_LENGTH, max_length=MAX_LENGTH)
if cli_preview.yaml_in and Path(cli_preview.yaml_in).exists():
    config_data = yaml.safe_load(Path(cli_preview.yaml_in).read_text()) or {}
    params.update({k: v for k, v in (config_data.get(SCRIPT_KEY) or {}).items() if k in params})
parser.set_defaults(**params)
args = parser.parse_args()
if args.yaml_out:
    config_data = (yaml.safe_load(Path(args.yaml_out).read_text()) or {}) if Path(args.yaml_out).exists() else {}
    config_data[SCRIPT_KEY] = {k: getattr(args, k) for k in params}
    Path(args.yaml_out).write_text(yaml.safe_dump(config_data, sort_keys=False))
###################################### HELPER_FUNCTIONS
def log(msg: str) -> None:
    """Print a tagged log line."""
    print(f"[PYTHON-INFO] {msg}", flush=True)
def records(path: Path) -> list[tuple[str, str, str]]:
    """Read a FASTA into (id, header, sequence), wrapped records allowed."""
    out, head, parts = [], None, []
    def push() -> None:
        """Store the record being read."""
        if head is not None: out.append((head.split()[0] if head.split() else "", head, "".join(parts)))
    for line in open(path):
        line = line.rstrip("\n\r")
        if line.startswith(">"): push(); head, parts = line[1:].strip(), []
        elif line.strip(): parts.append(line.strip())
    push()
    return out
def status(sid: str, seq: str, seen: set) -> str:
    """Return kept, or the reason a record is invalid."""
    if not sid: return "invalid:no_id"
    if sid in seen: return "invalid:duplicate_id"
    if not seq: return "invalid:empty"
    if set(seq) - AMINO_ACIDS: return "invalid:letters"
    if len(seq) < args.min_length or (args.max_length and len(seq) > args.max_length): return "invalid:length"
    return "kept"
###################################### MAIN
def main() -> None:
    """Validate and write the sample files (FASTA, status table, JSON summary with the UniProt provenance when present)."""
    t0 = time.time()
    if not (args.sample and args.source and args.out_fasta and args.out_tsv):
        if args.yaml_out: return
        parser.error("--sample, --source, --out_fasta and --out_tsv are required")
    out, src = Path(args.out_fasta), Path(args.source)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen, kept, rows = set(), [], []
    for sid, head, seq in tqdm(records(src), desc=f"{args.sample} records", mininterval=30):
        seq = seq.upper().rstrip("*"); st = status(sid, seq, seen); seen.add(sid)
        rows.append((sid, len(seq), st))
        if st == "kept": kept.append(f">{head}\n{seq}\n")
    if not kept: sys.exit(f"[PYTHON-INFO] !! {args.sample}: no valid protein in {src}")
    for path, text in ((out, "".join(kept)), (Path(args.out_tsv), "record\tlength\tstatus\n" + "".join(f"{a}\t{b}\t{c}\n" for a, b, c in rows))):
        tmp = path.with_name(path.name + ".tmp"); tmp.write_text(text); tmp.replace(path)
    counts = {s: sum(r[2] == s for r in rows) for s in sorted({r[2] for r in rows})}
    side = src.with_suffix(".json")
    summary = dict(source=str(src), source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(), uniprot=json.loads(side.read_text()) if side.is_file() else None, records=len(rows), kept=len(kept), status=counts,
                   fasta_sha256=hashlib.sha256(out.read_bytes()).hexdigest(), min_length=args.min_length, max_length=args.max_length)
    if args.out_json: Path(args.out_json).write_text(json.dumps(summary, indent=1))
    log(f"{args.sample}: {len(kept):,}/{len(rows):,} proteins kept ({', '.join(f'{k} {v}' for k, v in counts.items())}) -> {out}")
    log(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} | elapsed {time.time() - t0:.1f} s | max RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576:.2f} GB")
if __name__ == "__main__":
    main()
