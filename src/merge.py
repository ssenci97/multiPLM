#!/usr/bin/env python3
"""Merge the shard chunks of one (sample, backend) into the final safetensors file(s) with an index, an id -> row sidecar and a provenance record, streaming one chunk at a time."""
###################################### DEPENDENCIES_AND_CONSTANTS
import argparse
import datetime
import json
from pathlib import Path
import resource
import sys
import time
from tqdm import tqdm
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parent))
import store
YAML_PATH = "configs/param_configs.yaml"
SCRIPT_KEY = "merge"
MAX_PART_GB = 20.0
KEEP_CHUNKS = False
SAME_KEYS = ("backend", "revision", "input_sha256", "layers", "pooling", "store_dtype", "compute_dtype", "max_length", "weights_sha256")
###################################### CONFIG_AND_PARSER
parser = argparse.ArgumentParser(description="Merge shard chunks into the final safetensors embeddings of one sample and backend")
parser.add_argument("--sample", type=str, default=None)
parser.add_argument("--shards", type=int, default=1)
parser.add_argument("--shard_root", type=str, default=None, help="results/<a>/embeddings/<backend>/<sample>.shards")
parser.add_argument("--out_dir", type=str, default=None, help="results/<a>/embeddings/<backend>")
parser.add_argument("--max_part_gb", type=float, default=MAX_PART_GB, help="Split the final file into row parts above this size")
parser.add_argument("--keep_chunks", action=argparse.BooleanOptionalAction, default=KEEP_CHUNKS, help="Keep the chunk files after a successful merge")
parser.add_argument("--yaml-in", nargs="?", const=YAML_PATH, default=YAML_PATH)
parser.add_argument("--yaml-out", nargs="?", const=YAML_PATH, default=None)
cli_preview, _ = parser.parse_known_args()
params = dict(max_part_gb=MAX_PART_GB, keep_chunks=KEEP_CHUNKS)
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
def markers(root: Path, shards: int) -> list[dict]:
    """Done markers of every shard; all must exist and agree on input, backend, revision and settings."""
    out = []
    for k in range(1, shards + 1):
        p = root / f"shard_{k}.done.json"
        if not p.is_file(): sys.exit(f"[PYTHON-INFO] !! shard {k}/{shards} has no done marker {p}")
        out.append(json.loads(p.read_text()))
    same = {json.dumps({k: v for k, v in m.items() if k in SAME_KEYS}, sort_keys=True) for m in out}
    if len(same) != 1: sys.exit("[PYTHON-INFO] !! shards were embedded with different inputs or settings: remove the .shards folder and rerun")
    return out
###################################### MAIN
def main() -> None:
    """Merge, write the provenance record and drop the chunk files."""
    t0 = time.time()
    if not (args.sample and args.shard_root and args.out_dir):
        if args.yaml_out: return
        parser.error("--sample, --shard_root and --out_dir are required")
    root, out = Path(args.shard_root), Path(args.out_dir)
    ms = markers(root, args.shards)
    files = [c for k in range(1, args.shards + 1) for c in store.chunks(root / f"shard_{k}")]
    if sum(store.chunk_meta(c)["rows"] for c in tqdm(files, desc="chunk headers", mininterval=30)) != sum(m["proteins"] for m in ms): sys.exit(f"[PYTHON-INFO] !! chunk files of {root} do not match the shard markers (deleted after an earlier merge?): remove {root} and rerun the embed jobs")
    m0 = ms[0]
    meta = dict(sample=args.sample, backend=m0["backend"], revision=m0["revision"], input_sha256=m0["input_sha256"], layers=json.dumps(m0["layers"]), pooling=json.dumps(m0["pooling"]), store_dtype=m0["store_dtype"], compute_dtype=m0["compute_dtype"], max_length=m0["max_length"], mean="masked mean over real residues (special tokens and padding excluded)")
    index = store.merge(files, out, args.sample, int(args.max_part_gb * 1e9), meta)
    prov = dict(meta, layers=m0["layers"], pooling=m0["pooling"], n=index["n"], keys=index["keys"], parts=index["parts"], weights_sha256=m0["weights_sha256"], truncated=sum(m["truncated"] for m in ms), dim=m0["dim"],
                shards=[dict(shard=m["shard"], proteins=m["proteins"], chunks=m["chunks"], elapsed_s=m["elapsed_s"], gpu=m["gpu"], finished=m["finished"]) for m in ms], merged=f"{datetime.datetime.now():%F %T}")
    tmp = out / f".{args.sample}.json.tmp"; tmp.write_text(json.dumps(prov, indent=1)); tmp.replace(out / f"{args.sample}.json")
    if not args.keep_chunks:
        for c in files: c.unlink()
    log(f"{args.sample} {m0['backend']}: {index['n']:,} proteins, keys {', '.join(index['keys'])}, {len(index['parts'])} file(s) in {out}" + ("" if args.keep_chunks else f"; {len(files)} chunk file(s) removed"))
    log(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} | elapsed {time.time() - t0:.1f} s | max RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576:.2f} GB")
if __name__ == "__main__":
    main()
