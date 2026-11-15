#!/usr/bin/env python3
"""Embed one shard of a validated FASTA with one backend into atomic safetensors chunk files, resuming from the last complete chunk."""
###################################### DEPENDENCIES_AND_CONSTANTS
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import queue
import random
import resource
import shutil
import sys
import threading
import time
import numpy as np
from tqdm import tqdm
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parent))
from backends import batches, load_backend, model_dir, poolings, registry, resolve_layers, shard_batches, verify
import store
SEED = 42
YAML_PATH = "configs/param_configs.yaml"
SCRIPT_KEY = "embed"
MODELS_DIR = "data/models"
BATCH_TOKENS = 4096
MAX_LENGTH = 0
DTYPE = "auto"
LAYERS = [-1]
POOLING = ["mean"]
STORE_DTYPE = "float32"
FLUSH_PROTEINS = 2000
FLUSH_MB = 256
ALLOW_CPU = False
VERIFY_WEIGHTS = "full"
QUEUE_DEPTH = 4
random.seed(SEED)
np.random.seed(SEED)
###################################### CONFIG_AND_PARSER
parser = argparse.ArgumentParser(description="Embed one shard of a FASTA with one backend into safetensors chunks")
parser.add_argument("--backend", type=str, default=None, help="Backend name of configs/tools.tsv")
parser.add_argument("--fasta", type=str, default=None, help="Validated single-line FASTA (01_prepare output)")
parser.add_argument("--shard", type=int, default=1, help="Shard number, 1-based")
parser.add_argument("--shards", type=int, default=1, help="Number of shards of this sample")
parser.add_argument("--chunk_dir", type=str, default=None, help="Folder of this shard's chunk files")
parser.add_argument("--done", type=str, default=None, help="Done marker (JSON) written when the shard is complete")
parser.add_argument("--models_dir", type=str, default=MODELS_DIR)
parser.add_argument("--batch_tokens", type=int, default=BATCH_TOKENS, help="Padded tokens per batch (longest sequence x batch size)")
parser.add_argument("--max_length", type=int, default=MAX_LENGTH, help="Truncate longer sequences (0: backend limit only)")
parser.add_argument("--dtype", type=str, default=DTYPE, choices=["auto", "fp32", "fp16", "bf16"], help="Compute precision")
parser.add_argument("--layers", type=yaml.safe_load, default=LAYERS, help="List of layers (1..n or -1..-n) or 'all'")
parser.add_argument("--pooling", type=yaml.safe_load, default=POOLING, help="mean, tokens or both; a list, or a mapping layer -> list")
parser.add_argument("--store_dtype", type=str, default=STORE_DTYPE, choices=list(store.DTYPES))
parser.add_argument("--flush_proteins", type=int, default=FLUSH_PROTEINS, help="Write a chunk after this many proteins")
parser.add_argument("--flush_mb", type=int, default=FLUSH_MB, help="... or after this many MB of vectors")
parser.add_argument("--allow_cpu", action=argparse.BooleanOptionalAction, default=ALLOW_CPU, help="Run on CPU when no GPU is visible")
parser.add_argument("--verify_weights", type=str, default=VERIFY_WEIGHTS, choices=["full", "list"], help="full: sha256 of every weight file; list: file names only")
parser.add_argument("--yaml-in", nargs="?", const=YAML_PATH, default=YAML_PATH)
parser.add_argument("--yaml-out", nargs="?", const=YAML_PATH, default=None)
cli_preview, _ = parser.parse_known_args()
params = dict(models_dir=MODELS_DIR, batch_tokens=BATCH_TOKENS, max_length=MAX_LENGTH, dtype=DTYPE, layers=LAYERS, pooling=POOLING, store_dtype=STORE_DTYPE, flush_proteins=FLUSH_PROTEINS, flush_mb=FLUSH_MB, allow_cpu=ALLOW_CPU, verify_weights=VERIFY_WEIGHTS)
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
def read_fasta(path: str) -> list[tuple[str, str]]:
    """Read a FASTA into (id, sequence), ids cut at the first whitespace."""
    records, sid, parts = [], None, []
    for line in open(path):
        line = line.strip()
        if not line: continue
        if line.startswith(">"):
            if sid is not None: records.append((sid, "".join(parts)))
            sid, parts = line[1:].split()[0], []
        else: parts.append(line)
    if sid is not None: records.append((sid, "".join(parts)))
    return records
def pick_device(family: str) -> object:
    """CUDA device when visible; CPU only with allow_cpu (the fake test backend always uses CPU)."""
    if family == "fake": return "cpu"
    import torch
    torch.manual_seed(SEED)
    if not torch.cuda.is_available() and not args.allow_cpu: sys.exit("[PYTHON-INFO] !! no GPU visible (singularity --nv on a GPU node?) and allow_cpu is off")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
def settings(spec: dict, input_sha: str, layers: list[int], pools: dict, compute_dtype: str) -> dict:
    """Everything that changes the bytes of a shard, including the precision actually used; a chunk folder written with other settings is discarded."""
    return dict(backend=spec["name"], revision=spec["commit"], input_sha256=input_sha, shard=args.shard, shards=args.shards, layers=layers, pooling={str(k): v for k, v in pools.items()}, store_dtype=args.store_dtype, max_length=args.max_length, batch_tokens=args.batch_tokens, dtype=args.dtype, compute_dtype=compute_dtype)
def put(q: queue.Queue, item: object, state: dict) -> bool:
    """Hand an item to the writer without blocking forever: gives up (False) once the writer has failed."""
    while not state["error"]:
        try: q.put(item, timeout=1); return True
        except queue.Full: continue
    return False
def reader(groups: list[list[int]], seqs: list[str], q: queue.Queue) -> None:
    """Producer thread: hand ready-made batches of sequences to the GPU loop."""
    for b, g in enumerate(groups): q.put((b, g, [seqs[i] for i in g]))
    q.put(None)
def writer(d: Path, j: int, q: queue.Queue, ids: list[str], lens: list[int], full: list[int], state: dict) -> None:
    """Single writer thread: buffer batch results and write an atomic chunk every flush_proteins proteins or flush_mb MB."""
    buf, nbytes, first = [], 0, None
    def flush(hi: int) -> None:
        """Write the buffered batches as one chunk."""
        nonlocal buf, nbytes, first, j
        rows = [i for _, idx, _ in buf for i in idx]
        tensors = dict(lengths=np.array([full[i] for i in rows], dtype=np.int32), truncated=np.array([lens[i] < full[i] for i in rows], dtype=np.uint8))
        for k in buf[0][2]:
            if k.endswith("/mean"): tensors[k] = np.concatenate([r[k] for _, _, r in buf])
            else: tensors[k] = np.concatenate([m for _, _, r in buf for m in r[k]])
        if any(k.endswith("/tokens") for k in buf[0][2]): tensors["token_offsets"] = np.concatenate([[0], np.cumsum([lens[i] for i in rows])]).astype(np.int64)
        store.write_chunk(d, j, tensors, [ids[i] for i in rows], first, hi)
        j, buf, nbytes, first = j + 1, [], 0, None
        state["chunks"] += 1
    try:
        while (item := q.get()) is not None:
            b, idx, res = item
            first = b if first is None else first
            buf.append((b, idx, res)); nbytes += sum(v.nbytes if isinstance(v, np.ndarray) else sum(m.nbytes for m in v) for v in res.values())
            if sum(len(x[1]) for x in buf) >= args.flush_proteins or nbytes >= args.flush_mb << 20: flush(b + 1)
        if buf: flush(buf[-1][0] + 1)
    except Exception as e: state["error"] = e
###################################### MAIN
def main() -> None:
    """Verify the weights, embed the shard's batches not yet stored, and write the done marker."""
    t0 = time.time()
    if not (args.backend and args.fasta and args.chunk_dir and args.done):
        if args.yaml_out: return
        parser.error("--backend, --fasta, --chunk_dir and --done are required")
    spec = registry()[args.backend]
    rec, problems = verify(model_dir(Path(args.models_dir), spec["name"]), spec, full=args.verify_weights == "full")
    if problems: sys.exit("[PYTHON-INFO] !! weights check failed, nothing embedded:\n  " + "\n  ".join(problems))
    log(f"weights of {spec['name']} @ {spec['commit'][:12]}: {len(rec)} file(s) match CHECKSUMS ({args.verify_weights})")
    layers = resolve_layers(spec, args.layers)
    pools = {k: poolings(args.pooling, k, spec["n_layers"]) for k in layers}
    records, input_sha = read_fasta(args.fasta), hashlib.sha256(Path(args.fasta).read_bytes()).hexdigest()
    device = pick_device(spec["family"])
    be = load_backend(spec["name"], Path(args.models_dir), device, args.dtype)
    if (be.n_layers, be.dim) != (spec["n_layers"], spec["dim"]): sys.exit(f"[PYTHON-INFO] !! {spec['name']} loaded with {be.n_layers} layers x {be.dim}, configs/tools.tsv says {spec['n_layers']} x {spec['dim']}: fix the pins table")
    d, cfg = Path(args.chunk_dir), settings(spec, input_sha, layers, pools, be.dtype)
    if (d / "shard.json").is_file() and json.loads((d / "shard.json").read_text()) != cfg: shutil.rmtree(d); log(f"discarded {d}: written with other inputs or settings")
    d.mkdir(parents=True, exist_ok=True); (d / "shard.json").write_text(json.dumps(cfg, indent=1))
    limit = min([x for x in (be.max_length, args.max_length) if x] or [0])
    seqs = [s[:limit] if limit else s for _, s in records]
    lens, full = [len(s) for s in seqs], [len(s) for _, s in records]
    order = sorted(range(len(seqs)), key=lambda i: (lens[i], i))
    mine = shard_batches(batches(order, lens, args.batch_tokens, be.one_at_a_time), args.shard, args.shards)
    j, b0 = store.resume_batch(d)
    if b0: log(f"resuming shard {args.shard}/{args.shards}: {b0}/{len(mine)} batches already in {j} chunk(s)")
    np_dt = store.DTYPES[args.store_dtype][0]
    qin, qout, state = queue.Queue(QUEUE_DEPTH), queue.Queue(QUEUE_DEPTH), dict(chunks=j, error=None)
    tr = threading.Thread(target=reader, args=(mine[b0:], seqs, qin), daemon=True)
    tw = threading.Thread(target=writer, args=(d, j, qout, [r[0] for r in records], lens, full, state), daemon=True)
    tr.start(); tw.start(); done_proteins = sum(len(g) for g in mine[:b0])
    try:
        with tqdm(total=sum(len(g) for g in mine), initial=done_proteins, desc=f"{spec['name']} shard {args.shard}/{args.shards}", mininterval=30) as bar:
            while (item := qin.get()) is not None:
                b, idx, batch = item
                res = be.residues(batch, layers); out = {}
                for k in layers:
                    if "mean" in pools[k]: out[store.key(k, "mean")] = np.stack([m.mean(0) for m in res[k]]).astype(np_dt)
                    if "tokens" in pools[k]: out[store.key(k, "tokens")] = [m.astype(np_dt) for m in res[k]]
                bad = sum(int((~np.isfinite(a)).sum()) for v in out.values() for a in (v if isinstance(v, list) else [v]))
                if bad: raise RuntimeError(f"{bad} non-finite value(s) in batch {b + b0} ({args.store_dtype}): nothing of this batch was stored")
                if not put(qout, (b + b0, idx, out), state): break
                bar.update(len(idx))
    finally:
        put(qout, None, state); tw.join()
    if state["error"]: raise state["error"]
    n = sum(chunk["rows"] for chunk in map(store.chunk_meta, store.chunks(d)))
    gpu = None
    if spec["family"] != "fake" and getattr(device, "type", "") == "cuda":
        import torch
        gpu = dict(name=torch.cuda.get_device_name(0), peak_gb=round(torch.cuda.max_memory_allocated() / 1073741824, 2))
    marker = dict(cfg, proteins=n, batches=len(mine), chunks=state["chunks"], dim=be.dim, compute_dtype=be.dtype, truncated=sum(lens[i] < full[i] for g in mine for i in g), weights_sha256=rec, gpu=gpu, elapsed_s=round(time.time() - t0, 1), finished=f"{datetime.datetime.now():%F %T}")
    tmp = Path(f"{args.done}.tmp"); tmp.write_text(json.dumps(marker, indent=1)); tmp.replace(args.done)
    log(f"{spec['name']} shard {args.shard}/{args.shards}: {n:,} proteins in {state['chunks']} chunk(s), layers {layers}, {pools[layers[-1]]}, {args.store_dtype}" + (f", gpu peak {gpu['peak_gb']} GB" if gpu else ""))
    log(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} | elapsed {time.time() - t0:.1f} s | max RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576:.2f} GB")
if __name__ == "__main__":
    main()
