#!/usr/bin/env python3
"""t-SNE and UMAP of small point sets (never subsampled), run in the dadapy venv: a manifest of .npy point sets in, one JSON of 2D coordinates out."""
###################################### DEPENDENCIES_AND_CONSTANTS
import argparse
import datetime
import importlib.metadata
import json
from pathlib import Path
import resource
import time
import numpy as np
import yaml
SEED = 42
YAML_PATH = "configs/param_configs.yaml"
SCRIPT_KEY = "embed2d"
MAX_POINTS = 1000
MIN_POINTS = 10
PERPLEXITY = 30.0
N_NEIGHBORS = 15
###################################### CONFIG_AND_PARSER
parser = argparse.ArgumentParser(description="t-SNE and UMAP of small point sets")
parser.add_argument("--manifest", type=str, default=None, help="JSON: tasks [{name, path}] of .npy arrays")
parser.add_argument("--out", type=str, default=None, help="JSON results")
parser.add_argument("--max_points", type=int, default=MAX_POINTS, help="Refuse sets of this many points or more (no subsampling)")
parser.add_argument("--methods", type=str, default="tsne,umap", help="Comma list of tsne, umap")
parser.add_argument("--perplexity", type=float, default=PERPLEXITY)
parser.add_argument("--n_neighbors", type=int, default=N_NEIGHBORS)
parser.add_argument("--yaml-in", nargs="?", const=YAML_PATH, default=None)
parser.add_argument("--yaml-out", nargs="?", const=YAML_PATH, default=None)
cli_preview, _ = parser.parse_known_args()
params = dict(max_points=MAX_POINTS, methods="tsne,umap", perplexity=PERPLEXITY, n_neighbors=N_NEIGHBORS)
if cli_preview.yaml_in and Path(cli_preview.yaml_in).exists():
    sec = (yaml.safe_load(Path(cli_preview.yaml_in).read_text()) or {}).get(SCRIPT_KEY) or {}
    params.update({k: sec[k] for k in sec if k in params})
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
def tsne(X: np.ndarray, perplexity: float) -> np.ndarray:
    """scikit-learn t-SNE with PCA initialisation and the perplexity clamped below the point count."""
    from sklearn.manifold import TSNE
    return TSNE(n_components=2, perplexity=min(perplexity, (len(X) - 1) / 3), init="pca", random_state=SEED).fit_transform(X)
def umap_2d(X: np.ndarray, n_neighbors: int) -> np.ndarray:
    """UMAP with the neighbour count clamped below the point count."""
    import umap
    return umap.UMAP(n_components=2, n_neighbors=min(n_neighbors, len(X) - 1), random_state=SEED).fit_transform(X)
def umap_ok() -> bool:
    """Whether umap-learn imports in this environment."""
    try:
        import umap
        return True
    except Exception:
        return False
def rounded(Y: np.ndarray) -> list[list[float]]:
    """Coordinates rounded for the aggregates file."""
    return np.asarray(Y, dtype=np.float64).round(3).tolist()
###################################### MAIN
def main() -> None:
    """Embed every task of the manifest below the point limit and write the results."""
    t0 = time.time()
    if not args.manifest or not args.out:
        if args.yaml_out: return
        parser.error("--manifest and --out are required")
    m, out, want = json.loads(Path(args.manifest).read_text()), [], set(args.methods.split(","))
    if "umap" in want and not umap_ok(): want.discard("umap"); log("!! umap-learn is missing or broken in this image (rebuild and pull the image of this commit): UMAP skipped, t-SNE continues")
    log(f"{','.join(sorted(want))}, scikit-learn {importlib.metadata.version('scikit-learn')}, seed {SEED}")
    for t in m["tasks"]:
        X = np.asarray(np.load(t["path"]), dtype=np.float64)
        if len(X) >= args.max_points or len(X) < MIN_POINTS: out.append(dict(name=t["name"], points=len(X), skipped=f"n >= {args.max_points}" if len(X) >= args.max_points else f"n < {MIN_POINTS}")); continue
        out.append(dict(name=t["name"], points=len(X), **({"tsne": rounded(tsne(X, args.perplexity))} if "tsne" in want else {}), **({"umap": rounded(umap_2d(X, args.n_neighbors))} if "umap" in want else {})))
    tmp = Path(f"{args.out}.tmp"); tmp.write_text(json.dumps(out)); tmp.replace(args.out)
    log(f"{len(out)} point set(s) -> {args.out}")
    log(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} | elapsed {time.time() - t0:.1f} s | max RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576:.2f} GB")
if __name__ == "__main__":
    main()
