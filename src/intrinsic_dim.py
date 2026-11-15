#!/usr/bin/env python3
"""TWO-NN intrinsic dimension with DADApy (see REFERENCES.md), run in the dadapy venv: a manifest of .npy point sets in, one JSON of estimates out."""
###################################### DEPENDENCIES_AND_CONSTANTS
import argparse
import datetime
import importlib.metadata
import json
from pathlib import Path
import resource
import time
import numpy as np
from dadapy import Data
from tqdm import tqdm
SEED = 42
MIN_POINTS = 20
DUPLICATE_WARN = 0.1
DIFFERENCES = "DADApy: least-squares fit through the origin (scipy curve_fit) on the lowest mu_fraction of mu with full-N ranks, as in the paper; scale analysis on random (not disjoint) subsets of N/2, N/4, ... down to n_min, error = standard error of the mean; exact duplicate rows removed and counted here before the estimate"
###################################### CONFIG_AND_PARSER
parser = argparse.ArgumentParser(description="TWO-NN estimates of point sets with DADApy")
parser.add_argument("--manifest", type=str, required=True, help="JSON: settings and tasks [{name, path}] of .npy arrays")
parser.add_argument("--out", type=str, required=True, help="JSON results")
args = parser.parse_args()
###################################### HELPER_FUNCTIONS
def log(msg: str) -> None:
    """Print a tagged log line."""
    print(f"[PYTHON-INFO] {msg}", flush=True)
def prepare(X: np.ndarray, max_points: int, metric: str, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Seeded cap, L2-normalisation for the cosine metric, and removal of exact duplicate rows (returned as a count)."""
    X = np.asarray(X, dtype=np.float64)
    if len(X) > max_points: X = X[np.sort(rng.choice(len(X), size=max_points, replace=False))]
    if metric == "cosine": X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    elif metric != "euclidean": raise ValueError(f"metric {metric!r}: euclidean or cosine")
    U = np.unique(X, axis=0)
    return U, len(X) - len(U)
def estimate(X: np.ndarray, mu_fraction: float, n_min: int) -> dict:
    """compute_id_2NN on all points and return_id_scaling_2NN for the scale curve."""
    data = Data(coordinates=X, maxk=3, verbose=False, n_jobs=1)
    d, _, scale = data.compute_id_2NN(mu_fraction=mu_fraction)
    ids, errs, rs = data.return_id_scaling_2NN(n_min=n_min, mu_fraction=mu_fraction)
    sizes = [int(n) for n in np.round(len(X) / 2.0 ** np.arange(len(ids)))]
    curve = [dict(points=n, d=round(float(a), 3), spread=round(float(e), 3) if i else None, scale=round(float(r), 5)) for i, (n, a, e, r) in enumerate(zip(sizes, ids, errs, rs))]
    return dict(d=round(float(d), 3), error=curve[1]["spread"] if len(curve) > 1 else None, scale=round(float(scale), 5), decimation=curve)
###################################### MAIN
def main() -> None:
    """Estimate every task of the manifest and write the results."""
    t0 = time.time()
    m = json.loads(Path(args.manifest).read_text()); S = m["settings"]
    rng, version, out = np.random.default_rng(SEED), importlib.metadata.version("dadapy"), []
    log(f"TWO-NN with DADApy {version} (Glielmo et al. 2022, doi:10.1016/j.patter.2022.100589; Facco et al. 2017, doi:10.1038/s41598-017-11873-y)")
    for t in tqdm(m["tasks"], desc="TWO-NN", mininterval=30):
        X, dup = prepare(np.load(t["path"]), int(S["max_points"]), S["metric"], rng)
        r = dict(name=t["name"], points=len(X) + dup, dropped=dup, dropped_warning=dup > DUPLICATE_WARN * max(1, len(X) + dup), metric=S["metric"], mu_fraction=float(S["mu_fraction"]), n_min=int(S["n_min"]), engine=f"dadapy {version}", differences=DIFFERENCES)
        r.update(estimate(X, float(S["mu_fraction"]), int(S["n_min"])) if len(X) >= MIN_POINTS else dict(d=None, error=None, scale=None, decimation=[]))
        if r["dropped_warning"]: log(f"!! {t['name']}: {dup} of {r['points']} points are exact duplicates (above 10%): the estimate is unreliable")
        out.append(r)
    tmp = Path(f"{args.out}.tmp"); tmp.write_text(json.dumps(out)); tmp.replace(args.out)
    log(f"{len(out)} estimate(s) -> {args.out}")
    log(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} | elapsed {time.time() - t0:.1f} s | max RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576:.2f} GB")
if __name__ == "__main__":
    main()
