#!/usr/bin/env python3
"""Summarise an analysis into tables and cheap aggregates for the Results tab, incl. TWO-NN through DADApy and an optional joint PCA."""
###################################### DEPENDENCIES_AND_CONSTANTS
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import numpy as np
from tqdm import tqdm
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parent))
import store
SEED = 42
YAML_PATH = "configs/param_configs.yaml"
SCRIPT_KEY = "summarize"
PAIRS = 2000
NORM_BINS = 20
LENGTH_EDGES = [0, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 5000, 10**9]
TWONN = dict(enabled=False, max_points=10000, mu_fraction=0.9, n_min=20, metric="euclidean", tokens=True)
DADAPY_PYTHON = "/opt/envs/dadapy/bin/python"
WRAPPER = Path(__file__).resolve().parent / "intrinsic_dim.py"
PCA = dict(enabled=False, per_sample_cap=1000, hex_bins=30, per_sample_pca=False, max_components=50)
EMBED_COLS = ["sample_id", "backend", "proteins", "dim", "keys", "store_dtype", "compute_dtype", "truncated", "files", "revision", "input_sha256"]
SAMPLE_COLS = ["sample_id", "records", "kept", "invalid", "source", "uniprot_release", "fasta_sha256"]
ID_COLS = ["backend", "key", "layer", "pooling", "sample", "points", "d", "error", "dropped", "dropped_warning", "metric", "stratified_tokens", "proteins_used"]
###################################### CONFIG_AND_PARSER
parser = argparse.ArgumentParser(description="Summarise the embeddings of an analysis")
parser.add_argument("--analysis_dir", type=str, default=None, help="results/<analysis>")
parser.add_argument("--pairs", type=int, default=PAIRS, help="Random protein pairs for the within/between cosine")
parser.add_argument("--norm_bins", type=int, default=NORM_BINS)
parser.add_argument("--twonn", type=yaml.safe_load, default=TWONN, help="TWO-NN settings (mapping)")
parser.add_argument("--pca", type=yaml.safe_load, default=PCA, help="PCA settings (mapping)")
parser.add_argument("--yaml-in", nargs="?", const=YAML_PATH, default=YAML_PATH)
parser.add_argument("--yaml-out", nargs="?", const=YAML_PATH, default=None)
cli_preview, _ = parser.parse_known_args()
params = dict(pairs=PAIRS, norm_bins=NORM_BINS, twonn=TWONN, pca=PCA)
if cli_preview.yaml_in and Path(cli_preview.yaml_in).exists():
    config_data = yaml.safe_load(Path(cli_preview.yaml_in).read_text()) or {}
    sec = config_data.get(SCRIPT_KEY) or {}
    params.update({k: ({**params[k], **sec[k]} if isinstance(params[k], dict) else sec[k]) for k in sec if k in params})
parser.set_defaults(**params)
args = parser.parse_args()
if args.yaml_out:
    config_data = (yaml.safe_load(Path(args.yaml_out).read_text()) or {}) if Path(args.yaml_out).exists() else {}
    config_data[SCRIPT_KEY] = {k: getattr(args, k) for k in params}
    Path(args.yaml_out).write_text(yaml.safe_dump(config_data, sort_keys=False))
rng = np.random.default_rng(SEED)
###################################### HELPER_FUNCTIONS
def log(msg: str) -> None:
    """Print a tagged log line."""
    print(f"[PYTHON-INFO] {msg}", flush=True)
def write_tsv(path: Path, cols: list[str], rows: list[dict]) -> None:
    """Write a TSV atomically."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\t".join(cols) + "\n" + "".join("\t".join(str(r.get(c, "")) for c in cols) + "\n" for r in rows)); tmp.replace(path)
def hist(values: list, edges: list) -> list[int]:
    """Counts of values in bins."""
    return np.histogram(values, bins=edges)[0].tolist() if len(values) else [0] * (len(edges) - 1)
def unit(x: np.ndarray) -> np.ndarray:
    """Rows scaled to unit length."""
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
def layer_of(k: str) -> int:
    """Layer number of a key layer_<k>/<pooling>."""
    return int(k.split("/")[0].split("_")[1])
def hexbin(xy: np.ndarray, box: list[float], n: int) -> list[list[int]]:
    """Hexagonal binning on a shared box: nearest centre of two offset rectangular lattices; returns [col, row, lattice, count] for non-empty cells."""
    x0, x1, y0, y1 = box; sx, sy = (x1 - x0) / n, (y1 - y0) / n * np.sqrt(3)
    u, v = (np.clip(xy[:, 0], x0, x1) - x0) / sx, (np.clip(xy[:, 1], y0, y1) - y0) / sy
    a = np.stack([np.round(u), np.round(v)], 1); b = np.stack([np.floor(u) + 0.5, np.floor(v) + 0.5], 1)
    da, db = ((np.stack([u, v], 1) - a) * [1, np.sqrt(3)]) ** 2, ((np.stack([u, v], 1) - b) * [1, np.sqrt(3)]) ** 2
    use_a = da.sum(1) <= db.sum(1)
    cells = np.where(use_a[:, None], np.c_[a, np.zeros(len(a))], np.c_[np.floor(u), np.floor(v), np.ones(len(a))]).astype(np.int64)
    keys, cnt = np.unique(cells, axis=0, return_counts=True)
    return [[int(c), int(r), int(l), int(k)] for (c, r, l), k in zip(keys, cnt)]
def samples_of(idx: dict[str, Path], k: str, cap: int) -> dict[str, np.ndarray]:
    """A seeded random subsample of at most cap vectors per sample."""
    out = {}
    for s, p in idx.items():
        n = json.loads(p.read_text())["n"]
        out[s] = store.take_rows(p, rng.choice(n, size=min(cap, n), replace=False).tolist(), k).astype(np.float64)
    return out
def id_tasks(b: str, idx: dict[str, Path], k: str, n0: int, tmp: Path) -> list[dict]:
    """Write the point sets of one representation (per sample and all samples together; mean: proteins, tokens: residues stratified by protein) as .npy files for the DADApy wrapper."""
    T, pts, used = args.twonn, {}, {}
    for s, p in idx.items():
        if k.endswith("/tokens"): pts[s], used[s] = store.take_tokens(p, k, int(T["max_points"]), rng)
        else: pts[s] = samples_of({s: p}, k, int(T["max_points"]))[s]; used[s] = len(pts[s])
    per = max(1, int(T["max_points"]) // max(1, len(pts)))
    pts["all samples"] = np.concatenate([v[rng.choice(len(v), size=min(per, len(v)), replace=False)] for v in pts.values() if len(v)])
    out = []
    for i, (s, X) in enumerate(pts.items()):
        np.save(tmp / f"t{n0 + i}.npy", X.astype(np.float32))
        out.append(dict(name=f"t{n0 + i}", path=str(tmp / f"t{n0 + i}.npy"), backend=b, key=k, layer=layer_of(k), pooling=k.split("/")[1], sample=s, stratified_tokens=k.endswith("/tokens"), proteins_used=used.get(s, sum(used.values()))))
    return out
def run_dadapy(tasks: list[dict], tmp: Path) -> list[dict]:
    """Run src/intrinsic_dim.py with the dadapy venv interpreter on every point set."""
    (tmp / "manifest.json").write_text(json.dumps(dict(settings=args.twonn, tasks=[dict(name=t["name"], path=t["path"]) for t in tasks])))
    subprocess.run([dadapy_python(), str(WRAPPER), "--manifest", str(tmp / "manifest.json"), "--out", str(tmp / "out.json")], check=True)
    res = {r["name"]: r for r in json.loads((tmp / "out.json").read_text())}
    return [dict({k: v for k, v in t.items() if k not in ("path", "name")}, **{k: v for k, v in res[t["name"]].items() if k != "name"}) for t in tasks if res[t["name"]]["d"] is not None]
def dadapy_python() -> str:
    """Interpreter of the dadapy env (MP_DADAPY_PYTHON overrides); stops before any work if it is missing."""
    py = os.environ.get("MP_DADAPY_PYTHON") or DADAPY_PYTHON
    if not Path(py).is_file(): sys.exit(f"[PYTHON-INFO] !! TWO-NN needs the dadapy env ({py}) of the MultiPLM image; disable summarize.twonn.enabled to skip it")
    return py
def pca_block(idx: dict[str, Path], k: str, lens: dict[str, np.ndarray], cent: dict[str, np.ndarray]) -> dict:
    """Joint PCA of one key: hexbin densities, explained variance, PC1 vs length, nearest-centroid confusion."""
    P = args.pca; sub = samples_of(idx, k, int(P["per_sample_cap"])); names = list(idx)
    X = np.concatenate([sub[s] for s in names]); mu = X.mean(0); _, sv, vt = np.linalg.svd(X - mu, full_matrices=False)
    ev = sv ** 2 / (sv ** 2).sum(); cum = np.cumsum(ev)
    C = unit(np.stack([cent[s] for s in names])); xy, conf, spread = {}, np.zeros((len(names), len(names)), dtype=np.int64), {}
    for i, s in enumerate(names):
        parts, cos_own = [], []
        for _, blk in store.iter_blocks(idx[s], k):
            blk = blk.astype(np.float64); parts.append((blk - mu) @ vt[:2].T)
            sim = unit(blk) @ C.T; conf[i] += np.bincount(sim.argmax(1), minlength=len(names)); cos_own.append(1 - sim[:, i])
        xy[s], spread[s] = np.concatenate(parts), float(np.mean(np.concatenate(cos_own)))
    allxy = np.concatenate(list(xy.values())); lo, hi = np.percentile(allxy, 0.5, 0), np.percentile(allxy, 99.5, 0); pad = (hi - lo) * 0.05
    box = [float(lo[0] - pad[0]), float(hi[0] + pad[0]), float(lo[1] - pad[1]), float(hi[1] + pad[1])]
    pc1, L = np.concatenate([xy[s][:, 0] for s in names]), np.concatenate([lens[s] for s in names])
    r = float(np.corrcoef(pc1, L)[0, 1]) if len(pc1) > 2 and np.std(L) > 0 else None
    out = dict(key=k, samples=names, box=box, bins=int(P["hex_bins"]), hex={s: hexbin(xy[s], box, int(P["hex_bins"])) for s in names}, clipped={s: int(((xy[s][:, 0] < box[0]) | (xy[s][:, 0] > box[1]) | (xy[s][:, 1] < box[2]) | (xy[s][:, 1] > box[3])).sum()) for s in names},
               explained=ev[:int(P["max_components"])].round(5).tolist(), components_90=int(np.searchsorted(cum, 0.9) + 1), fit_points={s: len(sub[s]) for s in names},
               pc1_length=dict(r=None if r is None else round(r, 3), bins=[float(pc1[(L >= a) & (L < b)].mean()) if ((L >= a) & (L < b)).any() else None for a, b in zip(LENGTH_EDGES[:-1], LENGTH_EDGES[1:])]),
               confusion=conf.tolist(), within_spread={s: round(v, 4) for s, v in spread.items()})
    if P.get("per_sample_pca"):
        own = {}
        for s in names:
            Y = sub[s] - sub[s].mean(0); _, sv2, vt2 = np.linalg.svd(Y, full_matrices=False); q = Y @ vt2[:2].T
            b2 = [float(q[:, 0].min()), float(q[:, 0].max() + 1e-9), float(q[:, 1].min()), float(q[:, 1].max() + 1e-9)]
            own[s] = dict(box=b2, hex=hexbin(q, b2, int(P["hex_bins"])), explained=(sv2[:2] ** 2 / (sv2 ** 2).sum()).round(4).tolist())
        out["per_sample"] = own
    return out
def norm_stats(b: str, idx: dict[str, Path], k: str, lens: dict[str, dict[str, int]]) -> tuple[dict, dict, dict]:
    """Norm and cosine statistics of one mean key; also returns centroids and row lengths."""
    norms, cent, row_len = {}, {}, {}
    for s, p in idx.items():
        ids = list(json.loads((p.parent / f"{s}.ids.json").read_text()))
        tot, n, ns = None, 0, []
        for _, blk in store.iter_blocks(p, k):
            blk = blk.astype(np.float64); ns.append(np.linalg.norm(blk, axis=1)); tot = blk.sum(0) if tot is None else tot + blk.sum(0); n += len(blk)
        norms[s], cent[s], row_len[s] = np.concatenate(ns), tot / max(n, 1), np.array([lens[s].get(i, 0) for i in ids])
    allnorm = np.concatenate(list(norms.values())); edges = np.linspace(allnorm.min(), allnorm.max() + 1e-9, args.norm_bins + 1)
    names = list(idx); C = unit(np.stack([cent[s] for s in names])); within, between = [], []
    for _ in tqdm(range(args.pairs), desc=f"{b} pairs", mininterval=30):
        s = names[rng.integers(len(names))]
        if len(norms[s]) > 1:
            a, c = rng.choice(len(norms[s]), 2, replace=False); v = unit(np.stack(store.read_rows(idx[s], [int(a), int(c)], k))); within.append(float(v[0] @ v[1]))
        if len(names) > 1:
            s1, s2 = rng.choice(len(names), 2, replace=False); v = unit(np.stack(store.read_rows(idx[names[s1]], [int(rng.integers(len(norms[names[s1]])))], k) + store.read_rows(idx[names[s2]], [int(rng.integers(len(norms[names[s2]])))], k))); between.append(float(v[0] @ v[1]))
    lb = {s: [float(norms[s][(row_len[s] >= lo) & (row_len[s] < hi)].mean()) if ((row_len[s] >= lo) & (row_len[s] < hi)).any() else None for lo, hi in zip(LENGTH_EDGES[:-1], LENGTH_EDGES[1:])] for s in names}
    out = dict(key=k, norm_edges=edges.round(4).tolist(), norm_hist={s: hist(norms[s], edges) for s in names}, length_vs_norm=lb, centroid_cosine=dict(samples=names, matrix=(C @ C.T).round(4).tolist()),
               cosine=dict(within=dict(n=len(within), mean=round(float(np.mean(within)), 4) if within else None, hist=hist(within, np.linspace(-1, 1, 21))), between=dict(n=len(between), mean=round(float(np.mean(between)), 4) if between else None, hist=hist(between, np.linspace(-1, 1, 21)))))
    return out, cent, row_len
###################################### MAIN
def main() -> None:
    """Write summary.tsv, samples_summary.tsv, intrinsic_dimension.tsv and aggregates.json."""
    t0 = time.time()
    if not args.analysis_dir:
        if args.yaml_out: return
        parser.error("--analysis_dir is required")
    a = Path(args.analysis_dir); rows, samples, ids_rows = [], [], []
    agg = dict(length_edges=LENGTH_EDGES[:-1] + ["inf"], samples={}, backends={}, throughput={}, twonn=dict(settings=args.twonn, rows=[]), pca={})
    tasks = []
    if args.twonn.get("enabled"): dadapy_python()
    tmp = Path(tempfile.mkdtemp(dir=a, prefix=".twonn_"))
    lens = {}
    for side in sorted((a / "fasta").glob("*.json")):
        s, p = side.stem, json.loads(side.read_text())
        tab = list(csv.DictReader(open(a / "fasta" / f"{s}.tsv"), delimiter="\t")); lens[s] = {r["record"]: int(r["length"]) for r in tab if r["status"] == "kept"}
        up = (p.get("uniprot") or {}).get("requests") or [{}]
        samples.append(dict(sample_id=s, records=p["records"], kept=p["kept"], invalid=p["records"] - p["kept"], source=(p.get("uniprot") or {}).get("uniprot_query") or p["source"], uniprot_release=up[0].get("uniprot_release") or "", fasta_sha256=p["fasta_sha256"]))
        dd = a / "dedup" / f"{s}.dedup.json"
        agg["samples"][s] = dict(status=p["status"], length_hist=hist(list(lens[s].values()), LENGTH_EDGES), dedup=json.loads(dd.read_text()) if dd.is_file() else None)
    for b_dir in sorted(d for d in (a / "embeddings").glob("*") if d.is_dir()):
        b, idx = b_dir.name, {}
        for prov_p in sorted(b_dir.glob("*.json")):
            if prov_p.name.endswith((".ids.json", ".index.json")): continue
            s, prov = prov_p.stem, json.loads(prov_p.read_text())
            index = json.loads((b_dir / f"{s}.index.json").read_text()); idx[s] = b_dir / f"{s}.index.json"
            if agg["samples"].get(s): agg["samples"][s].setdefault("embedded", {})[b] = index["n"]
            rows.append(dict(sample_id=s, backend=b, proteins=index["n"], dim=prov["dim"], keys=",".join(index["keys"]), store_dtype=prov["store_dtype"], compute_dtype=prov["compute_dtype"], truncated=prov["truncated"], files=len(index["parts"]), revision=prov["revision"], input_sha256=prov["input_sha256"]))
            t = agg["throughput"].setdefault(b, dict(proteins=0, gpu_s=0.0, shards=0))
            for sh in prov["shards"]: t["proteins"] += sh["proteins"]; t["gpu_s"] += sh["elapsed_s"]; t["shards"] += 1
        if not idx: continue
        keys = sorted(json.loads(next(iter(idx.values())).read_text())["keys"])
        means = sorted((k for k in keys if k.endswith("/mean")), key=layer_of)
        if means:
            stats, cent, row_len = norm_stats(b, idx, means[-1], lens); agg["backends"][b] = stats
        if args.twonn.get("enabled"):
            for k in tqdm([k for k in keys if k.endswith("/mean") or (k.endswith("/tokens") and args.twonn.get("tokens"))], desc=f"{b} TWO-NN", mininterval=30):
                tasks += id_tasks(b, idx, k, len(tasks), tmp)
        if args.pca.get("enabled") and means:
            for k in tqdm(means, desc=f"{b} PCA", mininterval=30):
                cent_k = cent if k == means[-1] else {s: sum(blk.astype(np.float64).sum(0) for _, blk in store.iter_blocks(p, k)) for s, p in idx.items()}
                agg["pca"].setdefault(b, {})[k] = pca_block(idx, k, row_len, cent_k)
    if tasks: ids_rows = agg["twonn"]["rows"] = run_dadapy(tasks, tmp)
    shutil.rmtree(tmp)
    for t in agg["throughput"].values(): t["proteins_per_gpu_hour"] = round(t["proteins"] / (t["gpu_s"] / 3600), 1) if t["gpu_s"] else None
    write_tsv(a / "summary.tsv", EMBED_COLS, rows); write_tsv(a / "samples_summary.tsv", SAMPLE_COLS, samples); write_tsv(a / "intrinsic_dimension.tsv", ID_COLS, ids_rows)
    tmp = a / ".aggregates.json.tmp"; tmp.write_text(json.dumps(agg)); tmp.replace(a / "aggregates.json")
    log(f"{len(rows)} embedding set(s) of {len(samples)} sample(s); {len(ids_rows)} TWO-NN estimate(s); PCA for {sum(len(v) for v in agg['pca'].values())} representation(s) -> {a}")
    log(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} | elapsed {time.time() - t0:.1f} s | max RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576:.2f} GB")
if __name__ == "__main__":
    main()
