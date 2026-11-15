#!/usr/bin/env python3
"""Benchmarks of Facco et al. 2017 (Sci Rep 7:12140, p.3) with the pinned DADApy, run inside the image by containers/build.sh test (needs the dadapy env, not pytest)."""
###################################### DEPENDENCIES_AND_CONSTANTS
import sys
import numpy as np
from dadapy import Data
SEED = 42
N = 2500
CAUCHY_DRAWS = 5
MU_FRACTIONS = (0.8, 0.85, 0.9, 0.95)
###################################### HELPER_FUNCTIONS
def two_nn(X: np.ndarray, mu_fraction: float = 0.9, period: np.ndarray = None) -> float:
    """TWO-NN estimate of DADApy on all points."""
    return float(Data(coordinates=X, maxk=3, period=period, verbose=False, n_jobs=1).compute_id_2NN(mu_fraction=mu_fraction)[0])
def check(name: str, value: float, lo: float, hi: float) -> bool:
    """Print and test one benchmark."""
    ok = lo <= value <= hi
    print(f"[PYTHON-INFO] {'ok' if ok else '!! FAILED'} {name}: {value:.2f} (expected {lo}-{hi})", flush=True)
    return ok
###################################### MAIN
def main() -> None:
    """Hypercube d=14 with periodic boundaries, Swiss roll, Cauchy d=20 (median over seeds), and mu_fraction 0.80-0.95 agreement."""
    rng, ok = np.random.default_rng(SEED), []
    cube = rng.random((N, 14))
    ok.append(check("hypercube d=14 with pbc (paper 13.91)", two_nn(cube, period=np.ones(14)), 13.0, 15.0))
    t = 1.5 * np.pi * (1 + 2 * rng.random(N)); h = 21 * rng.random(N)
    ok.append(check("Swiss roll (paper 2.01)", two_nn(np.c_[t * np.cos(t), h, t * np.sin(t)]), 1.75, 2.25))
    cauchy = []
    for s in range(CAUCHY_DRAWS):
        r = np.random.default_rng(s); u = r.normal(size=(N, 20)); u /= np.linalg.norm(u, axis=1, keepdims=True)
        cauchy.append(two_nn(u * np.abs(r.standard_cauchy(N))[:, None]))
    ok.append(check("Cauchy d=20, median of 5 draws, 10% discarded (paper 22.16)", float(np.median(cauchy)), 17.0, 27.0))
    fr = [two_nn(cube, f, np.ones(14)) for f in MU_FRACTIONS]
    ok.append(check("hypercube: spread over mu_fraction 0.80-0.95", max(fr) - min(fr), 0.0, 1.5))
    sys.exit(0 if all(ok) else 1)
if __name__ == "__main__":
    main()
