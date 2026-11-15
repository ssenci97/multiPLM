"""Fake backend for tests (numpy only, no GPU): deterministic residue vectors; MP_FAKE_FAIL_AFTER=n raises on call n+1, MP_FAKE_NAN=1 returns NaN, MP_FAKE_DIM reports another dimension."""
###################################### DEPENDENCIES_AND_CONSTANTS
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
CALLS = {"n": 0}
###################################### HELPER_FUNCTIONS
def load(spec: dict, files: Path, device: object, dtype: str) -> SimpleNamespace:
    """Return a backend whose residue vectors are a fixed function of the sequence."""
    dim, fail, nan = spec["dim"], int(os.environ.get("MP_FAKE_FAIL_AFTER") or 0), os.environ.get("MP_FAKE_NAN") == "1"
    def residues(seqs: list[str], layers: list[int]) -> dict[int, list[np.ndarray]]:
        """Residue matrices; raises after MP_FAKE_FAIL_AFTER calls."""
        CALLS["n"] += 1
        if fail and CALLS["n"] > fail: raise RuntimeError("fake backend: simulated kill")
        return {k: [np.array([[np.nan if nan else (ord(c) * (j + 1) + d * k) % 97 / 97.0 for d in range(dim)] for j, c in enumerate(s)], dtype=np.float32) for s in seqs] for k in layers}
    return SimpleNamespace(dim=int(os.environ.get("MP_FAKE_DIM") or dim), n_layers=spec["n_layers"], max_length=0, dtype="fp32", one_at_a_time=False, residues=residues)
