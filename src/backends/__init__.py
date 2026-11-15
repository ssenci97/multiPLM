"""Backend registry and helpers: pinned weights of configs/tools.tsv, layer selection, batching and sharding, the weights folder checksum check, and the loader of each family (stdlib only, so the host and every env can import it)."""
###################################### DEPENDENCIES_AND_CONSTANTS
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[2]
TOOLS_TSV = Path(os.environ.get("MP_TOOLS_TSV") or ROOT / "configs" / "tools.tsv")
FAMILY_MODULE = dict(esm2="esm2", esm3="esm3", esmc="esmc", esmc6b="esmc", prott5="t5", prostt5="t5", fake="fake")
POOLINGS = ("mean", "tokens")
CHUNK = 1 << 24
###################################### HELPER_FUNCTIONS
def registry(path: Path = None) -> dict[str, dict]:
    """Read the weights rows of configs/tools.tsv (or MP_TOOLS_TSV).

    Args:
        path (Path): tools.tsv; defaults to TOOLS_TSV.

    Returns:
        dict[str, dict]: Backend name -> row with files as a list and dim/n_layers as int.
    """
    rows = csv.DictReader(open(path or TOOLS_TSV), delimiter="\t")
    return {r["name"]: dict(r, files=[f for f in r["files"].split(",") if f and f != "-"], dim=int(r["dim"]), n_layers=int(r["n_layers"])) for r in rows if r["kind"] == "weights"}
def resolve_layers(spec: dict, layers: object) -> list[int]:
    """Turn a layers setting (list of positive/negative indices, or 'all') into sorted block numbers 1..n_layers.

    Layer n_layers (= -1) is the final representation each library returns (normalised); 1..n_layers-1 are raw block outputs.

    Args:
        spec (dict): Registry row with n_layers and layers (all or last).
        layers (object): 'all', an int, or a list of ints.

    Returns:
        list[int]: Block numbers.

    Raises:
        ValueError: An index outside the model, or a non-final layer for a backend that exposes only the last one.
    """
    n = spec["n_layers"]
    want = list(range(1, n + 1)) if layers == "all" else [layers] if isinstance(layers, int) else list(layers or [-1])
    out = []
    for k in want:
        if not isinstance(k, int) or k == 0 or abs(k) > n: raise ValueError(f"{spec['name']}: layer {k!r} outside 1..{n} / -1..-{n}")
        out.append(k if k > 0 else n + 1 + k)
    if spec["layers"] == "last" and set(out) != {n}: raise ValueError(f"{spec['name']}: only the last layer (-1) is exposed by its forward pass")
    return sorted(set(out))
def poolings(setting: object, layer: int, n: int) -> list[str]:
    """Poolings of one layer: a list for every layer, or a mapping layer -> list (keys may be negative)."""
    if isinstance(setting, dict):
        got = setting.get(layer, setting.get(str(layer), setting.get(layer - n - 1, setting.get(str(layer - n - 1), ["mean"]))))
    else: got = setting or ["mean"]
    got = [got] if isinstance(got, str) else list(got)
    if not got or set(got) - set(POOLINGS): raise ValueError(f"pooling {got!r}: use mean, tokens or both")
    return got
def model_dir(models_dir: Path, name: str) -> Path:
    """Return data/models/<name>."""
    return Path(models_dir) / name
def sha256(path: Path) -> str:
    """Return the sha256 of a file, read in 16 MB blocks."""
    from tqdm import tqdm
    h, size = hashlib.sha256(), Path(path).stat().st_size
    with open(path, "rb") as f, tqdm(total=size, unit="B", unit_scale=True, desc=f"sha256 {Path(path).name}", mininterval=30, disable=size < 1 << 30) as bar:
        for block in iter(lambda: f.read(CHUNK), b""): h.update(block); bar.update(len(block))
    return h.hexdigest()
def weight_files(d: Path) -> list[Path]:
    """List the downloaded files of a model folder, without the Hugging Face local-dir metadata."""
    return sorted(p for p in (d / "files").rglob("*") if p.is_file() and ".cache" not in p.relative_to(d / "files").parts)
def read_checksums(path: Path) -> dict[str, str]:
    """Read a sha256sum-style file into {relative path: sha256}."""
    return {rel.strip(): sha for sha, rel in (line.split(None, 1) for line in open(path) if line.strip())} if Path(path).is_file() else {}
def write_checksums(d: Path, sums: dict[str, str]) -> None:
    """Write <model dir>/CHECKSUMS atomically in sha256sum format."""
    tmp = d / "CHECKSUMS.tmp"
    tmp.write_text("".join(f"{s}  {rel}\n" for rel, s in sorted(sums.items())))
    tmp.replace(d / "CHECKSUMS")
def compare(got: dict[str, str], want: dict[str, str], what: str) -> list[str]:
    """Return human-readable differences between two checksum maps."""
    errs = [f"{rel}: missing (listed in {what})" for rel in want if rel not in got]
    errs += [f"{rel}: not listed in {what}" for rel in got if rel not in want]
    return errs + [f"{rel}: sha256 {got[rel][:12]}… differs from {want[rel][:12]}… in {what}" for rel in got if rel in want and got[rel] != want[rel]]
def verify(d: Path, spec: dict, full: bool = True) -> tuple[dict[str, str], list[str]]:
    """Check a model folder against its pinned revision and its recorded CHECKSUMS.

    Args:
        d (Path): data/models/<name>.
        spec (dict): Registry row of the backend.
        full (bool): Hash every file; otherwise compare only the file list.

    Returns:
        tuple[dict[str, str], list[str]]: Recorded checksums and the list of problems (empty when the folder matches).
    """
    src, rec = d / "SOURCE.json", read_checksums(d / "CHECKSUMS")
    if not rec: return rec, [f"{d}/CHECKSUMS missing: run the 00_weights job for {spec['name']}"]
    meta = json.loads(src.read_text()) if src.is_file() else {}
    if meta.get("revision") != spec["commit"]: return rec, [f"{d} holds revision {meta.get('revision')}, configs/tools.tsv pins {spec['commit']}: move the folder aside and download again"]
    files = {str(p.relative_to(d)): p for p in weight_files(d)}
    got = {rel: sha256(p) for rel, p in files.items()} if full else {rel: rec.get(rel, "") for rel in files}
    return rec, compare(got, rec, f"{d}/CHECKSUMS")
def batches(order: list[int], lengths: list[int], budget: int, single: bool) -> list[list[int]]:
    """Group indices (already sorted by length) so that batch size x longest length stays within the budget."""
    out, cur = [], []
    for i in order:
        if cur and (single or (len(cur) + 1) * max(lengths[cur[-1]], lengths[i]) > budget): out.append(cur); cur = []
        cur.append(i)
    return out + ([cur] if cur else [])
def shard_batches(groups: list[list[int]], shard: int, shards: int) -> list[list[int]]:
    """Batches of shard (1-based) out of shards, dealt round-robin so every shard gets short and long proteins."""
    return [g for i, g in enumerate(groups) if i % shards == shard - 1]
def load_backend(name: str, models_dir: Path, device: object, dtype: str) -> SimpleNamespace:
    """Load one backend through its family module.

    Args:
        name (str): Backend name of configs/tools.tsv.
        models_dir (Path): data/models.
        device (object): torch.device (or "cpu" for the fake test backend).
        dtype (str): auto, fp32, fp16 or bf16.

    Returns:
        SimpleNamespace: dim, n_layers, max_length, dtype, one_at_a_time and residues(seqs, layers) -> {layer: [L x dim arrays]}.
    """
    spec = registry()[name]
    mod = importlib.import_module(f"backends.{FAMILY_MODULE[spec['family']]}")
    return mod.load(spec, model_dir(models_dir, name) / "files", device, dtype)
def torch_dtype(dtype: str, device: object, half: str = "bf16") -> tuple[object, str]:
    """Resolve auto/fp32/fp16/bf16 into a torch dtype for the device.

    Args:
        dtype (str): Requested dtype.
        device (object): torch.device.
        half (str): Reduced precision to use with auto on a GPU (bf16 falls back to fp32 where unsupported).

    Returns:
        tuple[object, str]: torch dtype and its name.
    """
    import torch
    table = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)
    if dtype != "auto": return table[dtype], dtype
    if device.type != "cuda": return torch.float32, "fp32"
    if half == "bf16" and not torch.cuda.is_bf16_supported(including_emulation=False): return torch.float32, "fp32"
    return table[half], half
