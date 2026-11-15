"""safetensors embedding storage: atomic shard chunks, streaming merge into one file or row parts with index and id sidecar, lazy reads (keys layer_<k>/mean, layer_<k>/tokens + token_offsets, lengths, truncated)."""
###################################### DEPENDENCIES_AND_CONSTANTS
import json
import os
from pathlib import Path
import struct
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file
PREFIX, SUFFIX, ALIGN = "chunk_", ".safetensors", 8
DTYPES = dict(float32=(np.float32, "F32"), float16=(np.float16, "F16"))
ST_DTYPES = dict(F32=np.float32, F16=np.float16, I32=np.int32, I64=np.int64, U8=np.uint8)
###################################### HELPER_FUNCTIONS
def key(layer: int, pooling: str) -> str:
    """Tensor key of one layer and pooling."""
    return f"layer_{layer}/{pooling}"
def write_chunk(d: Path, j: int, tensors: dict[str, np.ndarray], ids: list[str], batch_lo: int, batch_hi: int) -> Path:
    """Write chunk j of a shard atomically (hidden temporary name, then rename).

    Args:
        d (Path): Shard chunk folder.
        j (int): Chunk number.
        tensors (dict[str, np.ndarray]): layer keys plus lengths, truncated and (with tokens) token_offsets.
        ids (list[str]): FASTA ids of the chunk rows.
        batch_lo (int): First batch of the shard stored in this chunk.
        batch_hi (int): One past the last batch stored in this chunk.

    Returns:
        Path: The chunk file.
    """
    d.mkdir(parents=True, exist_ok=True)
    out, tmp = d / f"{PREFIX}{j:05d}{SUFFIX}", d / f".{PREFIX}{j:05d}{SUFFIX}.tmp"
    save_file({k: np.ascontiguousarray(v) for k, v in tensors.items()}, str(tmp), metadata=dict(ids=json.dumps(ids), batch_lo=str(batch_lo), batch_hi=str(batch_hi)))
    with open(tmp, "rb") as f: os.fsync(f.fileno())
    os.replace(tmp, out)
    return out
def chunks(d: Path) -> list[Path]:
    """Complete chunk files of a shard, in order (temporary files are ignored)."""
    return sorted(Path(d).glob(f"{PREFIX}*{SUFFIX}")) if Path(d).is_dir() else []
def chunk_meta(p: Path) -> dict:
    """ids, batch range, rows, tensor keys with dtype and shape, and token offsets of one chunk."""
    with safe_open(str(p), framework="numpy") as f:
        m, keys = f.metadata(), list(f.keys())
        shapes = {k: f.get_slice(k).get_shape() for k in keys}
        dts = {k: f.get_slice(k).get_dtype() for k in keys}
        offs = f.get_tensor("token_offsets") if "token_offsets" in keys else None
    return dict(ids=json.loads(m["ids"]), batch_lo=int(m["batch_lo"]), batch_hi=int(m["batch_hi"]), rows=shapes["lengths"][0], shapes=shapes, dtypes=dts, offsets=offs)
def resume_batch(d: Path) -> tuple[int, int]:
    """(next chunk number, first batch not yet stored) of a shard; a half-written or unreadable last chunk is removed and never counts."""
    cs = chunks(d)
    while cs:
        try: return len(cs), chunk_meta(cs[-1])["batch_hi"]
        except Exception: cs.pop().unlink()
    return 0, 0
def header(tensors: list[tuple[str, str, list[int], int]], meta: dict) -> bytes:
    """Safetensors header for tensors given as (name, dtype, shape, nbytes), laid out back to back, padded to 8 bytes."""
    h, off = {"__metadata__": {k: str(v) for k, v in meta.items()}}, 0
    for name, dt, shape, n in tensors: h[name] = dict(dtype=dt, shape=shape, data_offsets=[off, off + n]); off += n
    b = json.dumps(h, separators=(",", ":")).encode()
    return b + b" " * (-len(b) % ALIGN)
def write_part(path: Path, sources: list[tuple[Path, dict, int, int]], meta: dict) -> None:
    """Stream rows a:b of each source chunk into one safetensors file, one chunk slice in memory at a time.

    Args:
        path (Path): Output file (written as .tmp, then renamed).
        sources (list[tuple[Path, dict, int, int]]): (chunk file, its chunk_meta, first row, end row), in output order.
        meta (dict): Header metadata.
    """
    info = sources[0][1]
    keys = [k for k in info["shapes"] if k not in ("token_offsets",)]
    def tok_rows(m: dict, a: int, b: int) -> int:
        """Residue rows of chunk rows a:b."""
        return int(m["offsets"][b] - m["offsets"][a]) if m["offsets"] is not None else 0
    rows, toks = sum(b - a for _, _, a, b in sources), sum(tok_rows(m, a, b) for _, m, a, b in sources)
    spec = []
    for k in keys:
        dt = info["dtypes"][k]; n0 = toks if k.endswith("/tokens") else rows
        shape = [n0] + list(info["shapes"][k][1:])
        spec.append((k, dt, shape, int(np.prod(shape)) * np.dtype(ST_DTYPES[dt]).itemsize))
    if info["offsets"] is not None: spec.append(("token_offsets", "I64", [rows + 1], (rows + 1) * 8))
    hb, tmp = header(spec, meta), path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as out:
        out.write(struct.pack("<Q", len(hb))); out.write(hb)
        for k in keys:
            for src, m, a, b in sources:
                lo, hi = (int(m["offsets"][a]), int(m["offsets"][b])) if k.endswith("/tokens") else (a, b)
                with safe_open(str(src), framework="numpy") as f: out.write(np.ascontiguousarray(f.get_slice(k)[lo:hi]).tobytes())
        if info["offsets"] is not None:
            base, acc = 0, [np.zeros(1, dtype=np.int64)]
            for _, m, a, b in sources:
                o = m["offsets"][a:b + 1].astype(np.int64); acc.append(o[1:] - o[0] + base); base += int(o[-1] - o[0])
            out.write(np.concatenate(acc).tobytes())
    os.replace(tmp, path)
def merge(chunk_files: list[Path], out_dir: Path, stem: str, max_part_bytes: int, meta: dict) -> dict:
    """Concatenate chunk files (in the given order) into <stem>.safetensors, or row-split parts above max_part_bytes, plus <stem>.index.json and the id -> row sidecar <stem>.ids.json.

    Args:
        chunk_files (list[Path]): Chunks in output row order.
        out_dir (Path): Destination folder.
        stem (str): File stem (the sample id).
        max_part_bytes (int): Largest payload per file.
        meta (dict): Metadata copied into every header and the index.

    Returns:
        dict: The index (n, keys with shape per row, parts with row ranges, metadata).
    """
    infos = [chunk_meta(c) for c in chunk_files]
    if not infos: raise ValueError("no chunk to merge")
    sig = {(tuple(sorted((k, tuple(s[1:])) for k, s in i["shapes"].items())), tuple(sorted(i["dtypes"].items()))) for i in infos}
    if len(sig) != 1: raise ValueError("chunks disagree on tensor keys, shapes or dtypes")
    ids = [x for i in infos for x in i["ids"]]
    if len(set(ids)) != len(ids): raise ValueError("duplicate protein ids across chunks")
    def rowb(i: dict, r: int) -> int:
        """Bytes of one protein row over every key."""
        return sum((int(i["offsets"][r + 1] - i["offsets"][r]) if k.endswith("/tokens") else 1) * int(np.prod(s[1:])) * np.dtype(ST_DTYPES[i["dtypes"][k]]).itemsize for k, s in i["shapes"].items() if k != "token_offsets")
    groups, cur, size = [], [], 0
    for c, i in zip(chunk_files, infos):
        for r in range(i["rows"]):
            b = rowb(i, r)
            if cur and size + b > max_part_bytes: groups.append(cur); cur, size = [], 0
            cur.append((c, i, r)); size += b
    groups.append(cur)
    out_dir.mkdir(parents=True, exist_ok=True); parts, start = [], 0
    for p, g in enumerate(groups):
        name = f"{stem}{SUFFIX}" if len(groups) == 1 else f"{stem}-{p + 1:05d}-of-{len(groups):05d}{SUFFIX}"
        sources = []
        for c, i, r in g:
            if sources and sources[-1][0] == c and sources[-1][3] == r: sources[-1][3] = r + 1
            else: sources.append([c, i, r, r + 1])
        write_part(out_dir / name, [tuple(s) for s in sources], dict(meta, row_start=start, row_end=start + len(g)))
        parts.append(dict(file=name, row_start=start, row_end=start + len(g))); start += len(g)
    i0 = infos[0]
    index = dict(n=len(ids), keys={k: dict(dtype=i0["dtypes"][k], row_shape=list(s[1:]), per="residue" if k.endswith("/tokens") else "protein") for k, s in i0["shapes"].items() if k != "token_offsets"}, parts=parts, metadata=meta)
    for fname, obj in ((f"{stem}.ids.json", {x: r for r, x in enumerate(ids)}), (f"{stem}.index.json", index)):
        tmp = out_dir / f".{fname}.tmp"; tmp.write_text(json.dumps(obj, indent=None if fname.endswith("ids.json") else 1)); os.replace(tmp, out_dir / fname)
    return index
def part_of(index: dict, row: int) -> dict:
    """The part holding a global row."""
    return next(p for p in index["parts"] if p["row_start"] <= row < p["row_end"])
def read_rows(index_path: Path, rows: list[int], k: str) -> list[np.ndarray]:
    """Read selected proteins of one key lazily (safe_open + get_slice): a vector per protein for mean keys, a residues x dim matrix for token keys."""
    index, out = json.loads(Path(index_path).read_text()), []
    for r in rows:
        part = part_of(index, r); lr = r - part["row_start"]
        with safe_open(str(Path(index_path).parent / part["file"]), framework="numpy") as f:
            if k.endswith("/tokens"):
                o = f.get_slice("token_offsets")[lr:lr + 2]; out.append(f.get_slice(k)[int(o[0]):int(o[1])])
            else: out.append(f.get_slice(k)[lr:lr + 1][0])
    return out
def iter_blocks(index_path: Path, k: str, block: int = 4096) -> object:
    """Yield (first row, block of vectors) of a mean key over every part."""
    index = json.loads(Path(index_path).read_text())
    for part in index["parts"]:
        with safe_open(str(Path(index_path).parent / part["file"]), framework="numpy") as f:
            s, n = f.get_slice(k), part["row_end"] - part["row_start"]
            for a in range(0, n, block): yield part["row_start"] + a, s[a:min(n, a + block)]
def take_rows(index_path: Path, rows: list[int], k: str, block: int = 4096) -> np.ndarray:
    """Vectors of selected proteins of a mean key, read block by block (one pass, rows returned in sorted order)."""
    want, out = np.array(sorted(set(rows)), dtype=np.int64), []
    for r0, blk in iter_blocks(index_path, k, block):
        sel = want[(want >= r0) & (want < r0 + len(blk))]
        if len(sel): out.append(blk[sel - r0])
    return np.concatenate(out) if out else np.zeros((0, 0), dtype=np.float32)
def take_tokens(index_path: Path, k: str, cap: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Residue vectors of a tokens key, stratified by protein: at most ceil(cap / proteins) random residues from each of at most cap proteins."""
    index, out, used = json.loads(Path(index_path).read_text()), [], 0
    pick = sorted(rng.choice(index["n"], size=min(cap, index["n"]), replace=False).tolist())
    per = max(1, -(-cap // max(1, len(pick))))
    for part in index["parts"]:
        mine = [r - part["row_start"] for r in pick if part["row_start"] <= r < part["row_end"]]
        if not mine: continue
        with safe_open(str(Path(index_path).parent / part["file"]), framework="numpy") as f:
            off, s = f.get_tensor("token_offsets"), f.get_slice(k)
            for r in mine:
                a, b = int(off[r]), int(off[r + 1])
                if b > a: m = s[a:b]; out.append(m[np.sort(rng.choice(b - a, size=min(per, b - a), replace=False))]); used += 1
    return (np.concatenate(out)[:cap] if out else np.zeros((0, 0), dtype=np.float32)), used
