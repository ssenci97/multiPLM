"""safetensors storage: atomic chunks, resume point, streaming merge (single file and row parts), id -> row order, lazy reads of mean and token keys."""
import json
import numpy as np
import pytest
from conftest import load
store = load("store")

def chunkset(d, sizes, dim=4, seed=0):
    rng, means, toks, ids = np.random.default_rng(seed), [], [], []
    for j, lens in enumerate(sizes):
        t = [rng.random((L, dim), dtype=np.float32) for L in lens]; m = np.stack([x.mean(0) for x in t])
        ii = [f"c{j}_{i}" for i in range(len(lens))]
        store.write_chunk(d, j, {"layer_3/mean": m, "layer_3/tokens": np.concatenate(t), "token_offsets": np.concatenate([[0], np.cumsum(lens)]).astype(np.int64), "lengths": np.array(lens, np.int32), "truncated": np.zeros(len(lens), np.uint8)}, ii, j * 2, j * 2 + 2)
        means += list(m); toks += t; ids += ii
    return means, toks, ids

def test_partial_chunk_is_ignored_and_resume_point(tmp_path):
    d = tmp_path / "shard_1"
    assert store.resume_batch(d) == (0, 0)
    chunkset(d, [[3, 2], [4]])
    (d / ".chunk_00002.safetensors.tmp").write_bytes(b"half written")
    assert [p.name for p in store.chunks(d)] == ["chunk_00000.safetensors", "chunk_00001.safetensors"]
    assert store.resume_batch(d) == (2, 4)

@pytest.mark.parametrize("cap", [10**9, 150])
def test_merge_order_parts_and_lazy_reads(tmp_path, cap):
    d = tmp_path / "c"; means, toks, ids = chunkset(d, [[3, 5], [2], [4, 1, 6]])
    idx = store.merge(store.chunks(d), tmp_path / "out", "s", cap, dict(backend="fake"))
    assert idx["n"] == 6 and set(idx["keys"]) == {"layer_3/mean", "layer_3/tokens", "lengths", "truncated"}
    assert (len(idx["parts"]) == 1) == (cap == 10**9) and idx["parts"][-1]["row_end"] == 6
    assert json.loads((tmp_path / "out" / "s.ids.json").read_text()) == {x: i for i, x in enumerate(ids)}
    rows = list(range(6))
    assert np.allclose(np.stack(store.read_rows(tmp_path / "out" / "s.index.json", rows, "layer_3/mean")), np.stack(means))
    assert all(np.allclose(a, b) for a, b in zip(store.read_rows(tmp_path / "out" / "s.index.json", rows, "layer_3/tokens"), toks))
    assert np.allclose(np.concatenate([b for _, b in store.iter_blocks(tmp_path / "out" / "s.index.json", "layer_3/mean", 4)]), np.stack(means))

def test_merge_refuses_duplicate_ids(tmp_path):
    chunkset(tmp_path / "a", [[2]]); chunkset(tmp_path / "b", [[2]])
    with pytest.raises(ValueError, match="duplicate"): store.merge(store.chunks(tmp_path / "a") + store.chunks(tmp_path / "b"), tmp_path / "o", "s", 10**9, {})

def test_unreadable_last_chunk_is_dropped_on_resume(tmp_path):
    d = tmp_path / "shard_1"; chunkset(d, [[3, 2], [4]])
    (d / "chunk_00002.safetensors").write_bytes(b"renamed but empty after a crash")
    assert store.resume_batch(d) == (2, 4) and len(store.chunks(d)) == 2
