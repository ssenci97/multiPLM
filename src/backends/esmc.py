"""ESMC backend (EvolutionaryScale esm, esm-evo venv): 300m/600m through the esm builders, 6B shard by shard; residue representations of the requested layers, BOS/EOS/padding removed (layer n_layers is the final normed output, lower layers are raw block outputs)."""
###################################### DEPENDENCIES_AND_CONSTANTS
import gc
import json
from pathlib import Path
import re
from types import SimpleNamespace
import numpy as np
from safetensors.torch import load_file
import torch
import esm.pretrained
import esm.utils.constants.esm3 as esm_constants
from esm.models.esmc import ESMC
from esm.tokenization import get_esmc_model_tokenizers
from backends import torch_dtype
BUILDERS = dict(esmc_300m="ESMC_300M_202412", esmc_600m="ESMC_600M_202412")
HF_BLOCK_KEYS = {
    "input_layernorm.weight": "attn.layernorm_qkv.0.weight",
    "input_layernorm.bias": "attn.layernorm_qkv.0.bias",
    "self_attn.q_norm.weight": "attn.q_ln.weight",
    "self_attn.k_norm.weight": "attn.k_ln.weight",
    "self_attn.o_proj.weight": "attn.out_proj.weight",
    "post_attention_layernorm.weight": "ffn.0.weight",
    "post_attention_layernorm.bias": "ffn.0.bias",
    "mlp.down_proj.weight": "ffn.3.weight",
}
HF_FUSED_KEYS = {
    "attn.layernorm_qkv.1.weight": ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"),
    "ffn.1.weight": ("mlp.gate_proj.weight", "mlp.up_proj.weight"),
}
HF_ROOT_KEYS = {
    "esmc.embed_tokens.weight": "embed.weight",
    "esmc.norm.weight": "transformer.norm.weight",
    "lm_head.dense.weight": "sequence_head.0.weight",
    "lm_head.dense.bias": "sequence_head.0.bias",
    "lm_head.layer_norm.weight": "sequence_head.2.weight",
    "lm_head.layer_norm.bias": "sequence_head.2.bias",
    "lm_head.decoder.weight": "sequence_head.3.weight",
    "lm_head.decoder.bias": "sequence_head.3.bias",
}
HF_BLOCK_RE = re.compile(r"^esmc\.layers\.(\d+)\.(.+)$")
###################################### HELPER_FUNCTIONS
def local_root(files: Path) -> None:
    """Point the esm weight lookups at the local weights folder instead of a Hugging Face download."""
    def root(model: str) -> Path:
        """The local snapshot, whatever model name esm asks for."""
        return Path(files)
    esm.pretrained.data_root = root
    esm_constants.data_root = root
def translate(shard: dict[str, torch.Tensor], pending: dict[str, dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Map one published 6B shard onto ESMC parameter names; q/k/v and gate/up halves wait in pending until complete.

    Mirrors the layout translation of SCambrian src/extract_reps_biohub.py (esm 3.2.x nn.Sequential indices).

    Args:
        shard (dict[str, torch.Tensor]): Published tensors of one safetensors file.
        pending (dict[str, dict[str, torch.Tensor]]): Unfused leaves per block, carried across shards.

    Returns:
        dict[str, torch.Tensor]: ESMC state-dict entries completed by this shard.
    """
    out = {}
    for key, tensor in shard.items():
        if key in HF_ROOT_KEYS: out[HF_ROOT_KEYS[key]] = tensor; continue
        m = HF_BLOCK_RE.match(key)
        if m is None: out[key] = tensor; continue
        block, leaf = m.groups()
        if leaf in HF_BLOCK_KEYS: out[f"transformer.blocks.{block}.{HF_BLOCK_KEYS[leaf]}"] = tensor
        else: pending.setdefault(block, {})[leaf] = tensor
    for block, leaves in list(pending.items()):
        for fused, parts in HF_FUSED_KEYS.items():
            if all(p in leaves for p in parts): out[f"transformer.blocks.{block}.{fused}"] = torch.cat([leaves.pop(p) for p in parts], dim=0)
        if not leaves: del pending[block]
    return out
def load_6b(files: Path, device: torch.device) -> ESMC:
    """Build ESMC-6B on the device and fill it shard by shard, failing on any missing or unexpected key."""
    cfg = json.loads((files / "config.json").read_text())
    with torch.device(device): model = ESMC(d_model=cfg["hidden_size"], n_heads=cfg["num_attention_heads"], n_layers=cfg["num_hidden_layers"], tokenizer=get_esmc_model_tokenizers(), use_flash_attn=False).eval()
    want, seen, pending, extra = set(model.state_dict()), set(), {}, []
    for shard_path in sorted(files.glob("*.safetensors")):
        part = translate(load_file(str(shard_path)), pending)
        extra += [k for k in part if k not in want]
        model.load_state_dict({k: v for k, v in part.items() if k in want}, strict=False)
        seen |= set(part) & want
        del part
        gc.collect()
    missing = sorted(want - seen)
    if missing or extra or pending: raise RuntimeError(f"[PYTHON-INFO] !! ESMC-6B checkpoint mismatch: missing={missing[:5]} unexpected={extra[:5]} unfused={list(pending)[:5]}")
    return model
def load(spec: dict, files: Path, device: torch.device, dtype: str) -> SimpleNamespace:
    """Load ESMC 300m/600m/6B from the local weights folder.

    Args:
        spec (dict): Registry row.
        files (Path): data/models/<name>/files.
        device (torch.device): Target device.
        dtype (str): auto (bf16 where supported, else fp32), fp32, fp16 or bf16.

    Returns:
        SimpleNamespace: The backend handle.
    """
    local_root(files)
    model = load_6b(files, device) if spec["family"] == "esmc6b" else getattr(esm.pretrained, BUILDERS[spec["name"]])(device=device, use_flash_attn=False)
    dt, name = torch_dtype(dtype, device)
    model = model.to(dt).eval()
    pad = model.tokenizer.pad_token_id
    n = len(model.transformer.blocks)
    def residues(seqs: list[str], layers: list[int]) -> dict[int, list[np.ndarray]]:
        """Per-layer residue matrices of a padded batch."""
        with torch.inference_mode():
            toks = model._tokenize(seqs)
            out = model(sequence_tokens=toks, sequence_id=toks != pad)
        rep = {k: out.embeddings if k == n else out.hidden_states[k - 1] for k in layers}
        return {k: [rep[k][i, 1:len(s) + 1].float().cpu().numpy() for i, s in enumerate(seqs)] for k in layers}
    return SimpleNamespace(dim=model.embed.embedding_dim, n_layers=n, max_length=0, dtype=name, one_at_a_time=False, residues=residues)
