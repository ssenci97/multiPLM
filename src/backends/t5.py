"""ProtT5 / ProstT5 backend (transformers, base env): encoder residue states of the requested layers (layer n_layers = last_hidden_state after the final layer norm, lower layers = raw block outputs); the ProstT5 <AA2fold> prefix, </s> and padding are removed."""
###################################### DEPENDENCIES_AND_CONSTANTS
from pathlib import Path
import re
from types import SimpleNamespace
import numpy as np
import torch
from transformers import T5Config, T5EncoderModel, T5Tokenizer
from backends import torch_dtype
RARE_RE = re.compile(r"[UZOB]")
PROST_PREFIX = "<AA2fold>"
###################################### HELPER_FUNCTIONS
def read_state(files: Path) -> dict[str, torch.Tensor]:
    """Memory-mapped state dict of the snapshot (safetensors, else pytorch_model*.bin), no accelerate needed."""
    sd = {}
    parts = sorted(files.glob("*.safetensors"))
    if parts:
        from safetensors.torch import load_file
        for f in parts: sd.update(load_file(str(f)))
        return sd
    for f in sorted(files.glob("pytorch_model*.bin")): sd.update(torch.load(f, map_location="cpu", mmap=True, weights_only=True))
    if not sd: raise FileNotFoundError(f"no safetensors or pytorch_model*.bin in {files}")
    return sd
def load_encoder(files: Path, dt: torch.dtype) -> T5EncoderModel:
    """Encoder built on the meta device and filled tensor by tensor in the target dtype (peak memory about the final model, not model plus checkpoint)."""
    with torch.device("meta"): model = T5EncoderModel(T5Config.from_pretrained(str(files)))
    sd = {k: v.to(dt) for k, v in read_state(files).items() if k.startswith("encoder.") or k == "shared.weight"}
    sd.setdefault("shared.weight", sd.get("encoder.embed_tokens.weight")); sd.setdefault("encoder.embed_tokens.weight", sd["shared.weight"])
    missing, _ = model.load_state_dict(sd, strict=False, assign=True)
    model.tie_weights()
    if missing: raise RuntimeError(f"missing weights in {files}: {missing[:5]}")
    return model
def load(spec: dict, files: Path, device: torch.device, dtype: str) -> SimpleNamespace:
    """Load the T5 encoder and its sentencepiece tokenizer from the local snapshot.

    Args:
        spec (dict): Registry row (family prott5 or prostt5).
        files (Path): data/models/<name>/files.
        device (torch.device): Target device.
        dtype (str): auto (fp16 on GPU as recommended by Rostlab, fp32 on CPU), fp32, fp16 or bf16.

    Returns:
        SimpleNamespace: The backend handle.
    """
    prost = spec["family"] == "prostt5"
    tok = T5Tokenizer.from_pretrained(str(files), do_lower_case=False, legacy=True)
    dt, name = torch_dtype(dtype, device, half="fp16")
    model = load_encoder(files, dt).to(device).eval()
    skip = 1 if prost else 0
    n = model.config.num_layers
    def residues(seqs: list[str], layers: list[int]) -> dict[int, list[np.ndarray]]:
        """Per-layer residue matrices of a padded batch."""
        text = [(PROST_PREFIX + " " if prost else "") + " ".join(RARE_RE.sub("X", s)) for s in seqs]
        enc = tok(text, add_special_tokens=True, padding="longest", return_tensors="pt").to(device)
        with torch.inference_mode(): out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, output_hidden_states=layers != [n])
        rep = {k: out.last_hidden_state if k == n else out.hidden_states[k] for k in layers}
        return {k: [rep[k][i, skip:skip + len(s)].float().cpu().numpy() for i, s in enumerate(seqs)] for k in layers}
    return SimpleNamespace(dim=model.config.d_model, n_layers=n, max_length=0, dtype=name, one_at_a_time=False, residues=residues)
