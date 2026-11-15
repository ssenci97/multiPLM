"""ProtT5 / ProstT5 backend (transformers, base env): encoder residue states of the requested layers (layer n_layers = last_hidden_state after the final layer norm, lower layers = raw block outputs); the ProstT5 <AA2fold> prefix, </s> and padding are removed."""
###################################### DEPENDENCIES_AND_CONSTANTS
from pathlib import Path
import re
from types import SimpleNamespace
import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer
from backends import torch_dtype
RARE_RE = re.compile(r"[UZOB]")
PROST_PREFIX = "<AA2fold>"
###################################### HELPER_FUNCTIONS
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
    model = T5EncoderModel.from_pretrained(str(files), torch_dtype=dt, low_cpu_mem_usage=True).to(device).eval()
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
