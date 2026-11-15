"""ESM3 backend (EvolutionaryScale esm, esm-evo venv): the locally downloaded esm3-sm-open-v1 weights only, sequence track only, one protein at a time; final-layer residue embeddings with BOS/EOS removed (the forward pass exposes no intermediate layer). No API (Forge/Biohub) access."""
###################################### DEPENDENCIES_AND_CONSTANTS
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import esm.pretrained
import esm.utils.constants.esm3 as esm_constants
from esm.sdk.api import ESMProtein
from backends import torch_dtype
###################################### HELPER_FUNCTIONS
def load(spec: dict, files: Path, device: torch.device, dtype: str) -> SimpleNamespace:
    """Load ESM3-small-open from data/models/esm3_sm_open_v1/files.

    Args:
        spec (dict): Registry row.
        files (Path): Local snapshot of the pinned Hugging Face revision.
        device (torch.device): Target device.
        dtype (str): auto or bf16 (fp32 weights with bf16 autocast where supported, as esm's own logits()), or fp32.

    Returns:
        SimpleNamespace: The backend handle.
    """
    def root(model: str) -> Path:
        """The local snapshot, whatever model name esm asks for."""
        return Path(files)
    esm.pretrained.data_root, esm_constants.data_root = root, root
    model = esm.pretrained.ESM3_sm_open_v0(device)
    _, name = torch_dtype("fp32" if dtype == "fp16" else dtype, device)
    model = model.eval()
    n = len(model.transformer.blocks)
    def residues(seqs: list[str], layers: list[int]) -> dict[int, list[np.ndarray]]:
        """Final-layer residue matrices, one protein at a time."""
        if layers != [n]: raise ValueError(f"esm3 exposes only its last layer ({n}), asked {layers}")
        out = []
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=name == "bf16"):
            for s in seqs:
                t = model.encode(ESMProtein(sequence=s))
                out.append(model(sequence_tokens=t.sequence.unsqueeze(0)).embeddings[0, 1:len(s) + 1].float().cpu().numpy())
        return {n: out}
    return SimpleNamespace(dim=model.encoder.sequence_embed.embedding_dim, n_layers=n, max_length=0, dtype=name, one_at_a_time=True, residues=residues)
