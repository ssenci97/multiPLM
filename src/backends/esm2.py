"""ESM2 backend (fair-esm, esm-fair venv): residue representations of the requested layers, BOS/EOS/padding removed (layer n_layers is the final, layer-normed output)."""
###################################### DEPENDENCIES_AND_CONSTANTS
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import esm
from backends import torch_dtype
MAX_LENGTH = 1022
###################################### HELPER_FUNCTIONS
def load(spec: dict, files: Path, device: torch.device, dtype: str) -> SimpleNamespace:
    """Load an ESM2 checkpoint and its contact-regression file from the local weights folder.

    Args:
        spec (dict): Registry row; files[0] is the model .pt.
        files (Path): data/models/<name>/files.
        device (torch.device): Target device.
        dtype (str): auto (fp32), fp32 or fp16.

    Returns:
        SimpleNamespace: The backend handle.
    """
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(files / spec["files"][0]))
    dt, name = torch_dtype("fp32" if dtype == "auto" else dtype, device)
    model = model.eval().to(device=device, dtype=dt)
    convert = alphabet.get_batch_converter()
    def residues(seqs: list[str], layers: list[int]) -> dict[int, list[np.ndarray]]:
        """Per-layer residue matrices of a padded batch."""
        _, _, toks = convert([(str(i), s) for i, s in enumerate(seqs)])
        with torch.inference_mode(): reps = model(toks.to(device), repr_layers=layers)["representations"]
        return {k: [reps[k][i, 1:len(s) + 1].float().cpu().numpy() for i, s in enumerate(seqs)] for k in layers}
    return SimpleNamespace(dim=model.embed_dim, n_layers=model.num_layers, max_length=MAX_LENGTH, dtype=name, one_at_a_time=False, residues=residues)
