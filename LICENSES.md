# Licences

**Use rule.** Primary intended use is research. Commercial use is excluded for any model, tool or data whose licence does not explicitly allow it; ESM3-open (`esm3_sm_open_v1`) and ESM C 600M (`esmc_600m`) are non-commercial under the `LICENSE.md` of the pinned esm tag v3.2.1, so they are excluded from commercial use. The licence statements of the ESM3 and ESM C weights conflict (see the second table); this file records sources and dates only, it is not legal advice.

## Software in the MultiPLM image

Model weights are not in the image. They are downloaded at run time into `data/models/` and keep their own licences.

| Component | Version in the image | Licence (as checked 2026-11-14) | Text |
|---|---|---|---|
| EvolutionaryScale `esm` (ESMC, ESM3 code) | git `be185fb05e1aa3ccfda4e7bd10e1d341f81489a2` (tag v3.2.1) | Cambrian Open License Agreement (code), per `LICENSE.md` at that tag | `LICENSES/esm-evolutionaryscale.LICENSE.md` |
| `fair-esm` (ESM2 code) | git `0b59d87ebef95948c735b1f7aad463dc6dfa991b` (tag v2.0.0) | MIT | `LICENSES/fair-esm.LICENSE.txt` |
| MMseqs2 | bioconda `mmseqs2=18.8cc5c`, source tag `18-8cc5c` (`8cc5ce367b5638c4306c2d7cfc652dd099a4643f`) | **MIT** at that tag. Releases up to 15 were GPL-3.0; `LICENSE.md` is MIT from release 16 (checked on tags 13, 15, 16, 17, 18) | `LICENSES/mmseqs2.LICENSE.md` |
| DADApy | PyPI sdist 0.3.4 (sha256 `6907f08b…`), own venv `/opt/envs/dadapy` with jax 0.4.30, flax 0.8.5, numpy 1.26.4 | Apache-2.0; no NOTICE file in the sdist or the repository | `LICENSES/dadapy.LICENSE.txt` (sha256 `43070e2d…`) |
| PyTorch 2.4.1 (cu121 wheels, with NVIDIA CUDA runtime libraries) | pip | BSD-3-Clause; NVIDIA libraries under the NVIDIA EULA | inside the wheels |
| transformers, tokenizers, huggingface_hub, safetensors, sentencepiece | pip | Apache-2.0 | inside the packages |
| conda-forge packages (python, numpy, h5py, pyyaml, tqdm, git, curl) | conda | their own licences | `/opt/conda/envs/mp/conda-meta/` |

## Model weights: the conflict that was found

| Source | What it says | URL | Checked |
|---|---|---|---|
| Hugging Face model cards `biohub/esm3-sm-open-v1`, `biohub/esmc-300m-2024-12`, `biohub/esmc-600m-2024-12`, `biohub/ESMC-6B` | MIT (the ESMC cards also reference the Biohub Acceptable Use Policy); the HF API reports `gated: false` | https://huggingface.co/biohub/esm3-sm-open-v1 (and the other model pages) | 2026-11-14 |
| `LICENSE.md` on the main branch of evolutionaryscale/esm | MIT, Copyright 2026 Chan Zuckerberg Biohub, Inc. | https://github.com/evolutionaryscale/esm/blob/main/LICENSE.md | 2026-11-14 |
| `LICENSE.md` at the **pinned tag v3.2.1** of evolutionaryscale/esm | code and ESM C 300M: Cambrian Open License; **ESM3-open and ESM C 600M: Cambrian Non-Commercial License**; ESM C 6B and other ESM3 models: API only | https://github.com/evolutionaryscale/esm/blob/be185fb05e1aa3ccfda4e7bd10e1d341f81489a2/LICENSE.md | 2026-11-14 |
| fair-esm ESM2 weights | MIT (repository licence) | https://github.com/facebookresearch/esm | 2026-11-14 |
| ProtT5 (`Rostlab/prot_t5_xl_half_uniref50-enc`) | no licence tag on the model card (HF API) | https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc | 2026-11-14 |
| ProstT5 (`Rostlab/ProstT5`) | MIT (model card tag) | https://huggingface.co/Rostlab/ProstT5 | 2026-11-14 |
| UniProtKB data | CC BY 4.0 | https://www.uniprot.org/help/license | — |
| Facco et al. 2017 paper (`docs/references/facco2017/`) | CC BY 4.0 | see REFERENCES.md | 2026-11-14 |
