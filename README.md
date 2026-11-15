# MultiPLM

MultiPLM extracts protein language model representations for sets of proteins.

- **Backends:** fair-esm ESM2, ESM3-small-open, ESMC 300M/600M/6B, ProtT5 and ProstT5.
- **What you get:** any layer, either the masked mean over real residues or the per-residue vectors, stored as safetensors.
- **Where it runs:** everything runs in one pinned CUDA image on SLURM.
- **Weights:** downloaded once, pinned by revision and checked by sha256 before every use.
- **Optional extras:** MMseqs2 deduplication before embedding, and summary aggregates computed after it. The aggregates include TWO-NN intrinsic dimension (Facco et al. 2017, computed with DADApy) and a joint PCA.

The full protocol is in [MANUAL.md](MANUAL.md). Primary intended use is research; commercial use is excluded for anything whose licence does not explicitly allow it, including ESM3-open and ESM C 600M: see [LICENSES.md](LICENSES.md).

## Inputs and parameters
- **`configs/samples.tsv`**, one row per sample, with columns:
  - `sample_id`;
  - either `fasta_path` (an absolute path) or `uniprot_query` (a query, or `accessions:…`, downloaded beforehand with `python3 src/prefetch.py` on a login node);
  - `description`.
- **`configs/param_configs.yaml`**, with these sections:
  - `embed`: backends, layers (`[-1]` or `all`), pooling (`mean` / `tokens`), `store_dtype`, size guard and sharding;
  - `dedup`;
  - `summarize`: settings for TWO-NN and PCA;
  - `slurm`.
- **`configs/tools.tsv`** holds every pin: software git SHAs, MMseqs2, and Hugging Face revisions with dimensions and layer counts.

## Quick start (SLURM)
```bash
srun -p THIN --mem=16G -c 4 --time=01:00:00 containers/build.sh pull     # pull the image by digest, then smoke and unit tests
python3 src/analysis.py kg --example example_kinase_gpcr                # freeze 2 x 250 Swiss-Prot proteins
python3 src/pipeline.py kg --dry-run                                    # 15 jobs, size estimate 0.062 GB
python3 src/pipeline.py kg --online-here                                # download the weights here, submit the rest
```

## Console
Run `python3 src/gui_serve.py` on the login node and open it through `ssh -J <user>@<login> -L 8000:127.0.0.1:8000 -N <user>@<node>`.

| Tab | What it does |
|---|---|
| Samples | Schema-driven sample cards (Local FASTA or UniProtKB query) and read-only examples |
| Run | Pick backends (each shows its GPU ask and weights status), see the size estimate, launch and stop, and follow the steps, the queue and the node and GPU lane choices |
| Results | A sample × backend table and per-sample records. Aggregates per backend and layer: norms, centroid cosines, TWO-NN d by layer and d(N), and PCA hexbin densities with explained variance, PC1-vs-length and nearest-centroid confusion. Also the weights used and the provenance. Raw vectors never reach the browser |
| Performance | A dark page opened from the top-right pill. Shows partitions and nodes with CPUs and GPUs, and marks our running jobs with ◆ and amber stripes. Also shows efficiency, waits, OOM and timeouts, core-hours, GPU-hours, and throughput per backend |

No state is shown by colour alone: each one has a glyph and a word (✓ ✕ ▶ ■ ○). Bars are textured, series use a fixed CVD-safe palette, and line markers have distinct shapes. Both themes are supported, and keyboard focus is visible.

## SLURM use
- There is one job per sample and step, and one per shard for embedding. Jobs are chained with `--dependency=afterok`.
- All jobs go through `src/mp_job.sh`, which runs `singularity exec --cleanenv`. GPU jobs also get `--nv`.
- Resources per job:
  - CPU jobs: 1 core and 1 GB;
  - embedding jobs: 2 cores and 4 to 16 GB depending on the backend.

  The user approved these asks; they are listed in MANUAL §11.
- An analysis uses 1 GPU at a time by default and at most 2, with never more than one DGX job at a time.

## Pipeline
`00_weights` → `01_prepare` → `02_dedup` (optional) → `03_embed` (GPU shards) → `04_merge` → `05_summary`. Files on disk decide what is done, so rerunning resumes. A killed shard continues from its last complete chunk.

## Containers
- One image holds:
  - a base conda env `mp` (torch 2.4.1 cu121, transformers, safetensors, h5py, MMseqs2 18-8cc5c);
  - the venvs `esm-fair` and `esm-evo`, because fair-esm and EvolutionaryScale `esm` both own the top-level `esm` module;
  - an isolated venv `dadapy` (DADApy 0.3.4 with jax 0.4.30 and numpy < 2), kept apart from torch.
- The base image is pinned by digest. The tools are pinned by git SHA, and the build asserts each SHA after checkout.
- `containers/build.sh` modes: `build`, `lock`, `login`, `push`, `ship`, `pull`, `test [--gpu]`.

## Reproducibility
- The inputs, the frozen UniProt FASTA and `tools.tsv` are recorded with sha256 when an analysis is created.
- The sha256 of every weight file is checked before every shard and stored in the provenance.
- Archive `data/models/` after the first download (MANUAL §14).

## Example datasets
| Example | Content |
|---|---|
| `example_toy` | 2 × 5 human proteins; `esmc_300m`, last layer |
| `example_kinase_gpcr` | 2 × 250 human Swiss-Prot kinases and GPCRs (UniProtKB 2026_04, CC BY 4.0); `esm2_t6_8M` + `esmc_300m`, all layers; dedup, TWO-NN and PCA on |

## Citation and credits
- The intrinsic dimension is computed with **DADApy** (Glielmo et al. 2022, *Patterns* 3:100589, doi:10.1016/j.patter.2022.100589; Apache-2.0), which implements TWO-NN from **Facco E, d'Errico M, Rodriguez A, Laio A (2017) Sci Rep 7:12140, doi:10.1038/s41598-017-11873-y**. Thank you to the DADApy authors and the Laio group. The paper and its SI are in `docs/references/facco2017/` (CC BY 4.0).
- Model and tool references: [REFERENCES.md](REFERENCES.md).
- Extraction logic adapted from SCambrian.
- Driver, job wrapper and console ported from KNOCKIN-VAL.

## Development
- Tests: `nice -n 19 python3 -m pytest -q tests`. 64 tests; they need no cluster, no image and no GPU, and use fake SLURM, MMseqs2, backend and a stub DADApy. The real-DADApy benchmarks run in the image with `containers/build.sh test`.
- Style rules: `.claude/agents/simple-code-reviewer.md`.
- Repository rules for Claude Code: `CLAUDE.md`.
