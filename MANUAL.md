# MultiPLM manual

Status on 2026-11-15: the code, the tests and the console are written and checked locally, without a GPU, an image or a cluster. **No real backend has been run yet** (§15). Commercial use is restricted (§14).

## 1. What this is

| Item | Content |
|---|---|
| Purpose | Protein language model representations of FASTA samples, any layer, mean-pooled or per residue, stored as safetensors |
| Backends | fair-esm ESM2 (8M–3B), ESM3-small-open, ESMC 300M/600M/6B, ProtT5-XL-half, ProstT5 (§2) |
| Extras | optional MMseqs2 deduplication; summary aggregates with TWO-NN intrinsic dimension (computed with DADApy) and an optional joint PCA, and t-SNE/UMAP only below 1000 points (never subsampled) |
| How it runs | Every job runs inside one pinned image (Docker → Singularity `.sif`) as a plain SLURM job. `src/pipeline.py` decides what to run from the files on disk; there is no workflow engine. GPU jobs form their own lane (§11) |
| Origin | Extraction logic from SCambrian (ESMC-6B key translation, UniProt paging). Driver, job wrapper and console ported from KNOCKIN-VAL `2263299`. TWO-NN (Facco et al. 2017) through DADApy (§18) |

## 2. Backends and layers

Pins come from `configs/tools.tsv`. Revisions and sizes were read from the Hugging Face API on 2026-11-14.

| Backend | Env | Weights (pinned revision) | Size | Dim | Layers | Partition |
|---|---|---|---|---|---|---|
| `esm2_t6_8M` … `esm2_t36_3B` | `esm-fair` | fair-esm URLs (code `0b59d87…`); bytes pinned by `configs/checksums/` after the first download | 0.03–5.7 GB | 320–2560 | 6–36, all | GPU |
| `esm3_sm_open_v1` | `esm-evo` | `biohub/esm3-sm-open-v1@47f0545…` | 5.5 GB | 1536 | 48, **last only** | GPU |
| `esmc_300m` / `esmc_600m` | `esm-evo` | `biohub/esmc-300m-2024-12@7f10b20…` / `esmc-600m-2024-12@e4d83bc…` | 1.33 / 2.3 GB | 960 / 1152 | 30 / 36, all | GPU |
| `esmc_6b` | `esm-evo` | `biohub/ESMC-6B@9460559…` (fp32 safetensors, loaded shard by shard) | 25.4 GB | 2560 | 80, all | DGX |
| `prott5_xl_half` | `mp` | `Rostlab/prot_t5_xl_half_uniref50-enc@94a6abc…` | 2.42 GB | 1024 | 24, all | GPU |
| `prostt5` | `mp` | `Rostlab/ProstT5@d7d097d…` | 11.3 GB | 1024 | 24, all | GPU |

| Convention | Rule |
|---|---|
| Layer numbers | `1..n` are the outputs of blocks 1..n; `-1` = `n`. Layer `n` is what each library returns as its final representation: ESM2 after the final layer norm, ESMC after `transformer.norm`, T5 `last_hidden_state` after the final layer norm, but **ESM3 before its final norm** (esm 3.2.1 passes the pre-norm output as `embeddings`). Lower layers are raw block outputs. `'all'` = 1..n. ESM3's public forward returns only its final embeddings |
| Residues kept | Only real residues; special tokens and padding are removed for every backend. ESM2, ESMC, ESM3: BOS (`<cls>`) and EOS dropped. ProtT5: the trailing `</s>` dropped (sequences are space-separated, U/Z/O/B → X). ProstT5: the `<AA2fold>` prefix (amino-acid → 3Di direction) and `</s>` dropped; inputs are upper-case amino acids (lower case would mean 3Di) |
| Mean | `pooling: mean` = masked mean over those real residues. `tokens` = the residue matrix, written chunk by chunk and never held whole in RAM. Both can be asked for |
| Length | ESM2 is truncated to its first 1022 residues: ESM2 uses rotary positions, so this is its training crop (1024 tokens), not a hard limit; a truncated protein's mean covers only those residues. The others are cut only at `embed.max_length`. `lengths` keeps the original length and `truncated` flags the protein. Proteins sharing a 1022-residue prefix get identical ESM2 vectors (§12, duplicates) |
| Precision | compute `dtype: auto` = bf16 where the GPU supports it natively (A100/H100; emulated bf16 on V100 is not used), else fp32 (ESMC; ESM3 through autocast); fp16 for T5 on GPU; fp32 on CPU. Stored as `store_dtype` float32 (default) or float16 |
| Last-layer jump | for ESM2, ESMC and T5 the last layer is LayerNorm-ed and the others are not; LayerNorm acts per vector, so norms, neighbourhoods, TWO-NN d and PCA can change abruptly at the last layer. Read that step as a normalisation effect, not biology |
| Local weights only | No API (Forge/Biohub) backend. If one is ever added, it must be an explicit opt-in backend, flagged non-reproducible in the provenance, and never used by an example. Only `esm3-sm-open-v1` has downloadable ESM3 weights |

## 3. Requirements

| Item | Requirement |
|---|---|
| Cluster | SLURM, Singularity or Apptainer (`module load singularity`), GPU partitions |
| Host python | python3 ≥ 3.9 with PyYAML, used by the driver, the console and `prefetch.py` (all outside the image) |
| Disk | about 20 GB for the image pull; weights as in §2; embeddings as estimated at submission (§9) |
| Workstation (build only) | Docker or Podman, about 15 GB free |
| Network | the weights jobs and `prefetch.py` need internet. Compute-node internet on ORFEO is not documented: use `--online-here` and `prefetch.py` on a login node |

ORFEO facts from orfeo-doc.areasciencepark.it (read 2026-11-14):

| Fact | Value |
|---|---|
| Partitions | EPYC, THIN, GENOA (CPU); GPU: 4 nodes, 2× V100 32 GB, 18 h limit; DGX: 2 nodes, 8× A100 40 GB, access not granted by default; H100 |
| GPU requests | `--gres=gpu:<type>:N` with type V100, A100 or H100 (a bare `--gpus=N` is rejected); this project asks for A100 |
| Default partition | none: always pass `-p` |
| CUDA | module `cuda/12.1`; DGX needs no `module load cuda` |
| Not documented | singularity module name, internet from compute nodes, NVIDIA driver version |

## 4. Layout

| Path | Content |
|---|---|
| `configs/tools.tsv` | software pins (git SHAs, MMseqs2, DADApy sdist sha256) and weight pins (HF revisions, dim, layer count) |
| `configs/checksums/<backend>.sha256` | tracked reference checksums (you add them, §14) |
| `configs/param_configs.yaml` | one section per script: `prefetch`, `prepare`, `dedup`, `embed`, `merge`, `summarize`, `slurm`, `gui_serve` |
| `configs/examples/` | `example_toy` (2 × 5 proteins, `esmc_300m`, defaults) and `example_kinase_gpcr` (2 × 250 Swiss-Prot proteins, `esm2_t6_8M` + `esmc_300m`, all layers, dedup, TWO-NN and PCA on) and `example_chaperones` (3 Swiss-Prot UniProtKB queries downloaded when you click Run: small heat-shock proteins PF00011, HSP70 PF00012 and HSP90 PF00183 of eukaryotes; `esmc_300m`, all layers, dedup, TWO-NN and PCA on; at most 500 entries per sample, 50–1022 residues kept) |
| `src/` | steps `00_`–`05_*.sh`, `mp_job.sh`, `common.sh`; Python: `analysis`, `pipeline`, `prefetch`, `weights`, `prepare`, `dedup`, `embed`, `merge`, `summarize`, `intrinsic_dim` (runs in the dadapy env), `store`, `sample_meta`, `gui_serve`, `backends/` |
| `data/models/<backend>/` | `files/`, `CHECKSUMS`, `SOURCE.json`, shared by all analyses |
| `data/uniprot/<sample>/` | prefetched FASTA + provenance JSON |
| `docs/references/facco2017/` | the TWO-NN paper and its Supplementary Information (CC BY 4.0) |

```
results/<analysis>/
  inputs/      samples.tsv params.yaml tools.tsv samples/ uniprot/ analysis.json (sha256 of every file)
  fasta/       <s>.fasta (validated) .tsv (status per record) .json (counts, sha256, UniProt provenance)
  dedup/       <s>.fasta (representatives) .clusters.tsv (representative -> members) .dedup.json
  embeddings/<backend>/
               <s>.safetensors (or <s>-0000k-of-0000N.safetensors)  <s>.index.json  <s>.ids.json  <s>.json (provenance)
               <s>.shards/shard_<k>.done.json  (chunk files are removed after a successful merge)
  meta/        <sample>.json  _analysis.json   (steps, perf records, image, pins, weight sha256)
  logs/        00_weights/ 01_prepare/ 02_dedup/ 03_embed/<b>/ 04_merge/<b>/ 05_summary/ node_pick.log
  summary.tsv  samples_summary.tsv  intrinsic_dimension.tsv  aggregates.json  jobs.tsv  run.log
```

## 5. Conventions

| Convention | Rule |
|---|---|
| Container | step scripts refuse to run outside the image; there is no native mode |
| Done | outputs non-empty (JSON outputs must parse), the log ends with the footer, no output older than its inputs. Weights count as done only for the pinned revision. Rerunning resumes |
| Atomic writes | `.tmp` (or a hidden `.name.tmp`) then rename, everywhere; a killed shard resumes from its last complete chunk |
| Frozen inputs | editing a frozen file, `configs/tools.tsv` or a local FASTA refuses the launch: create a new analysis |
| Tags | `[SLURM-INFO]`, `[PYTHON-INFO]`, `[BASH-INFO]` |
| Seed | `SEED=42` everywhere (subsamples, pairs, PCA; TWO-NN itself is deterministic) |
| Never | Snakemake or other engines, job arrays, a console inside the container, tokens in logs, raw vectors in the browser |

## 6. Install

| Step | Where | Command |
|---|---|---|
| 1 | workstation | `containers/build.sh build` |
| 2 | workstation | `containers/build.sh lock`, then `containers/build.sh build` again (from the lock files) |
| 3 | workstation | `MP_OWNER=<github user> MP_GITHUB_TOKEN=… containers/build.sh ship` (writes `containers/images.env`; commit it) |
| 4 | cluster | `srun -p THIN --mem=16G -c 4 --time=01:00:00 containers/build.sh pull` |
| 5 | GPU node | `srun -p DGX --gres=gpu:A100:1 -n 1 -c 1 --mem=4G --time=00:15:00 containers/build.sh test --gpu` |

## 7. Inputs and settings

| `samples.tsv` column | Meaning |
|---|---|
| `sample_id` | letters, digits, underscore; unique |
| `fasta_path` | absolute path to a `.fa .faa .fasta .fas` file (examples may use paths relative to their folder) |
| `uniprot_query` | instead of a FASTA: a UniProtKB query, or `accessions:P69905,P68871,…`; the console downloads it on its own host (`prefetch.py`) when you click Run, before the inputs are frozen; already downloaded queries are reused |
| `description` | free text |

| Key | Default | Meaning |
|---|---|---|
| `prefetch.reviewed_only` | false | add `reviewed:true` (Swiss-Prot) to every query |
| `prefetch.max_entries` | 0 | refuse a sample whose query announces more entries, after the first page and before anything is saved (0: only the hard cap below; it can only lower the hard cap). UniProt counts change between releases, so a query near the limit can start failing later |
| `dedup.enabled`, `scope` | false, sample | MMseqs2 dedup, per sample or across samples (`joint`) |
| `dedup.mode`, `min_seq_id`, `coverage`, `cov_mode` | easy-linclust, 0.95, 0.8, 0 | MMseqs2 settings (`easy-cluster` also allowed) |
| `embed.backends` | `[esmc_300m]` | backends to run (several allowed) |
| `embed.layers` | `[-1]` | layer list (positive or negative) or `all`, checked against each backend's layer count |
| `embed.pooling` | `[mean]` | `mean`, `tokens` or both; a list for every layer, or a mapping layer → list |
| `embed.store_dtype` | float32 | or float16 |
| `embed.shard_proteins` | 25000 | proteins per shard job |
| `embed.flush_proteins`, `flush_mb` | 2000, 256 | write a chunk after this many proteins or MB |
| `embed.size_warn_gb`, `size_cap_gb` | 50, 500 | warn above, refuse above unless `--allow-large` |
| hard protein caps | 100 000 per sample, 200 000 per analysis | not a parameter: `MAX_PROTEINS_SAMPLE` and `MAX_PROTEINS_ANALYSIS` in `src/analysis.py`. `prefetch.py` refuses a UniProt query above the sample cap before downloading anything; `pipeline.py` (also `--dry-run` and the console size line) refuses a sample or an analysis above them. `--allow-large` does not apply |
| `embed.batch_tokens`, `max_length`, `dtype`, `verify_weights` | 4096, 0, auto, full | padded tokens per batch; truncation; compute precision; sha256 of the weights before every shard |
| `merge.max_part_gb`, `keep_chunks` | 20, false | row-split the final file above this size; keep chunk files |
| `summarize.twonn.*` | enabled false; max_points 10000; mu_fraction 0.9; n_min 20; metric euclidean; tokens true | §12 |
| `summarize.pca.*` | enabled false; per_sample_cap 1000; hex_bins 30; per_sample_pca false; max_components 50 | §13 |

## 8. Steps

| Step | Unit | Does | Outputs |
|---|---|---|---|
| 00_weights | backend (shared) | download the pinned revision; sha256 of every file; compare with any existing `CHECKSUMS` and with `configs/checksums/<b>.sha256`; **any difference stops the job** | `data/models/<b>/CHECKSUMS`, `SOURCE.json` |
| 01_prepare | sample | read the local or frozen UniProt FASTA; upper-case, strip `*`; reject empty records, bad letters, duplicate ids and out-of-bound lengths | `fasta/<s>.{fasta,tsv,json}` |
| 02_dedup (optional) | sample, or one job (joint) | `mmseqs easy-linclust|easy-cluster --threads <job cores>`; representatives keep their headers; the cluster table maps every member; the MMseqs2 version is recorded | `dedup/<s>.{fasta,clusters.tsv,dedup.json}` |
| 03_embed | sample × backend × shard (GPU) | check the weights; proteins sorted by length, batched under `batch_tokens`, batches dealt round-robin to shards; producer thread → GPU → single writer thread flushing atomic chunks | `<s>.shards/shard_<k>/chunk_*.safetensors`, `shard_<k>.done.json` |
| 04_merge | sample × backend (CPU) | all shards must agree on input, revision, settings and weight checksums; stream the chunks into the final file(s), one chunk in RAM at a time | `<s>.safetensors` (or parts), `<s>.index.json`, `<s>.ids.json`, `<s>.json` |
| 05_summary | analysis | tables and aggregates (§12, §13) | `summary.tsv`, `samples_summary.tsv`, `intrinsic_dimension.tsv`, `aggregates.json` |

## 9. Storage format

| Item | Content |
|---|---|
| Tensors | `layer_<k>/mean` (proteins × dim); `layer_<k>/tokens` (residues × dim), where protein i's rows are `token_offsets[i]:token_offsets[i+1]`; `lengths` (original lengths); `truncated` |
| Header metadata | sample, backend, revision, input sha256, layers, pooling, store and compute dtype, max length, row range |
| `<s>.index.json` | number of proteins, keys (dtype, shape per row, per protein or per residue), parts with row ranges, metadata |
| `<s>.ids.json` | `{protein id: row}`. **Rows are not in FASTA order** (length-sorted batches dealt to shards); always map ids through this file |
| `<s>.json` | provenance: everything above plus the sha256 of every weight file, shards (proteins, chunks, time, GPU) and truncation count |
| Size estimate | proteins × (mean: dim; tokens: residues × dim) × layers × bytes, printed per sample and backend by `--dry-run` and at submission |

```python
import json
from safetensors import safe_open
idx = json.load(open("results/<a>/embeddings/esmc_300m/<s>.index.json"))
part = idx["parts"][0]
with safe_open(f"results/<a>/embeddings/esmc_300m/{part['file']}", framework="numpy") as f:
    first100 = f.get_slice("layer_30/mean")[0:100]
```

`src/store.py` also has `read_rows` (any rows, any part) and `iter_blocks`.

## 10. Running

| Task | Command |
|---|---|
| Download UniProt samples (login node) | automatic when you click Run in the console; by hand: `python3 src/prefetch.py` (or `--samples configs/examples/<x>/samples.tsv --yaml-in configs/examples/<x>/param_configs.yaml`) |
| Create from the saved sheet / an example | `python3 src/analysis.py <name>` / `python3 src/analysis.py <name> --example example_kinase_gpcr` |
| See what would run, with the size estimate | `python3 src/pipeline.py <name> --dry-run` |
| Submit | `python3 src/pipeline.py <name>` (`--allow-large` above the size cap; `--online-here` to download the weights on this host first) |
| Stop | `python3 src/pipeline.py <name> --stop` |
| Console | `python3 src/gui_serve.py`, then `ssh -J <user>@<login> -L 8000:127.0.0.1:8000 -N <user>@<node>`, then http://127.0.0.1:8000/ |
| Hugging Face token (only if a repository becomes gated) | `~/.cache/huggingface/token` (mode 600) or `MP_HF_TOKEN_FILE`; only download jobs see it, as a read-only file |

## 11. Resources and GPU guidance

| Job | Cores | Memory | Time | Reason |
|---|---|---|---|---|
| 00_weights, 01_prepare, 02_dedup, 04_merge | 1 | 1 GB | 120 / 30 / 60 / 60 min | download / parsing / MMseqs2 with `--threads 1` / streaming merge (one chunk in RAM): default 1/1 |
| 05_summary | 1 | total size of the reps it loads (minimum 1 GB), **+ 5 GB** when intrinsic dimension is on | 120 min | the jax/DADApy runtime of the TWO-NN step needs a fixed allowance; with TWO-NN off the default stays 1 GB. Past runs (sacct) and OOM doubling still apply on top |
| 03_embed esm2_t6/t12/t30 | 2 | 4 GB | 60–120 min | 2 cores: producer and writer threads beside the GPU loop; memory: CUDA/torch runtime (~3 GB) + checkpoint while loading; chunk buffers ≤ `flush_mb`, independent of set size |
| 03_embed esmc_300m | 2 | 6 GB | 120 min | as above |
| 03_embed esm2_t33_650M, esmc_600m, esm3, prott5 | 2 | 8 GB | 240–480 min | as above (2.3–5.5 GB checkpoints) |
| 03_embed esm2_t36_3B, prostt5, esmc_6b | 2 | 16 GB | 240–720 min | ProstT5 11.3 GB and ESM2 3B 5.7 GB checkpoints; 6B loads one 5 GB shard at a time |

The user approved these asks (2026-11-14). After the first runs, memory and time are learned from `sacct` (OOM or timeout doubles the ask).

| Guidance | Value |
|---|---|
| CPU lane | one CPU node from `node_pick.sh`, at most `node_fraction` (0.5) of it; above 0.5 is refused |
| GPU lane (pragmatic default) | 1 GPU at a time per analysis (`slurm.gpus_max: 1`); exceptionally 2, never 3+; never more than one DGX job at a time. GPUs are the scarcest shared resource; justified exceptions are agreed with the user. Default lane since 2026-11-15: `slurm.gpu_partition: DGX` with `--gres=gpu:A100:1`, so every GPU job takes the single DGX token and they run one at a time (the V100s on GPU are not used) |
| Shards | one GPU job per (sample, backend, shard); the GPU tokens serialise them (two at a time with `gpus_max: 2`) |
| Limits | ≤ 100 jobs per launch, ≤ 200 in your queue; one launch per analysis; another analysis downloading the same weights blocks the launch |

## 12. Intrinsic dimension (TWO-NN with DADApy)

| Item | Detail |
|---|---|
| Engine | **DADApy 0.3.4** (PyPI sdist, sha256 `6907f08b…`, Apache-2.0), in its own venv `/opt/envs/dadapy` without system site-packages (it pins numpy < 2 and jax 0.4.30). `src/summarize.py` writes the point sets as `.npy` files and runs `src/intrinsic_dim.py` with that env's interpreter; it stops at once if the env is missing |
| Method (Facco et al. 2017, p.3) | μ = r2/r1 per point; F_emp(μ_(i)) = i/N over all N points; fit −log(1 − F) = d·log(μ) through the origin, keeping the lowest `mu_fraction` (0.9) of μ with full-N ranks. DADApy does this with a least-squares fit (scipy `curve_fit`) |
| Scale curve | DADApy `return_id_scaling_2NN`: d on N, N/2, N/4 … down to `n_min`, each from random subsets (not disjoint partitions as in our earlier code), error = standard error of the mean. Read it as a curve; a plateau is the relevant dimension; no plateau is declared automatically |
| Our additions | seeded cap (`max_points`); exact duplicate rows removed and **counted** (DADApy itself would give r1 = 0, μ = ∞), warning above 10%; cosine = L2-normalised rows with Euclidean distance; fewer than 20 points → no estimate. Engine version, settings and these differences are stored with every estimate |
| Where | every extracted `(backend, layer)` mean representation, per sample and on all samples together (balanced cap). Tokens: residues as points, at most `max_points`, stratified by protein, flagged |
| Caveats | assumes only **local** uniform density; estimates the tangent-space dimension. Sharp boundaries and Gaussians make TWO-NN **underestimate** (SI p.4: about 1.5 at d = 10 and 4 at d = 20 for a hypercube without periodic boundaries; the main text p.4 says "overestimation", the SI numbers are used here). Heavy tails need the 10% discard (Cauchy d = 20: 22.16 kept vs 6.05 for the paper's single full fit, which does not reproduce reliably). Noise inflates d at large N; small N underestimates with large errors |
| Outputs | `intrinsic_dimension.tsv`; `aggregates.json → twonn`; Results tab: d by layer, d at relative depth across backends, d(N) |
| Checks | host tests use a stub DADApy (wiring only). The paper's benchmarks with the real DADApy (hypercube d = 14 with periodic boundaries, Swiss roll, Cauchy d = 20 median of 5 draws, `mu_fraction` 0.80–0.95 agreement) run inside the image: `containers/build.sh test` |

## 13. Other aggregates (Results tab)

| Aggregate | Computed in 05_summary from |
|---|---|
| Length histogram, status table, MMseqs2 counts | `fasta/*.tsv`, `dedup/*.json` |
| Norm histogram, mean norm by length, centroid cosines, within- vs between-sample cosine (2000 random pairs) | the last layer with mean pooling |
| Joint PCA per (backend, layer) | fit on a balanced subsample (`per_sample_cap` per sample), **every** vector projected in blocks, sent only as hexbin densities per sample on shared axes and a shared colour scale (0.5–99.5% box, clipped points counted); cumulative explained variance with the components for 90%; PC1 versus length (Pearson r, warning above 0.5); nearest-centroid confusion and within-sample spread. Optional per-sample PCA on its own axes |
| Throughput | proteins per GPU-job hour per backend (Performance page) |

## 14. Long-term reproducibility and licences

| Risk | What to do |
|---|---|
| A repository is withdrawn, renamed or re-gated (EvolutionaryScale models moved to `biohub/` in 2026; ESM3-open used to be gated) | after the first download, **archive `data/models/<backend>/`** (files + `CHECKSUMS` + `SOURCE.json`, including the model card `README.md`) for every backend used. Restoring that folder gives an offline run with identical bytes |
| The bytes change behind the same name (fair-esm URLs carry no revision) | copy `data/models/<b>/CHECKSUMS` to `configs/checksums/<b>.sha256` and commit it; from then on any difference stops the job |
| A local folder is altered | every shard re-hashes the weights (`verify_weights: full`) and stops on any difference; the provenance lists the sha256 of every weight file |
| The image or UniProt changes | jobs pull by digest; UniProt release, date and exact query are recorded and the FASTA is frozen per analysis |

| Licence question (checked 2026-11-14) | Finding |
|---|---|
| Conflict | The HF model cards of the `biohub/` ESM3/ESMC repositories and `LICENSE.md` on esm `main` say **MIT**, and the HF API shows them not gated. `LICENSE.md` at the **pinned esm tag v3.2.1** lists **ESM3-open and ESM C 600M under the Cambrian Non-Commercial License**, and the code under the Cambrian Open License. URLs are in `LICENSES.md` |
| Use rule | Primary intended use is research. Commercial use is excluded for any model, tool or data whose licence does not explicitly allow it; ESM3-open (`esm3_sm_open_v1`) and ESM C 600M (`esmc_600m`) are non-commercial under the `LICENSE.md` of the pinned esm tag v3.2.1, so they are excluded from commercial use. This manual records sources; it is not legal advice |
| Licence acceptance (if a repository is gated again) | accept the terms on its Hugging Face page while logged in, create a read token, store it as in §10; the pinned revision stays the same |
| MMseqs2 / DADApy | MMseqs2: MIT at the pinned tag 18-8cc5c (GPL-3.0 only up to release 15). DADApy: Apache-2.0, no NOTICE file |

## 15. Verification performed

| Check | Result (2026-11-14, laptop, no GPU, no image) |
|---|---|
| `bash -n` on all scripts | pass |
| `pytest tests/` (fake `sbatch`, `squeue`, `sacct`, `scancel`, `sinfo`, `singularity`, `mmseqs`; fake numpy backend; stub DADApy) | 75 passed. Covered: pins and weight checksums; layer and pooling validation; batching and shards; chunk atomicity, fsync and resume (also after an unreadable last chunk); streaming merge with id order and lazy reads; a killed shard resumes identically; changed settings, a different compute dtype, non-finite values, a backend shape that disagrees with `tools.tsv`, and tampered weights all stop the job; a failed writer cannot hang the job; bf16 detection ignores emulation; dedup; the DADApy wrapper (cap, cosine, duplicates, schema) and its missing-env error; summary tables, TWO-NN rows and PCA; plan, done rule, frozen inputs, UniProt freeze, size guard, protein hard caps per sample and per analysis (`--allow-large` ignored), GPU lane, launch and stop; the UniProt prefetch refuses above the sample cap after the first page (also across accession chunks) and when `x-total-results` is missing, and `prefetch.max_entries` can only lower the cap; the console downloads the UniProt samples of `example_chaperones` on Run from a local stand-in server and plans all 14 jobs, and above `max_entries` it saves and submits nothing; every console endpoint |
| Console in a browser (2026-11-15, workstation, scratch copy, fake SLURM commands, Firefox headless) | Samples, Run, Results and Performance at 1200 and 500 px, light and dark: no horizontal overflow (two Results tables were wrapped in a scroll box for 500 px), no badge overflow, no overlapping tick labels, no raw JSON cells, smallest text 13 px, one "no data yet" per empty table after repeated refreshes, Performance dark to the bottom of the window, Back restores the console. Run on `example_chaperones`: 14 jobs submitted to the fake `sbatch`, six step rows and the progress bar shown. Hard caps through the console (100 001 proteins in one sample; 200 001 in one analysis): refused with the message, nothing submitted |
| `example_chaperones` UniProt sizes (real download, 2026-11-15, release 2026_04) | sHSP 381, HSP70 319, HSP90 101 entries; lengths 20–1002 residues, so `prepare` (50–1022) drops 3 short fragments and ESM2 would truncate none; exact duplicate copies of a sequence before dedup: sHSP 87, HSP70 14, HSP90 3 (MMseqs2 at 95 % identity was not run on them). Pfam names in the UniProt records: PF00011 HSP20, PF00012 HSP70, PF00183 HSP90 |
| Example dry runs (scratch copy) | `example_toy`: 8 jobs would run (1 weights, 2 prepare, 2 embed, 2 merge, 1 summary), estimate < 0.001 GB. `example_chaperones` (console, real UniProt files): 14 jobs (1 weights, 3 prepare, 3 dedup, 3 embed, 3 merge, 1 summary), 0.093 GB, 801 proteins. `example_kinase_gpcr`: 15 jobs (2 weights, 2 prepare, 2 dedup, 4 embed, 4 merge, 1 summary), 0.062 GB; ESM2 sees 171 018 kinase residues against 194 473 for ESMC because of its 1022-residue truncation |
| **Verified 2026-11-15** | image built from the lock files on the workstation (12.4 GB, digest in `containers/images.env`) and pulled as a 3.6 GB `.sif` on the cluster; smoke test of torch 2.4.1+cu121, esm-evo, esm-fair, MMseqs2 and DADApy 0.3.4 on a THIN node; `--nv` on a DGX A100-SXM4-40GB (CUDA 12.1, bf16 True, access for this account works); the paper benchmarks with the real DADApy (14.05, 2.12, 22.29, 0.05); downloads of the ten pinned weight sets with sha256 recorded in `configs/checksums/`, repeated on the cluster by `src/00_weights_array.sh` (identical checksums); `example_toy` end to end through `pipeline.py` on a DGX A100 (`esmc_300m`, 2 × 5 proteins, dim 960, bf16, no truncation, weight sha256 equal to the tracked reference); `example_kinase_gpcr` end to end (`esm2_t6_8M` fp32 and `esmc_300m` bf16, 2 × 250 Swiss-Prot proteins, MMseqs2 dedup, TWO-NN on all layers: 497 points per backend and layer, d from about 2 to 11; ESM2 truncated 73 of 497 proteins at its 1022-residue limit, ESMC none) |
| **Not verified** | forward passes of ESM3, ProtT5, ProstT5 and the larger ESM2 sizes; ESMC 6B weights (not downloaded); PCA output and the aggregates were produced but not inspected; the console against the live cluster, including the UniProt download from a login node and a real run of `example_chaperones` (rendered only on the workstation with fake SLURM commands, see above) |

## 16. Troubleshooting

| Symptom | Cause → fix |
|---|---|
| `refusing to run outside the container image` | a step script was started by hand → use `pipeline.py` or the console |
| `weights check failed` / `differs from the recorded CHECKSUMS` | the weights changed or differ from `configs/checksums/` → restore the archive; if intended, move the folder aside and download again |
| `UniProt query not downloaded yet` | click Run in the console, or run `python3 src/prefetch.py` on a login node |
| `entries exceed the prefetch.max_entries of …` / `did not announce x-total-results` | the query is larger than the example or sheet allows, or UniProt did not report a count; nothing was saved for that sample → narrow the query (counts drift between UniProt releases) or raise `prefetch.max_entries` |
| `exceed embed.size_cap_gb` | fewer layers or no tokens, float16, or `--allow-large` |
| `exceed the hard cap of 100,000 per sample` / `200,000 per analysis` | narrow the query or split the data into several analyses; `--allow-large` cannot override it |
| `only the last layer` (esm3) | ESM3 exposes only its final embeddings → `layers: [-1]` |
| `chunk files … do not match the shard markers` | chunks were removed after an earlier merge → delete `<s>.shards/` and rerun |
| `another analysis has live jobs downloading the same weights` | wait for it, then launch again |
| TWO-NN `above 10%: unreliable` | many identical vectors (truncation, duplicates) → dedup, raise `max_length`, or read d with care |
| embed job killed (OOM / TIMEOUT) | it resumes from its last chunk on the next launch, and the ask doubles |
| `singularity: command not found` | `module load singularity` before starting the console |

## 17. Your own data

| Step | Action |
|---|---|
| 1 | Console › Samples: one card per sample, Local FASTA or UniProtKB query → Save samples; queries are downloaded when you click Run |
| 2 | Run › tick the backends → name the analysis → check the size line → Run on SLURM |
| 3 | Results › table, records, aggregates by backend and layer, TWO-NN and PCA, weights, provenance |

## 18. Citation and acknowledgements

The intrinsic dimension is computed with **DADApy** (Glielmo A, Macocco I, Doimo D, Carli M, Zeni C, Wild R, d'Errico M, Rodriguez A, Laio A (2022) DADApy: Distance-based analysis of data-manifolds in Python. *Patterns* 3:100589, doi:10.1016/j.patter.2022.100589), which implements the TWO-NN method of **Facco E, d'Errico M, Rodriguez A, Laio A (2017) Sci Rep 7:12140, doi:10.1038/s41598-017-11873-y**. We thank the DADApy authors and the Laio group. The paper and its SI are in `docs/references/facco2017/` (CC BY 4.0); other references are in `REFERENCES.md`.
