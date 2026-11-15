# MultiPLM: resume notes

Do not delete this file without asking the user. Current state on 2026-11-15; light checks only, nothing built or run on a GPU or a cluster.

## Current state

| Item | State |
|---|---|
| Pipeline | `00_weights` → `01_prepare` → `02_dedup` (optional, MMseqs2) → `03_embed` (GPU shards, atomic safetensors chunks, resume) → `04_merge` (streaming, row parts + index + id sidecar) → `05_summary` (tables, aggregates, TWO-NN through DADApy, optional joint PCA) |
| Intrinsic dimension | DADApy 0.3.4 (PyPI sdist, sha256 `6907f08b…`, Apache-2.0) in an isolated venv `/opt/envs/dadapy`; `src/intrinsic_dim.py` is the thin wrapper; the handmade `twonn.py` was removed |
| Review fixes applied | B1 bf16 detection ignores emulation; B3 compute dtype in shard settings and merge check; R6 backend shape checked against `tools.tsv`; R1 writer failure cannot hang the job; R2 non-finite values stop the shard; N4 fsync + unreadable last chunk dropped on resume. Documented only (MANUAL §2, §9): B2, R7, R4, R5 |
| Tests | 64 pass with `nice -n 19` (about 9 s), no cluster, image or GPU |
| Docs | MANUAL (18 sections), README, REFERENCES (DADApy + Facco et al. 2017 with both PDFs), LICENSES (research only; ESM licence conflict; MMseqs2 MIT; DADApy Apache-2.0) |
| Jason KB | GPU guidance (pragmatic default) in PLAYBOOK §6b/§11, CHECKLIST 30c, LESSONS, memory; `sync.sh` run |

## Open items (recommended default in brackets)

| Item | Note |
|---|---|
| Image build | the user runs `build.sh build`, `lock` and `test` (the test also runs the DADApy paper benchmarks) |
| Summary job memory | decided: total size of the loaded reps (minimum 1 GB) + 5 GB when TWO-NN is on; sacct learning and OOM doubling on top |
| Commercial use | decided: research is the primary use; commercial use excluded wherever a licence does not explicitly allow it (ESM3-open, ESM C 600M excluded) |
| First real run | `example_toy`, then `example_kinase_gpcr`, `--dtype fp32` for the first runs [as decided] |
| Not applied from the review | sliding windows, layer 0, pre-norm layer export, token-budget nits, extra provenance fields [skip, as decided] |
