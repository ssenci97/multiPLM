# MultiPLM — instructions for Claude Code

Containerized multi-backend protein language model embedding extraction, in the KNOCKIN-VAL pattern (pinned image → Singularity `.sif`, SLURM jobs only inside it, file-driven `src/pipeline.py`, stdlib console). `MANUAL.md` is the protocol.

| Rule | Detail |
|---|---|
| Commits | Never `git commit` or `git push`, and never offer to. Leave the tree committable |
| Style | Every script follows `.claude/agents/simple-code-reviewer.md` (no comments, type hints and short docstrings, `######` sections, `[PYTHON-INFO]`/`[SLURM-INFO]` tags, `--yaml-in`/`--yaml-out`, footer line). Run that agent after editing scripts |
| Resources | 1 core / 1 GB per job unless justified; half a node is the ceiling, not a target; GPU: 1 GPU at a time per analysis by default, at most 2, never more than one DGX job. Every ask above 1/1 is confirmed by the user |
| Reproducibility | Weights only from local pinned snapshots with sha256 checks; no API backends by default; pins live in `configs/tools.tsv`; TWO-NN only through the pinned DADApy env (`src/intrinsic_dim.py`) |
| Ask first | `docker push`, registry login, `sbatch`, deleting data, multi-GB downloads or builds on the laptop |
| Tests | `nice -n 19 python3 -m pytest -q tests` (no cluster, no container, no GPU needed; DADApy is stubbed); the real-DADApy benchmarks run in the image via `containers/build.sh test` |
