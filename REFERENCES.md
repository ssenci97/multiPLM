# References and acknowledgements

## Intrinsic dimension (TWO-NN)

The intrinsic dimension is computed with **DADApy**; we thank its authors and the Laio group (SISSA).

**Glielmo A, Macocco I, Doimo D, Carli M, Zeni C, Wild R, d'Errico M, Rodriguez A, Laio A (2022).** DADApy: Distance-based analysis of data-manifolds in Python. *Patterns* 3(10):100589. doi:[10.1016/j.patter.2022.100589](https://doi.org/10.1016/j.patter.2022.100589) (title, authors, volume and article number checked with Crossref on 2026-11-15)

**Facco E, d'Errico M, Rodriguez A, Laio A (2017).** Estimating the intrinsic dimension of datasets by a minimal neighborhood information. *Scientific Reports* 7:12140. doi:[10.1038/s41598-017-11873-y](https://doi.org/10.1038/s41598-017-11873-y)

| Item | Value |
|---|---|
| DADApy | https://github.com/sissa-data-science/DADApy, licence Apache-2.0 (no NOTICE file); used: PyPI sdist `dadapy-0.3.4.tar.gz`, sha256 `6907f08b4323de86d84ca985c9c403efd49ae9b5ab2da37315d975b68634244b`, uploaded 2026-05-01; checked 2026-11-14/15. The git HEAD `d729b8835e771001ec7ba6ca68938b737b96edae` (2026-10-09) also says 0.3.4 but declares other dependencies (numpy ≥ 2 to build, optax); its TWO-NN code (`_compute_id_2NN`, `compute_id_2NN`, `return_id_scaling_2NN`) is identical to the sdist |
| Licence text | `LICENSES/dadapy.LICENSE.txt`, sha256 `43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1` |
| Paper copy | `docs/references/facco2017/Facco2017_SciRep_2NN.pdf`, from https://www.nature.com/articles/s41598-017-11873-y.pdf, published online 2017-09-22, CC BY 4.0, 8 pages, sha256 `066501096eaf65332786a2cfcd1e5d79c3666ffb6ada25f12ce31ddaf1339c26` |
| SI copy | `docs/references/facco2017/Facco2017_SciRep_2NN_SI.pdf`, 6 pages, sha256 `d0e4cd3d081348ec9667bbd24ed4b31d0065dbafb534fcb743950572c28335b3` |
| Reproduction notes | the discarded-fit benchmarks (hypercube d = 14 with periodic boundaries ≈ 13.91, Swiss roll ≈ 2.01, Cauchy d = 20 ≈ 22.16) are checked with tolerances in `tests/dadapy_benchmarks.py`; the paper's single-draw full fit of 6.05 for the Cauchy set does not reproduce reliably across draws (a seeded draw with our earlier handmade code gave about 21.5) |
| Boundaries | the main text (p.4) says sharp boundaries give an overestimation; the SI (p.4) reports an **underestimation** (about 1.5 at d = 10 and 4 at d = 20 for a hypercube without periodic boundaries); MANUAL §12 follows the SI |

## Models and tools used by MultiPLM

The exact pins (git SHAs and Hugging Face revisions) are in `configs/tools.tsv`. The citations below were written from memory and **were not re-checked against the publishers on 2026-11-14**. Verify them before you cite.

| Component | Reference |
|---|---|
| ESM2 (fair-esm) | Lin Z et al. (2023) Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science* 379:1123–1130. doi:10.1126/science.ade2574 |
| ESM3 | Hayes T et al. (2025) Simulating 500 million years of evolution with a language model. *Science*. doi:10.1126/science.ads0018 |
| ESM C | EvolutionaryScale (2024), ESM Cambrian release; model cards at huggingface.co/biohub |
| ProtT5 | Elnaggar A et al. (2022) ProtTrans: toward understanding the language of life through self-supervised learning. *IEEE TPAMI* 44:7112–7127. doi:10.1109/TPAMI.2021.3095381 |
| ProstT5 | Heinzinger M et al. (2024) Bilingual language model for protein sequence and structure. *NAR Genomics and Bioinformatics* 6:lqae150. doi:10.1093/nargab/lqae150 |
| MMseqs2 | Steinegger M, Söding J (2017) MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets. *Nature Biotechnology* 35:1026–1028. doi:10.1038/nbt.3988 |
| Linclust | Steinegger M, Söding J (2018) Clustering huge protein sequence sets in linear time. *Nature Communications* 9:2542. doi:10.1038/s41467-018-04964-5 |
| UniProtKB | The UniProt Consortium, UniProt: the Universal Protein Knowledgebase (cite the current release paper); data under CC BY 4.0 |
| Code origins | Extraction logic adapted from SCambrian (`extract_reps_biohub.py`, `dw_uniprot_seqs.py`); driver, job wrapper and console ported from KNOCKIN-VAL |
