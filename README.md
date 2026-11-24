# Heterogeneous Attention in Small Vision Transformers

[Español](README.es.md)

Code, configurations and experiment records for the article *Inductive Biases, Not Similarity Functions: A Controlled Evaluation of Performance, Scale and Calibration of Heterogeneous Attention in Small Vision Transformers* (Martínez-Marín, Hernández-Nolasco and Zacarías-Morales, submitted).

**In one sentence:** in a 119k-parameter ViT trained from scratch, combining two similarity functions within a head (dot-product + additive attention) does not beat an equal-capacity control, while adding an inductive bias that attention lacks, a 5×5 depthwise convolution over the values of each head, adds about 5 points on CIFAR-10. The useful heterogeneity is in inductive biases, not in similarity functions.

## Contents

| Path | What it is |
|---|---|
| `src/ham/` | Model (`models/`: ViT, attention branches, fusions, output terms, ResNet baseline), training (`train.py`), evaluation (`evaluate.py`), post-hoc analysis (`analyze.py`), visualization |
| `configs/` | One YAML per variant (`<dataset>_<variant>.yaml`), inheriting from `_<dataset>.yaml` and `base.yaml` |
| `scripts/run_paper.sh` | Every experiment phase of the article (`fase1` … `fase18`) |
| `scripts/ieee_tables.py`, `scripts/tables_en.py`, `scripts/paper_figures.py` | Tables and figures of the article, generated from the run records |
| `scripts/costos.py` → `results/costos.json` | Hardware-independent costs (MACs, FLOPs, memory) and CPU latency |
| `records/runs.tar.gz` | Records of the 830 runs used in the article (see below) |
| `paper/ieee_en/` | LaTeX sources of the article (preprint, IEEE layout) |
| `tests/` | Unit tests (`pytest`) |

Code comments and some docstrings are in Spanish; the names of the variants are the ones used in the article.

## Installation

Requires [uv](https://docs.astral.sh/uv/). `uv.lock` pins the exact versions used (Python 3.13, PyTorch 2.14.1 + CUDA 13.0).

```bash
uv sync
uv run pytest
```

## Reproducing the tables without training

Every run stored its resolved configuration, code commit, environment, per-epoch metrics, test evaluation and post-hoc analysis. `records/runs.tar.gz` contains those files (`config.resolved.yaml`, `provenance.json`, `metrics.json`, `test.json`, `analysis.json`) for all 830 runs, in the same directory layout the scripts expect. Weights (`best.pt`), predictions and the train/val split indices are not included: the split is regenerated from `data.split_seed`.

```bash
tar xzf records/runs.tar.gz                     # creates runs/, runs_seeds/, runs_corta/, runs_noaug/
uv run python scripts/ieee_tables.py            # Spanish tables (intermediate step) -> paper/ieee/tablas/
uv run python scripts/tables_en.py              # English tables -> paper/ieee_en/tablas/
git diff --stat paper/ieee_en/tablas            # should be empty: identical to the article
```

The figures that only need the records (`curvas.pdf`, `semillas.pdf`, `kernel.pdf`, `mecanismo.pdf`) are regenerated with:

```bash
uv run python scripts/paper_figures.py --lang en --out paper/ieee_en --sin-mapas
```

The attention maps (`mapas.png`) and their similarity table (`paper/tablas/mapas.tex`, included as is) need the model weights.

## Training

Data are expected in `data/` (`data.root`). CIFAR-10, CIFAR-100 and SVHN are downloaded automatically by torchvision. PlantVillage (`data/plantvillage/{train,val}/<class>/`) and Food-101 (`data/food-101/{images,meta}/`) must be downloaded separately.

```bash
# one run
uv run python -m ham.train --config configs/cifar10_dot_local_k5.yaml --seed 0
# several seeds
scripts/run_seeds.sh "0 1 2 3 4" configs/cifar10_dot.yaml configs/cifar10_dot_local_k5.yaml
# a whole phase of the article
scripts/run_paper.sh fase12
# post-hoc analysis (calibration, robustness, attention metrics) of a run
uv run python -m ham.analyze --run runs/cifar10/cifar10_dot_local_k5/seed0
```

Any field can be overridden from the command line (`train.epochs=1`, `model.attention.branches=[dot,additive]`); unknown keys raise an error. Each run is written to `runs/<dataset>/<dataset>_<variant>/seed<k>/`.

**Protocol.** 45k/5k/10k split on CIFAR-10. The model with the best validation accuracy is kept, and the test set is evaluated once. Training is deterministic: with the same seed, code and environment the metrics are reproduced bit for bit, which is why seeds are paired across variants. The article used an NVIDIA RTX 5080; on other GPUs or CUDA versions small differences are expected, so each run records its environment.

### Main variants

| Variant (config) | Description |
|---|---|
| `dot` | Base ViT: D=48, 4 layers, 4 heads, 119,290 parameters |
| `dot_wide` | Capacity control: MLP of 217 units |
| `ham_concat` | dot + additive *pooling* branch, concatenated and projected per head |
| `ham_weighted`, `ham_sum`, `ham_zscore`, `ham_wout`, … | Other two-branch fusions (score mixing and output mixing) |
| `dot_local`, `dot_local_k5`, `dot_local_k7`, `dot_local_k1` | dot + k×k depthwise convolution over the values (LePE); k=1 is the control without neighborhood |
| `dot_wide_local_k5` | Capacity control of the triad |
| `ham_concat_local_k5` | The triad: pairwise similarity + global summary + 5×5 neighborhood |
| `localvit`, `spt_lsa`, `convit`, `conv_stem`, `resnet` | Published mechanisms reimplemented at the same budget |

## Compiling the article

```bash
cd paper/ieee_en && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Checkpoints

The weights of all runs (about 1.1 GB) will be published as a release or on Zenodo. <!-- PENDIENTE: enlace -->

## Citation and license

Archived in Zenodo: [doi:10.5281/zenodo.24452704](https://doi.org/10.5281/zenodo.24452704).

See `CITATION.cff`. Code: MIT (`LICENSE`). Records and manuscript sources: CC BY 4.0 (`LICENSE-records`).

This repository is exported from the author's working repository; internal working notes are not included.
