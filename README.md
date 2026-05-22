# rna-syn-bench

Benchmark code for the preprint **"Synthetic RNA-seq cohorts for data sharing: a discovery-aware benchmark at transcriptome scale"** (Nanda & Saha, 2026).

---

## Overview

Sharing patient-level gene expression data accelerates translational research but carries re-identification risk. This repository provides a multi-axis benchmark that evaluates synthetic RNA-seq cohorts on three complementary axes:

1. **Differential expression (DE) fidelity** — does the synthetic cohort reproduce real DE signals (gene recovery, log2FC, and significance concordance)?
2. **ML utility / TSTR** — can a model trained on synthetic data classify real held-out patients as well as one trained on real data? Do SHAP feature attributions agree?
3. **Empirical privacy risk** — does distance-to-closest-record (DCR) analysis show signs of memorization?

---

## Cohorts

Four dataset-target pairs drawn from three public studies:

| Cohort | Source | Target | Task |
|--------|--------|--------|------|
| TCGA-LUAD (EGFR) | TCGA | `clinical_egfr_mut` | Binary — EGFR mutation status |
| TCGA-LUAD (Stage) | TCGA | `clinical_stage_clean` | Multiclass — clinical stage |
| Sepsis | GSE184900 | `Classification` | Binary — sepsis classification |
| Pediatric IBD (RISK) | GSE57945 | `sample_disease` | Multiclass — Crohn's / UC / control |

---

## Models Benchmarked

| Model | Description |
|-------|-------------|
| **dbTwin** | Non-deep-learning, target-conditioned method operating natively at RNA-seq scale |
| **class-MVN** | Low-rank target-conditioned multivariate Gaussian |
| **PCA-CTGAN** | Tabular GAN trained in PCA-compressed expression space |

All synthetic cohorts were generated from training folds of a five-fold stratified cross-validation design.

---

## Repository Layout

```
rna-syn-bench/
├── run_de.py                    # Compute DE fidelity (requires raw data)
├── run_ml.py                    # Compute ML / TSTR metrics (requires raw data)
├── compute_dcr_privacy.py       # Compute DCR privacy metrics (requires raw data)
│
├── create_binary_de_ml_figs.py  # Figures for binary cohorts (DE + ML)
├── create_multiclass_figs.py    # Figures for multiclass cohorts (ML + SHAP)
├── create_privacy_dcr_figs.py   # DCR privacy figures
│
├── bench_utils/
│   ├── gexp_de.py               # DESeq2-based DE fidelity pipeline
│   ├── ml_wrapper.py            # TSTR/TRTR training, MAD feature selection, SHAP
│   ├── privacy_metrics.py       # DCR and NNDR computation
│   ├── config.py                # demo_run flag and global settings
│   └── ...                      # Plotting utilities and helpers
│
├── data/
│   ├── tcga_luad/               # Pre-computed .npz result files
│   ├── sepsis/
│   ├── risk_ped/
│   └── gene_mapping_cache/
│
└── figs/                        # Output figures (PDF/PNG) per cohort-target
```

---

## Quick Start — Reproducing Figures

The raw expression matrices live in a Synapse project (see [Data Access](#data-access) below), but pre-computed summary statistics for all four cohorts are included in `data/`. The three figure scripts read these `.npz` files directly and write output to `figs/`:

```bash
# Binary cohorts: DE volcano plots, DE bar charts, ROC-AUC bars, SHAP panels
python create_binary_de_ml_figs.py

# Multiclass cohorts: ROC-AUC bars, per-class SHAP panels
python create_multiclass_figs.py

# All cohorts: DCR distributions per model
python create_privacy_dcr_figs.py
```

Figures are written to `figs/<dataset>_<target>/`.

---

## Full Benchmark Run (requires raw data)

Once the raw data is placed under `data/` (see [Data Access](#data-access)), the full pipeline is:

```bash
python run_de.py           # DE fidelity — saves bench_results_de_<target>.npz
python run_cross_validation.py # Implements hyperparameter sweep on real training and test folds and saves it (run this before run_ml.py)
python run_ml.py           # ML / TSTR   — saves bench_results_ml_<target>.npz
python compute_dcr_privacy.py  # Privacy    — saves bench_results_dcr_<target>.npz
```

Then re-run the figure scripts above.

---

## Key Modules

### `bench_utils/gexp_de.py`
Runs DESeq2 (via `pydeseq2`) independently on a real and a synthetic cohort using the same contrast, then measures DE gene recovery (overlap %), log2FC correlation, and significance (padj) concordance at two stringency thresholds.

### `bench_utils/ml_wrapper.py`
Implements the TSTR/TRTR pipeline: log1p transformation, MAD-based top-gene selection, elastic-net logistic regression with hyperparameter tuning, AUC/PR-AUC evaluation on real held-out data, and SHAP feature attribution comparison between real-trained and synthetic-trained models.

### `bench_utils/plot_utils_*.py`
Implements plotting code for DE/ML and DCR plots

### `baseline_generators/baseline_gen.py`
Implements the two baseline synthetic data generators: `class_mvn` generates synthetic samples via a low-rank SVD projection followed by per-class Gaussian sampling in the reduced space, and `pca_ctgan` reduces expression dimensionality with PCA before fitting SDV's CTGAN on the combined clinical and PCA-compressed data.

### `bench_utils/de_ml_utils.py`
Implements simple utility functions
- compute_shap_correlation(): computes mean absolute SHAP correlations between real-trained and synthetic-trained models on held-out data for each fold.
- compute_de_corrs(): aligns gene-sets from DE outputs to overlapping genes and computes log2FC and padj correlations over all folds.

---

## Dependencies

```
scikit-learn==1.6.1
scipy==1.15.1
numpy==2.2.0
pandas==2.3.1
pydeseq2==0.5.3
joblib==1.4.2
umap-learn==0.5.9.post2
mygene==3.2.2
matplotlib
shap
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Data Access

The pre-computed `.npz` summary files in this repository are sufficient to reproduce all figures without the raw data. Raw count matrices (real patient data and synthetic cohorts) including for class-MVN and PCA-CTGAN are deposited at Synapse project [syn75080394](https://www.synapse.org/Synapse:syn75080394). Access requires a brief, free registration and acceptance of a non-commercial / research-use-only DUA to access dbTwin-generated cohorts. 

---

## Citation

If you use this benchmark, please cite the preprint:

> Nanda A, Saha S. *Synthetic RNA-seq cohorts for data sharing: a discovery-aware benchmark at transcriptome scale.* 2025. Preprint.

And the software: 

[![DOI](https://zenodo.org/badge/1246845386.svg)](https://doi.org/10.5281/zenodo.20347265)

Corresponding author: aditya@dbtwin.com
