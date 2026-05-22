#!/usr/bin/env python3
"""
DE-only pipeline for recomputing differential expression results.

Iterates over the same trgt-data pairs as make_de_ml and compute_benchmarks,
runs the DE pipeline exactly as in compute_benchmarks.py, and appends the
new DE results to existing .npz files.

All existing variables in the npz file are retained; only DE results are overwritten.
"""

import numpy as np
import pandas as pd
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench_utils.gexp_de import de_computes
from bench_utils.make_obj_array import make_obj_array

# Path of the script (repo location)
pwd = os.path.dirname(os.path.abspath(__file__))

# Binary classification target-dataset pairs (same as make_de_ml.py)
dataset_trgt_pairs = [
    ("tcga_luad", "clinical_egfr_mut"),  # EGFR mutation prediction (binary)
    ("sepsis", "Classification"),  # Sepsis classification (binary)
]

n_samp = 5
do_real = True
do_dbtwin = True
do_ctg = True
do_mvn = True

import time

# Control which DE analyses to run - surgical update flags

for dataset, trgt in dataset_trgt_pairs:
    print(f"\nProcessing {dataset} - {trgt}...")

    dfolder = os.path.join(pwd, "data", dataset)
    # DE results storage (genes_min, genes_mod, lfc_all, pval_all, genes_all)
    de_real, de_dbTwin, de_ctg, de_mvn = [], [], [], []

    # Load metadata for phenotype column names
    phtp0 = pd.read_csv(
        os.path.join(dfolder, "full_datasets", "metadata.csv"), index_col=0
    )
    phtp_cols = phtp0.columns

    for ss in np.arange(1, n_samp + 1):
        print(f"  Processing sample {ss}/{n_samp}...")

        # Load training data
        df_trn = pd.read_csv(
            os.path.join(dfolder, "df_trn", f"df_trn_{trgt}_{ss}.csv"), index_col=0
        )

        # Get gene expression columns
        gexp_cols = df_trn.drop(columns=phtp_cols).columns

        # Load synthetic data conditionally based on flags
        if do_dbtwin:
            df_syn = pd.read_csv(
                os.path.join(dfolder, "dbTwin", f"df_syn_{trgt}_{ss}.csv"), index_col=0
            )            

        if do_ctg:
            df_syn_ctg = pd.read_csv(
                os.path.join(dfolder, "pca_ctgan", f"df_syn_ctg_{trgt}_{ss}.csv"),
                index_col=0,
            )            

        if do_mvn:
            df_syn_mvn = pd.read_csv(
                os.path.join(dfolder, "class_mvn", f"df_syn_mvn_{trgt}_{ss}.csv"),
                index_col=0,
            )            

        # preprocessing - eliminate low-20% genes
        from bench_utils.ml_wrapper import mad
        gene_mad = mad(df_trn[gexp_cols])
        ix_trn = np.where(gene_mad> np.percentile(gene_mad, 20))[0]  # top mad genes

        # Real data (reference) - conditional
        if do_real:
            de_tmp = de_computes(df_trn[[trgt]+list(gexp_cols[ix_trn])], trgt)
            de_real.append(de_tmp)

        # dbTwin synthetic - conditional
        if do_dbtwin:
           de_tmp = de_computes(df_syn[[trgt]+list(gexp_cols[ix_trn])],
                        trgt)
           de_dbTwin.append(de_tmp)

        # class_mvn synthetic - conditional
        if do_mvn:
            de_tmp = de_computes(df_syn_mvn[[trgt]+list(gexp_cols[ix_trn])],
                            trgt)
            de_mvn.append(de_tmp)

        # pca_ctgan synthetic - conditional
        if do_ctg:
            de_tmp = de_computes(df_syn_ctg[[trgt]+list(gexp_cols[ix_trn])],
                            trgt)
            de_ctg.append(de_tmp)

    # Load existing npz file
    results_file = os.path.join(dfolder, f"bench_results_de_{trgt}.npz")
    if do_real:
        de_real = make_obj_array(de_real)
    if do_dbtwin:
        de_dbTwin = make_obj_array(de_dbTwin)
    if do_ctg:
        de_ctg = make_obj_array(de_ctg)
    if do_mvn:
        de_mvn = make_obj_array(de_mvn)

     # Save updated data
    print(f"  Saving updated results to {results_file}...")
    np.savez(results_file, de_real=de_real,
             de_dbTwin=de_dbTwin, de_ctg=de_ctg, de_mvn=de_mvn)

    print(f"  Done with {dataset} - {trgt}")

print("\nAll DE results updated successfully!")
