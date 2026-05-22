#!/usr/bin/env python3
"""
Generate utility and privacy figures for all dataset-target pairs.

Iterates over all 4 data-trgt pairs and creates:
1. plot_coexpression - correlation matrix of top 300 highest std genes
2. umap12 plots - UMAP1 vs UMAP2 visualization
3. plot_dcr_distributions - DCR violin plots
4. plot_nndr_distributions - NNDR histogram with inset

Outputs stored in figs/<dataset>_<target>/ folder.
"""
import importlib
import os
import numpy as np
import pandas as pd

reload_mod = lambda name: importlib.reload(importlib.import_module(name))

from bench_utils.plot_utils_ml import MODEL_COLORS
from bench_utils.plot_utils_priv_utils import (
    plot_dcr_distributions,
    get_coexpression_gene_order,
)

# ---- paths ----
pwd = os.path.dirname(os.path.abspath(__file__))

# All 4 dataset-target pairs
dataset_trgt_pairs = [
    ("tcga_luad", "clinical_stage_clean"),  # multi-class
    ("tcga_luad", "clinical_egfr_mut"),  # binary
    ("sepsis", "Classification"),  # binary
    ("risk_ped", "sample_disease"),  # multi-class
]

# Model names for plotting
MODEL_NAMES = {
    "dbtwin": "dbTwin",
    "pca_ctgan": "PCA-CTGAN",
    "mvn": "Class MVN",
}
do_coexp = False
do_umap = False
do_dcr = True

legend_str= {"clinical_stage_clean": "stage:",  # multi-class
    "clinical_egfr_mut": "EGFR:",  # binary
    "Classification": None,  # binary
    "sample_disease":None}

for dataset, trgt in dataset_trgt_pairs:
    print(f"\nProcessing {dataset} - {trgt}...")

    dfolder = os.path.join(pwd, "data", dataset)

    # Check if results file exists
    results_file = os.path.join(dfolder, f"bench_results_dcr_{trgt}.npz")
    if not os.path.exists(results_file):
        print(f"  Warning: {results_file} not found, skipping...")
        continue

    # ---- load benchmark results from npz ----
    tmp = np.load(results_file, allow_pickle=True)

    # Main model (dbTwin) privacy metrics
    expr_dists_dbt = tmp["expr_dists_dbt"]
    # CTG model privacy metrics
    expr_dists_ctg = tmp["expr_dists_ctg"]
    # MVN model privacy metrics
    expr_dists_mvn = tmp["expr_dists_mvn"]
    tmp.close()
    del tmp

    # Convert object arrays from make_obj_array to list of dicts
    def extract_object_array(obj):
        """Extract list of dicts from object dtype array created by make_obj_array."""
        if isinstance(obj, np.ndarray) and obj.dtype == object:
            # Object array from make_obj_array - convert to list
            return obj.tolist()
        return obj

    # Extract privacy metrics from object arrays
    expr_dists_dbt = extract_object_array(expr_dists_dbt)
    expr_dists_ctg = extract_object_array(expr_dists_ctg)
    expr_dists_mvn = extract_object_array(expr_dists_mvn)
        # Use first sample's data for plotting (privacy metrics are stored per sample)
    # Each element is a dict with keys like "d_trn", "d_tst" for DCR
    fig_folder = os.path.join(pwd, "figs", f"{dataset}_{trgt}")
    os.makedirs(fig_folder, exist_ok=True)

    # reload in case plot_utils was edited between runs
    reload_mod("bench_utils.plot_utils_priv_utils")
    from bench_utils.plot_utils_priv_utils import (
       plot_dcr_distributions, )

    # ---- DCR distributions for all 3 models ----
    if do_dcr:
        print(f"  Generating DCR distributions...")

        for expr_d, model_key in [
            (expr_dists_dbt, "dbtwin"),
            (expr_dists_mvn, "class-mvn"),
            (expr_dists_ctg, "pca-ctgan"),
        ]:
            plot_dcr_distributions(expr_d, model_name=model_key, fig_folder=fig_folder)

    print(f"Done with {dataset} - {trgt}")

print("\nAll utility and privacy figures generated successfully!")
