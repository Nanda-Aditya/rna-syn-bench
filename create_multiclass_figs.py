"""
Generate ML figures for multiclass classification targets.

Targets:
- "clinical_stage_clean" with "tcga_luad" dataset (3 classes)
- "sample_disease" with "risk_ped" dataset (3 classes)

Figures generated for each target-dataset pair:
- Macro-AUC bar plot with error bars (using auc_bar_plots)
- Macro per-class recall (PCR) bar plot with error bars (using pcr_bar_plots)
- SHAP plots for each of the 3 models (using median performing sample)

Note: No DE figures for multiclass targets.

Outputs stored in figs/<dataset>_<target>/ folder
"""

import importlib
import os
import numpy as np
import pandas as pd


reload_mod = lambda name: importlib.reload(importlib.import_module(name))

from bench_utils.plot_utils_ml import (auc_bar_plots,
    pcr_bar_plots,shap_plots, MODEL_COLORS)

from bench_utils.de_ml_utils import compute_shap_correlation

# ---- paths ----
pwd = os.path.dirname(os.path.abspath(__file__))

# Multiclass classification target-dataset pairs
dataset_trgt_pairs = [
    ("risk_ped", "sample_disease"),  # Disease classification (3 classes)
    ("tcga_luad", "clinical_stage_clean"),  # Stage prediction (3 classes)
]

top_k = 9 # shap_columns
sp_k =50  # num_features for sp. corr computation
r_k=50  # pearson's r
create_si_figs=False
def find_median_sample(auc1_values):
    """
    Find the index of the median performing sample based on macro-AUC values.

    Parameters:
    -----------
    auc1_values : array-like (n_samples,) or (n_samples, n_classes)
        AUC values for synthetic models across samples. For multi-class,
        mean across classes is computed to get a single score per sample.

    Returns:
    --------
    median_idx : int
        Index of the sample with median AUC performance
    """
    auc1_arr = np.asarray(auc1_values)
    # For multi-class (2D array), compute mean across classes first
    if auc1_arr.ndim > 1 and auc1_arr.shape[1] > 1:
        auc1_arr = auc1_arr.mean(axis=1)
    # Sort and find median
    sorted_indices = np.argsort(auc1_arr)
    n = len(auc1_arr)
    median_idx = sorted_indices[n // 2]
    return median_idx


for dataset, trgt in dataset_trgt_pairs:
    print(f"\nProcessing {dataset} - {trgt}...")

    dfolder = os.path.join(pwd, "data", dataset)

    # Check if results file exists
    results_file = os.path.join(dfolder, f"bench_results_ml_{trgt}.npz")
    if not os.path.exists(results_file):
        print(f"  Warning: {results_file} not found, skipping...")
        continue

    # ---- load benchmark results from npz ----
    tmp = np.load(results_file, allow_pickle=True)
    ml_real = tmp["ml_real"]
    ml_dbtwin = tmp["ml_dbtwin"]
    ml_mvn = tmp["ml_mvn"]
    ml_ctg = tmp["ml_ctg"]
    gexp_cols_trn= tmp["gexp_cols_trn"].tolist()

    # Convert object arrays to lists (for arrays created by make_obj_array)
    # SHAP arrays may have different shapes across samples, so keep as list
    def ensure_array(x):
        if isinstance(x, np.ndarray) and x.dtype == object:
            return x.tolist()
        return x

    # Ensure shap values are properly formatted (handle object dtype arrays from make_obj_array)
    for i in range(len(ml_real)):
        ml_real[i]["shap"] = ensure_array(ml_real[i]["shap"])
    for i in range(len(ml_dbtwin)):
        ml_dbtwin[i]["shap"] = ensure_array(ml_dbtwin[i]["shap"])
    for i in range(len(ml_ctg)):
        ml_ctg[i]["shap"] = ensure_array(ml_ctg[i]["shap"])
    for i in range(len(ml_mvn)):
        ml_mvn[i]["shap"] = ensure_array(ml_mvn[i]["shap"])

     # Output folder: figs/<dataset>_<target>/ to avoid overwriting make_de_ml.py results
    # e.g., figs/tcga_luad_clinical_stage_clean/ and figs/risk_ped_sample_disease/
    fig_folder = os.path.join(pwd, "figs", f"{dataset}_{trgt}")
    os.makedirs(fig_folder, exist_ok=True)

    # ---- ML figures ----
    print(f"  Generating ML figures...")

    # AUC bar plots with error bars for all 3 models (Logistic Regression)
    print(f"  Generating AUC bar plots (Logistic Regression)...")
    auc1_dict = {
        "dbtwin": [x["auc"] for x in ml_dbtwin],
        "class-mvn": [x["auc"] for x in ml_mvn],
        "pca-ctgan": [x["auc"] for x in ml_ctg],
    }
    auc0_list = [x["auc"] for x in ml_real]
    auc_bar_plots(auc0_list, auc1_dict, fig_folder)

    # PCR (per-class recall) bar plots with error bars for Logistic Regression models
    print(f"  Generating PCR bar plots (Logistic Regression)...")
    pcr1_dict = {
        "dbtwin": [x["pcr"] for x in ml_dbtwin],
        "class-mvn": [x["pcr"] for x in ml_mvn],
        "pca-ctgan": [x["pcr"] for x in ml_ctg],
    }
    pcr0_list = [x["pcr"] for x in ml_real]
    pcr_bar_plots(pcr0_list, pcr1_dict, fig_folder)

    # SHAP plots using median fold for all 3 models (Logistic Regression)
    print(f"  Generating SHAP plots (Logistic Regression)...")

    shap0_list = [x["shap"] for x in ml_real]
    shap1_list = [x["shap"] for x in ml_dbtwin]
    shap1_ctg_list = [x["shap"] for x in ml_ctg]
    shap1_mvn_list = [x["shap"] for x in ml_mvn]

    # Get class information from any sample (all have same classes)
    classes = ml_real[0]['classes_']
    orig_classes_ = ml_real[0]['orig_classes_']

    # Find median performing sample for each model based on AUC
    auc1_dict = {
        "dbtwin": [x["auc"] for x in ml_dbtwin],
        "pca-ctgan": [x["auc"] for x in ml_ctg],
        "class-mvn": [x["auc"] for x in ml_mvn],
    }
    
    median_idx = find_median_sample(auc0_list)

    for shap_mat, model in zip(
        [shap1_list, shap1_ctg_list, shap1_mvn_list],
        ["dbTwin", "PCA-CTGAN", "class-MVN"]):
        
        # Compute SHAP correlations across all folds for title annotation
        shap_corr = compute_shap_correlation(shap0_list, shap_mat, sp_k=100, r_k=50)
        
        shap_results = shap_plots(
            shap0_list[median_idx],  # Use median fold from real data
            shap_mat[median_idx],    # Use median fold from synthetic model
            fig_folder,
            gexp_cols_trn[median_idx],
            syn_color=MODEL_COLORS[model.lower()],
            model_name=model.lower(),
            top_k=top_k,
            sp_k=sp_k,
            r_k=r_k,
            sp_rho=shap_corr["median_sp_rho"],
            sp_rho_std=shap_corr["std_sp_rho"],
            pearson_r=shap_corr["median_r"],
            pearson_r_std=shap_corr["std_r"],
            dataset_name=dataset,
            classes=classes,
            orig_classes_=orig_classes_,
        )

    # Generate SHAP plots for all other folds and save to si_shap folder

    if create_si_figs:
        print(f"  Generating SHAP plots for all other folds (si_shap)...")
        si_shap_folder = os.path.join(fig_folder, "si_shap")
        os.makedirs(si_shap_folder, exist_ok=True)

        for shap_mat, model in zip(
        [shap1_list, shap1_ctg_list, shap1_mvn_list],
        ["dbTwin", "PCA-CTGAN", "class-MVN"],
        ):
            print(f"    Processing all folds for {model} (si_shap)...")
            for fold_idx in range(len(shap_mat)):
                if fold_idx == median_idx:
                    continue  # Skip median fold as it's already processed above

                shap_results = shap_plots(
                    shap0_list[fold_idx],  # Use current fold from real data
                    shap_mat[fold_idx],    # Use current fold from synthetic model
                    si_shap_folder,
                    gexp_cols_trn[fold_idx],
                    syn_color=MODEL_COLORS[model.lower()],
                    model_name=f"{model.lower()}_fold{fold_idx}",
                    top_k=top_k,
                    sp_k=sp_k,
                    r_k=r_k,
                    dataset_name=dataset,
                    classes=classes,
                    orig_classes_=orig_classes_,
                )
print("\nAll multiclass ML figures generated successfully!")
