"""
Generate DE + ML figures for binary classification targets.

Targets:
- "clinical_egfr_mut" with "tcga_luad" dataset
- "Classification" with "sepsis" dataset

Figures generated for each target-dataset pair:
- volcano_plot (using median performing sample based on AUC)
- de_bars_mod with error bars
- auc bar plot with error bars
- shap plot (using median performing sample)

Outputs stored in figs/<dataset>/ folder.
"""
from bench_utils.de_ml_utils import compute_shap_correlation
import importlib
import os
import numpy as np
import pandas as pd
from bench_utils.de_ml_utils import align_de_to_real

reload_mod = lambda name: importlib.reload(importlib.import_module(name))
from bench_utils.de_ml_utils import compute_de_corrs

from bench_utils.plot_utils_ml import (
    auc_bar_plots,
    shap_plots,
    MODEL_COLORS,
)
from bench_utils.plot_utils_de import (
    de_bar_plots,
    volcano_plotter,
)

# ---- paths ----
pwd = os.path.dirname(os.path.abspath(__file__))

# Binary classification target-dataset pairs
dataset_trgt_pairs = [
    ("sepsis", "Classification"),  # Sepsis classification (binary)
    ("tcga_luad", "clinical_egfr_mut"),  # EGFR mutation prediction (binary)
]

# Global plotting parameters for SHAP plots
top_k = 9
sp_k = 50
r_k=50
do_ml=True
do_de=True
do_si_shap=False

def find_median_sample(auc1_values):
    """
    Find the index of the median performing sample based on AUC values.

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

    dfolder = os.path.join(pwd, "data", dataset)
    # Check if results file exists
    results_file = os.path.join(dfolder, f"bench_results_ml_{trgt}.npz")
    tmp = np.load(results_file, allow_pickle=True)
    if do_de:
        de_results_file = os.path.join(dfolder, f"bench_results_de_{trgt}.npz")
        tmp_de = np.load(de_results_file, allow_pickle=True)

    # ---- load benchmark results from npz ----

    # ML results - new format (list of dicts per sample)
    if do_ml:
        ml_real = tmp["ml_real"]
        ml_dbtwin = tmp["ml_dbtwin"]
        ml_mvn = tmp["ml_mvn"]
        ml_ctg = tmp["ml_ctg"]
        gexp_cols_trn= tmp["gexp_cols_trn"].tolist()

    # DE results - new format (list of dicts per sample)
    if do_de:
        de_real = tmp_de["de_real"]
        de_dbTwin = tmp_de["de_dbTwin"]
        de_ctg = tmp_de["de_ctg"]
        de_mvn = tmp_de["de_mvn"]


    # Convert object arrays to lists (for arrays created by make_obj_array)
    # SHAP arrays may have different shapes across samples, so keep as list
    def ensure_array(x):
        if isinstance(x, np.ndarray) and x.dtype == object:
            return x.tolist()
        return x

    # Ensure shap values are properly formatted (handle object dtype arrays from make_obj_array)
    if do_ml:
        for i in range(len(ml_real)):
            ml_real[i]["shap"] = ensure_array(ml_real[i]["shap"])
        for i in range(len(ml_dbtwin)):
            ml_dbtwin[i]["shap"] = ensure_array(ml_dbtwin[i]["shap"])
        for i in range(len(ml_ctg)):
            ml_ctg[i]["shap"] = ensure_array(ml_ctg[i]["shap"])
        for i in range(len(ml_mvn)):
            ml_mvn[i]["shap"] = ensure_array(ml_mvn[i]["shap"])

    # Ensure DE results are properly formatted
    if do_de:
        de_real = ensure_array(de_real)
        de_dbTwin = ensure_array(de_dbTwin)
        de_ctg = ensure_array(de_ctg)
        de_mvn = ensure_array(de_mvn)

    # Output folder: figs/<dataset>/ (e.g., figs/tcga_luad/)
    fig_folder = os.path.join(pwd, "figs", dataset + "_" + trgt)
    os.makedirs(fig_folder, exist_ok=True)

    # ---- Find median performing sample for scatter and SHAP plots ----
    # Use model-specific AUC to determine median performance for each model
    median_idx = find_median_sample([x["auc"] for x in ml_real])
        # ---- DE figures (binary only) ----

    if do_de:
        print(f"  Generating DE figures...")

        aligned_ = {
            "dbTwin": align_de_to_real(de_real[median_idx], de_dbTwin[median_idx]),
            "pca-ctgan": align_de_to_real(de_real[median_idx], de_ctg[median_idx]),
            "mvn": align_de_to_real(de_real[median_idx], de_mvn[median_idx]),
        }

        # bar plots
        de_bar_plots([de_real, de_dbTwin, de_mvn], fig_folder)
        print(f"  Generating volcano plots...")

        # Volcano plots:

        # real data  #
        volcano_plotter(
            de_real[median_idx]["lfc_all"],
            -np.log10(de_real[median_idx]["padj_all"]),
            de_real[median_idx]["genes_all"],
            fig_folder,
            model="real",
        )

        # dbTwin et al.
        for model_name, model_dict, idx in zip(
            ["dbTwin", "pca-ctgan", "mvn"],
            [de_dbTwin, de_ctg, de_mvn],
            [median_idx, median_idx, median_idx],
        ):
           m_lfc, m_pj, s_l, s_pj = compute_de_corrs(de_real, model_dict, mod_genes=True)
           volcano_plotter(
                aligned_[model_name]["lfc_aligned"],
                -np.log10(aligned_[model_name]["padj_aligned"]),
                aligned_[model_name]["genes_aligned"],
                fig_folder,
                model=model_name,
                de_corrs=[ m_lfc, m_pj, s_l, s_pj])



    # ---- ML figures ----
    if do_ml:
        print(f"  Generating ML figures...")

        # AUC bar plots with error bars for all 3 models (Logistic Regression)
        print(f"  Generating AUC bar plots (Logistic Regression)...")
        auc1_dict = {
            "dbtwin": [x["auc"] for x in ml_dbtwin],
            "class-mvn": [x["auc"] for x in ml_mvn],
            "pca-ctgan": [x["auc"] for x in ml_ctg],
        }
        auc0_list = [x["auc"] for x in ml_real]
        auc_bar_plots(auc0_list, auc1_dict, fig_folder, algo="logreg")

        # SHAP plots using median performing sample for all 3 models (Logistic Regression)
        print(f"  Generating SHAP plots (Logistic Regression)...")

        shap0_list = [x["shap"] for x in ml_real]
        shap1_list = [x["shap"] for x in ml_dbtwin]
        shap1_ctg_list = [x["shap"] for x in ml_ctg]
        shap1_mvn_list = [x["shap"] for x in ml_mvn]

        for shap_mat, idx, model in zip(
            [shap1_list, shap1_ctg_list, shap1_mvn_list],
            [median_idx, median_idx, median_idx],
            ["dbTwin", "PCA-CTGAN", "class-MVN"],
        ):
            shap_corr = compute_shap_correlation(shap0_list, shap_mat, sp_k=sp_k, r_k=r_k)
            print(f"SHAP correlation for model: {model}", shap_corr)
            # dbTwin - median sample

            shap_plots(
                shap0_list[idx],
                shap_mat[idx],
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
            )

        # Generate SHAP plots for all other folds and save to si_shap folder
        if do_si_shap:
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

                    shap_plots(
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
                    )

    print(f"  Done with {dataset} - {trgt}")

print("\nAll DE + ML figures generated successfully!")
