import json
import os

import numpy as np
from scipy.stats import spearmanr

"""
Correlation computer for differential expression results between real and synthetic data.
"""

def compute_de_corrs(de_real, de_syn, clip_max=30, mod_genes=True):
    """
    Compute mean correlation between real and synthetic DE results.

    Returns a single scalar that is the average of:
    - Mean LFC correlation across all comparisons
    - Mean -log10(pval) correlation across all comparisons

    Parameters
    ----------
    de_real : list
        List of real differential expression result dicts with "lfc_all" and "pval_all" keys
    de_syn : list
        List of synthetic differential expression result dicts matching de_real structure
    clip_max : float
        Maximum value to clip -log10(pvals) to (default 30)
    mod_genes - simply computes correlations only over moderate genes
    Returns
    -------
    float
        Scalar mean correlation (average of LFC and pval correlations)
    """
    n = len(de_real)
    lfc_corrs = []
    pval_corrs = []
    jsd_div = []

    for i in range(n): # iterate over folds
        syn_aligned = align_de_to_real(de_real[i], de_syn[i])
        aligned_genes = syn_aligned["genes_aligned"]
        # Build real arrays aligned to the same common gene set
        real_gene_to_idx = {gene: idx for idx, gene in enumerate(de_real[i]["genes_all"])}
        real_idx = [real_gene_to_idx[g] for g in aligned_genes]

        lfc_real = np.nan_to_num(np.array(de_real[i]["lfc_all"])[real_idx], nan=0.0)
        pval_real_raw = np.nan_to_num(np.array(de_real[i]["padj_all"])[real_idx], nan=1.0)

        lfc_syn = np.nan_to_num(syn_aligned["lfc_aligned"], nan=0.0)
        pval_syn_raw = np.nan_to_num(syn_aligned["padj_aligned"], nan=1.0)

        if mod_genes:
            mod_genes_i = de_real[i]["genes_mod"]
            ix_mod = np.isin(aligned_genes, mod_genes_i)
        else:
            ix_mod = np.arange(len(aligned_genes))

        lfc_corr = np.corrcoef(lfc_real[ix_mod], lfc_syn[ix_mod])[0, 1]
        lfc_corrs.append(lfc_corr)

        pval_real = np.clip(-np.log10(pval_real_raw), 0, clip_max)
        pval_syn = np.clip(-np.log10(pval_syn_raw), 0, clip_max)
        pval_corr = np.corrcoef(pval_real[ix_mod], pval_syn[ix_mod])[0, 1]
        pval_corrs.append(pval_corr)

    mean_lfc_corr = np.mean(lfc_corrs)
    mean_pval_corr = np.mean(pval_corrs)

    return mean_lfc_corr, mean_pval_corr, np.std(lfc_corrs), np.std(pval_corrs)

def align_de_to_real(de_real, de_syn):
    """
    Align real and synthetic DE results to the same set of genes."""

    genes_real = np.array(de_real["genes_all"])
    genes_syn = np.array(de_syn["genes_all"])
    lfc_syn = np.array(de_syn["lfc_all"])
    pval_syn = np.array(de_syn["padj_all"])

    syn_gene_to_idx = {gene: idx for idx, gene in enumerate(genes_syn)}

    aligned_genes = []
    aligned_lfc_syn = []
    aligned_pval_syn = []

    for gene in genes_real:
        if gene in syn_gene_to_idx:
            syn_idx = syn_gene_to_idx[gene]
            aligned_genes.append(gene)           # ← only append when found
            aligned_lfc_syn.append(lfc_syn[syn_idx])
            aligned_pval_syn.append(pval_syn[syn_idx])

    return {
        "lfc_aligned": np.array(aligned_lfc_syn),
        "padj_aligned": np.array(aligned_pval_syn),
        "genes_aligned": np.array(aligned_genes),  # ← common genes only
    }


def compute_shap_correlation(shap0_list, shap1_list, sp_k=150, r_k=150):
    """
    Compute Spearman and Pearson correlations of mean |SHAP| values between real and synthetic.

    For binary, computes a single correlation per fold.
    For multiclass, computes per-class correlations per fold and returns
    a (n_classes,) array of median correlations across folds.

    Parameters
    ----------
    shap0_list : list of arrays
        List of 5 real SHAP arrays. Each array is (n_samples, n_features) for binary
        or (n_samples, n_features, n_classes) for multiclass
    shap1_list : list of arrays
        List of 5 synthetic SHAP arrays matching shap0_list structure
    sp_k : int
        Number of top features (by real mean |SHAP|) to use for Spearman correlation
    r_k : int
        Number of top features (by real mean |SHAP|) to use for Pearson correlation

    Returns
    -------
    dict
        - "median_sp_rho": scalar (binary) or (n_classes,) array of median Spearman correlations
        - "median_r": scalar (binary) or (n_classes,) array of median Pearson correlations  
        - "std_sp_rho": scalar (binary) or (n_classes,) array of std across folds for Spearman
        - "std_r": scalar (binary) or (n_classes,) array of std across folds for Pearson
        - "raw_correlations_sp": list of scalars (binary) or list of (n_classes,) arrays for Spearman
        - "raw_correlations_r": list of scalars (binary) or list of (n_classes,) arrays for Pearson
    """
    shap0_probe = np.asarray(shap0_list[0])
    is_multiclass = shap0_probe.ndim == 3 and shap0_probe.shape[2] > 1

    if is_multiclass:
        n_classes = shap0_probe.shape[2]
        # Each element: (n_classes,) array of per-class correlations
        fold_corrs_sp = []
        fold_corrs_r = []

        for shap0, shap1 in zip(shap0_list, shap1_list):
            shap0_arr = np.asarray(shap0)
            shap1_arr = np.asarray(shap1)
            class_corrs_sp = np.empty(n_classes)
            class_corrs_r = np.empty(n_classes)
            for c in range(n_classes):
                mas0_c = np.abs(shap0_arr[:, :, c]).mean(axis=0)
                mas1_c = np.abs(shap1_arr[:, :, c]).mean(axis=0)
                
                # Spearman correlation
                ix0_sp = np.argsort(mas0_c)[::-1][:sp_k]
                if np.count_nonzero(mas1_c[ix0_sp]) == 0:
                    print("WARNING: model shap vals are all zeros for Spearman")
                    if np.count_nonzero(mas0_c[ix0_sp]) == 0:
                        print(f"WARNING: class {c} — real SHAP all zeros, skipping")
                        class_corrs_sp[c] = np.nan  # not 0
                    else:
                        class_corrs_sp[c] = 0
                else:
                    class_corrs_sp[c] = spearmanr(mas0_c[ix0_sp], mas1_c[ix0_sp])[0]
                
                # Pearson correlation
                ix0_r = np.argsort(mas0_c)[::-1][:r_k]
                if np.count_nonzero(mas1_c[ix0_r]) == 0:
                    print("WARNING: model shap vals are all zeros for Pearson")
                    if np.count_nonzero(mas0_c[ix0_r]) == 0:
                        print(f"WARNING: class {c} — real SHAP all zeros, skipping")
                        class_corrs_r[c] = np.nan  # not 0
                    else:
                        class_corrs_r[c] = 0
                else:
                    class_corrs_r[c] = np.corrcoef(mas0_c[ix0_r], mas1_c[ix0_r])[0, 1]
            
            fold_corrs_sp.append(class_corrs_sp)
            fold_corrs_r.append(class_corrs_r)

        # (n_folds, n_classes)
        stacked_sp = np.vstack(fold_corrs_sp)
        stacked_r = np.vstack(fold_corrs_r)

        return {
            "median_sp_rho": np.nanmedian(stacked_sp, axis=0),   # (n_classes,)
            "median_r": np.nanmedian(stacked_r, axis=0),        # (n_classes,)
            "std_sp_rho": np.nanstd(stacked_sp, axis=0),        # (n_classes,)
            "std_r": np.nanstd(stacked_r, axis=0),              # (n_classes,)
            "raw_correlations_sp": fold_corrs_sp,                # list of (n_classes,) arrays
            "raw_correlations_r": fold_corrs_r,                  # list of (n_classes,) arrays
        }
    else:
        correlations_sp = []
        correlations_r = []
        for shap0, shap1 in zip(shap0_list, shap1_list):
            shap0_arr = np.asarray(shap0)
            shap1_arr = np.asarray(shap1)
            mas0 = np.abs(shap0_arr).mean(axis=0)
            mas1 = np.abs(shap1_arr).mean(axis=0)
            
            # Spearman correlation
            ix0_sp = np.argsort(mas0)[::-1][:sp_k]
            if np.count_nonzero(mas1) == 0:
                print("WARNING: model shap vals are all zeros for Spearman")
                corr_sp = 0
            else:
                corr_sp = spearmanr(mas0[ix0_sp], mas1[ix0_sp])[0]
            correlations_sp.append(corr_sp)
            
            # Pearson correlation
            ix0_r = np.argsort(mas0)[::-1][:r_k]
            if np.count_nonzero(mas1) == 0:
                print("WARNING: model shap vals are all zeros for Pearson")
                corr_r = 0
            else:
                corr_r = np.corrcoef(mas0[ix0_r], mas1[ix0_r])[0, 1]
            correlations_r.append(corr_r)

        return {
            "median_sp_rho": np.nanmedian(correlations_sp),
            "median_r": np.nanmedian(correlations_r),
            "std_sp_rho": np.nanstd(correlations_sp),
            "std_r": np.nanstd(correlations_r),
            "raw_correlations_sp": correlations_sp,
            "raw_correlations_r": correlations_r,
        }


def map_ensembl_to_symbols(ensembl_ids, dataset_name=None):
    """
    Takes a list of Ensembl IDs, returns a list of gene symbols in the same order.
    Unmapped IDs are kept as-is. Strips version suffixes (e.g., .14) for mygene query.

    Uses persistent cache file per dataset. First call queries mygene.info and saves
    cache; subsequent calls load from cache.

    Parameters
    ----------
    ensembl_ids : list
        List of Ensembl IDs to map.
    dataset_name : str, optional
        Dataset identifier (e.g., 'tcga_luad', 'sepsis', 'risk_ped').
        If provided, uses/creates cache at data/gene_mapping_cache/{dataset}_ensembl_to_symbol.json

    Returns
    -------
    list
        Gene symbols in same order as input (unmapped IDs kept as-is).
    """
    import mygene
    import re

    # Strip version suffixes (.XX) for mygene query
    clean_ids = [re.sub(r"^(ENSG\d+)\.\d+$", r"\1", eid) for eid in ensembl_ids]

    # Use cache if dataset_name provided
    if dataset_name is not None:
        cache_path = os.path.join(
            "data", "gene_mapping_cache", f"{dataset_name}_ensembl_to_symbol.json"
        )
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                ensembl_to_symbol = json.load(f)
            # Return from cache
            return [
                ensembl_to_symbol.get(clean_id, orig_id)
                for clean_id, orig_id in zip(clean_ids, ensembl_ids)
            ]

    # Query mygene.info
    mg = mygene.MyGeneInfo()
    result = mg.querymany(
        clean_ids,
        scopes="ensembl.gene",
        fields="symbol",
        species="human",
        as_dataframe=True,
    )

    ensembl_to_symbol = result["symbol"].dropna().to_dict()

    # Save cache if dataset_name provided
    if dataset_name is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(ensembl_to_symbol, f)

    # Preserve order, fallback to original ID if unmapped
    return [
        ensembl_to_symbol.get(clean_id, orig_id)
        for clean_id, orig_id in zip(clean_ids, ensembl_ids)
    ]
