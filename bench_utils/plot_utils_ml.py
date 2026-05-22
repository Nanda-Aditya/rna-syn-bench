import os
import numpy as np
from scipy.stats import spearmanr
from typing import Dict, Any
import matplotlib
from bench_utils.de_ml_utils import map_ensembl_to_symbols
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Publication style — two figure-size presets
#   small: 3.5 × 3.0 in  (single-column journal / whitepaper)
#   large: 5.0 × 4.0 in  (double-column / wider panels)
# ---------------------------------------------------------------------------
_SMALL: Dict[str, Any] = {
    "figsize": (3.5, 3.0),
    "title": 14,
    "label": 12,
    "tick": 11,
    "legend": 11,
    "annot": 12,
    "text": 11,
}
_LARGE: Dict[str, Any] = {
    "figsize": (5.0, 4.0),
    "title": 16,
    "label": 15,
    "tick": 13.5,
    "legend": 14,
    "annot": 14,
    "text": 14,
}


def _s(size: str) -> Dict[str, Any]:
    """Return style preset dict: 'small' → 3.5×3 in, 'large' → 5×4 in."""
    return _LARGE if size == "large" else _SMALL


# One-time baseline rcParams — no individual function should call plt.rcParams.update()
glb_fnt= 15.7
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": glb_fnt,
        "axes.labelsize": glb_fnt,
        "axes.titlesize": glb_fnt,
        "xtick.labelsize": glb_fnt,
        "ytick.labelsize": glb_fnt,
        "legend.fontsize": glb_fnt,
    }
)

# ---------------------------------------------------------------------------
# Model color scheme - consistent across all plots
# ---------------------------------------------------------------------------
MODEL_COLORS = {
    "real": "#333333",  # dark gray
    "dbtwin": "#35de92",  # light blue-greenish (main model)
    "pca-ctgan": "#d19fbf",  # magenta (ctgan baseline)
    "class-mvn": "#cfbaa0",  # browinsh (mvn baseline)
}

MODEL_NAMES = {
    "dbtwin": "dbTwin",
    "pca-ctgan": "PCA-CTGAN",
    "class-mvn": "class-MVN",
}


def _style_ax(ax: plt.Axes, grid: bool = False) -> None:
    """Remove top/right spines; optionally add a light dashed grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(alpha=0.3, linestyle="--")
    else:
        ax.grid(False)


# ---------------------------------------------------------------------------
# Helper / non-plotting utilities
# ---------------------------------------------------------------------------


def get_shap_class(shap_values: np.ndarray, cls: int = 0) -> np.ndarray:
    # Extract per-class SHAP matrix: 2-D for binary classification, 3-D slice for multiclass
    if shap_values.ndim == 2:
        return shap_values
    elif shap_values.ndim == 3:
        return shap_values[:, :, cls]
    else:
        raise ValueError(f"Unexpected SHAP shape: {shap_values.shape}")

def metric_bar_plots(
    metric0, metric_dict, folder, metric_name="Metric", size="large", algo=None
):
    """
    Bar plot comparing metrics across 3 synthetic models with error bars.
    Reusable for AUC, PCR, or any other metric.

    Parameters:
    -----------
    metric0 : array-like (n_samples,) or (n_samples, n_classes)
        Real→Real metric values (baseline)
    metric_dict : dict
        {"dbtwin": metric1, "pca_ctgan": metric1_ctg, ""class-mvn"": metric1_shf}
        Each value is array of shape (n_samples,) for binary or (n_samples, n_classes) for multi-class
    folder : str
        Output folder
    metric_name : str
        Name of metric for ylabel and title (e.g., "ROC-AUC", "PCR")
    size : str
        Figure size preset
    """
    s = _s(size)

    # Determine if multi-class (2D array) or binary (1D array)
    metric0_arr = np.asarray(metric0)
    is_multiclass = metric0_arr.ndim > 1 and metric0_arr.shape[1] > 1

    models = ["dbtwin", "class-mvn", "pca-ctgan"]
    model_labels = [MODEL_NAMES[m] for m in models]
    colors = [MODEL_COLORS[m] for m in models]

    # Compute means and stds
    # For multi-class: take mean across classes first, then mean/std across samples
    if is_multiclass:
        mean0 = np.mean(metric0_arr.mean(axis=1))
        std0 = np.std(metric0_arr.mean(axis=1))
        model_means = []
        model_stds = []
        for m in models:
            arr = np.asarray(metric_dict[m])
            model_means.append(np.mean(arr.mean(axis=1)))
            model_stds.append(np.std(arr.mean(axis=1)))
    else:
        mean0 = np.mean(metric0_arr)
        std0 = np.std(metric0_arr)
        model_means = []
        model_stds = []
        for m in models:
            arr = np.asarray(metric_dict[m])
            model_means.append(np.mean(arr))
            model_stds.append(np.std(arr))

    fig, ax = plt.subplots(figsize=(s["figsize"][0] * 1.2, s["figsize"][1]))

    # All values including real baseline
    all_means = [mean0] + model_means
    all_stds = [std0] + model_stds
    all_labels = ["Real→Real"] + model_labels
    all_colors = [MODEL_COLORS["real"]] + colors

    x = np.arange(len(all_labels))*1.2
    bars = ax.bar(
        x,
        all_means,
        yerr=all_stds,
        capsize=4,
        color=all_colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_ylabel(metric_name, fontsize=s["label"])
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, fontsize=s["tick"], rotation=15, ha="right")
    ax.tick_params(axis="y", labelsize=s["tick"])
    _style_ax(ax)

    # Set y-axis to start from reasonable value (e.g., 0.4 for AUC)
    if "AUC" in metric_name:
        ax.set_ylim(0.4, min(1.05,max(all_means) * 1.2))
    elif "PCR" in metric_name:
        ax.set_ylim(0, max(all_means) * 1.2)

    fig.tight_layout()
    if len(folder):
        if algo is None:
            fname = f"{metric_name.lower().replace('-', '_')}_bar_plots.pdf"
        else:
            fname = f"{metric_name.lower().replace('-', '_')}_{algo}_bar_plots.pdf"

        plt.savefig(os.path.join(folder, fname), bbox_inches="tight", format="pdf")
    plt.close()


def auc_bar_plots(auc0, auc1_dict, folder, size="large", algo="logistic"):
    """
    Wrapper for metric_bar_plots specifically for AUC.

    Parameters:
    -----------
    auc0 : array-like (n_samples,) or (n_samples, n_classes)
        Real→Real AUC values
    auc1_dict : dict
        {"dbtwin": auc1, "pca-ctgan": auc1_ctg, ""class-mvn"": auc1_shf}
    folder : str
        Output folder
    size : str
        Figure size preset
    """
    metric_bar_plots(
        auc0, auc1_dict, folder, metric_name="ROC-AUC", size=size, algo=algo
    )


def pcr_bar_plots(pcr0, pcr1_dict, folder, size="large"):
    """
    Wrapper for metric_bar_plots specifically for PCR.

    Parameters:
    -----------
    pcr0 : array-like (n_samples,) or (n_samples, n_classes)
        Real→Real PCR values
    pcr1_dict : dict
        {"dbtwin": pcr1, "pca_ctgan": pcr1_ctg, ""class-mvn"": pcr1_shf}
    folder : str
        Output folder
    size : str
        Figure size preset
    """
    metric_bar_plots(pcr0, pcr1_dict, folder, metric_name="PCR", size=size)


def shap_plots(
    shap0,
    shap1,
    fig_folder,
    df_cols,
    syn_color=None,
    model_name="Synthetic",
    top_k=11,
    sp_k=100,
    r_k=30,
    size="large",
    map_genes: bool = True,
    sp_rho=None,
    sp_rho_std=None,
    pearson_r=None,
    pearson_r_std=None,
    dataset_name=None,
    classes=None,
    orig_classes_=None,
):
    """
    Grouped bar chart of mean |SHAP| for top features.
    Spearman rho and Pearson r measure real-vs-synthetic agreement.

    Parameters:
    -----------
    shap0 : array-like
        Real SHAP values, shape (n_samples, n_features) or (n_samples, n_classes, n_features)
    shap1 : array-like
        Synthetic SHAP values, same shape as shap0
    fig_folder : str
        Output folder path
    df_cols : array-like
        Feature names (may contain Ensembl IDs)
    syn_color : str, optional
        Color for synthetic bars. If None, uses tab:orange
    model_name : str, optional
        Name of synthetic model for legend
    top_k : int
        Number of top features to plot
    sp_k : int
        Number of top features for Spearman correlation (default 100)
    r_k : int
        Number of top features for Pearson correlation (default 30)
    size : str
        Figure size preset
    map_genes : bool, optional
        If True, convert Ensembl IDs to gene symbols using map_ensembl_to_symbols.
        Default is True.
    sp_rho : float or array-like, optional
        Spearman correlation median value(s) for title
    sp_rho_std : float or array-like, optional
        Spearman correlation std value(s) for title
    pearson_r : float or array-like, optional
        Pearson correlation median value(s) for title
    pearson_r_std : float or array-like, optional
        Pearson correlation std value(s) for title
    dataset_name : str, optional
        Dataset identifier for cache lookup (e.g., 'tcga_luad', 'sepsis', 'risk_ped').
        Passed to map_ensembl_to_symbols.

    Returns:
    --------
    sp_rho : float
        Spearman correlation coefficient
    """
    # Auto-detect if multiclass from array shape
    shap0_arr = np.asarray(shap0)
    shap1_arr = np.asarray(shap1)

    # Determine if multiclass: 3D array with shape (n_samples, n_features, n_classes)
    is_multiclass = shap0_arr.ndim == 3 and shap0_arr.shape[2] > 1

    # Set colors
    real_color = MODEL_COLORS["real"]
    syn_color = syn_color if syn_color is not None else "tab:orange"

    # Convert Ensembl IDs to gene symbols if requested
    df_cols_arr = np.asarray(df_cols)
    if map_genes:
        gene_labels = map_ensembl_to_symbols(
            df_cols_arr.tolist(), dataset_name=dataset_name
        )
    else:
        gene_labels = df_cols_arr.tolist()
    gene_labels = np.asarray(gene_labels)

    s = _s(size)
    plt.close()

    # Determine class information
    if is_multiclass:
        n_classes = shap0_arr.shape[2]
        if classes is None:
            classes = np.arange(n_classes)
        if orig_classes_ is None:
            orig_classes_ = classes
    else:
        n_classes = 1
        classes = [0]
        orig_classes_ = [""]

    # Parse correlation parameters
    # Check if we have the new format with separate median and std values
    has_correlations = sp_rho is not None and pearson_r is not None

    # Create separate figure for each class
    for cls_idx, cls_label in zip(classes, orig_classes_):
        # Extract class-specific SHAP values
        if is_multiclass:
            shap0_cls = shap0_arr[:, :, cls_idx]
            shap1_cls = shap1_arr[:, :, cls_idx]
        else:
            shap0_cls = shap0_arr
            shap1_cls = shap1_arr

        # Compute mean absolute SHAP values for this class
        mas0 = np.abs(shap0_cls).mean(axis=0)
        mas1 = np.abs(shap1_cls).mean(axis=0)

        fig, ax = plt.subplots(figsize=s["figsize"])
        ix0 = np.argsort(mas0)[::-1]
        xvec = np.arange(0, 4 * top_k, step=4)
        ax.bar(xvec, mas0[ix0[:top_k]], 0.89, label="Real", color=real_color)
        ax.bar(xvec + 1.2, mas1[ix0[:top_k]], 0.89, label=model_name, color=syn_color)
        ax.set_xticks(xvec + 0.6)
        ax.set_xticklabels(
            gene_labels[ix0[:top_k]], rotation=35, ha="right", fontsize=s["tick"]
        )
        ax.set_ylabel("Mean |SHAP|", fontsize=s["label"])
        ax.tick_params(axis="y", labelsize=s["tick"])

        # Determine title based on correlation parameters
        if has_correlations:
            # Use provided median and std values for both correlations
            if is_multiclass:
                cls_sp_mean = sp_rho[cls_idx] if hasattr(sp_rho, '__len__') else sp_rho
                cls_sp_std = sp_rho_std[cls_idx] if hasattr(sp_rho_std, '__len__') else sp_rho_std
                cls_r_mean = pearson_r[cls_idx] if hasattr(pearson_r, '__len__') else pearson_r
                cls_r_std = pearson_r_std[cls_idx] if hasattr(pearson_r_std, '__len__') else pearson_r_std
            else:
                cls_sp_mean = sp_rho
                cls_sp_std = sp_rho_std
                cls_r_mean = pearson_r
                cls_r_std = pearson_r_std
            
            sp_display = np.round(cls_sp_mean, 2)
            sp_std_display = np.round(cls_sp_std, 3)
            r_display = np.round(cls_r_mean, 2)
            r_std_display = np.round(cls_r_std, 3)
            
            # Title with both correlations: r (top r_k) and Spearman ρ (top sp_k)
            ax.set_title(
                rf"$r$ = ${r_display} \pm {r_std_display}$ (top {r_k})" + "\n" +
                rf"Sp. $\rho$ = ${sp_display} \pm {sp_std_display}$ (top {sp_k})",
                fontsize=s["title"],
            )
        else:
            # Compute fold-wise correlations
            rho = np.round(spearmanr(mas0[ix0[:sp_k]], mas1[ix0[:sp_k]])[0], 3)
            r = np.round(np.corrcoef(mas0[ix0[:r_k]], mas1[ix0[:r_k]])[0, 1], 3)
            ax.set_title(
                rf"$r$ = ${r}$ (top {r_k})" + "\n" +
                rf"Sp. $\rho$ = ${rho}$ (top {sp_k})",
                fontsize=s["title"],
            )

        ax.legend(fontsize=s["legend"], frameon=False)
        _style_ax(ax)

        # Determine filename based on whether it's multiclass
        if is_multiclass:
            figname = os.path.join(fig_folder, f"SHAP_{model_name}_{cls_label}.pdf")
            if has_correlations:
                print(f"  Class {cls_label} -- r = {r_display} +/- {r_std_display} (top {r_k}), Sp. rho = {sp_display} +/- {sp_std_display} (top {sp_k})")
            else:
                print(f"  Class {cls_label} -- r = {r} (top {r_k}), Sp. rho = {rho} (top {sp_k})")
        else:
            figname = os.path.join(fig_folder, f"SHAP_{model_name}.pdf")
            if has_correlations:
                print(f"r = {r_display} +/- {r_std_display} (top {r_k}), Sp. rho = {sp_display} +/- {sp_std_display} (top {sp_k})")
            else:
                print(f"r = {r} (top {r_k}), Sp. rho = {rho} (top {sp_k})")

        fig.tight_layout()
        if len(fig_folder):
            os.makedirs(fig_folder, exist_ok=True)
            plt.savefig(figname, bbox_inches="tight", format="pdf")
        plt.close()

    # Return fold-wise rho for the first class (or only class for binary)
    if is_multiclass:
        shap0_first = shap0_arr[:, :, 0]
        shap1_first = shap1_arr[:, :, 0]
    else:
        shap0_first = shap0_arr
        shap1_first = shap1_arr
    mas0_first = np.abs(shap0_first).mean(axis=0)
    mas1_first = np.abs(shap1_first).mean(axis=0)
    ix0_first = np.argsort(mas0_first)[::-1]
    return_rho = np.round(spearmanr(mas0_first[ix0_first[:sp_k]], mas1_first[ix0_first[:sp_k]])[0], 3)
    return return_rho

def fig_cv_sparsity(bench_cv_trn, C_values, trgt,dataset, c_opt):

    # --- data ---
    auc_vec = [x["mean_auc"] for x in bench_cv_trn[trgt][0]]
    mn_vec = [x["mean_n_active"] for x in bench_cv_trn[trgt][0]]
    log_C = np.log10(C_values)
     # --- figure with two panels ---
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    fig.subplots_adjust(wspace=0.42)
    # panel A: log(C) vs AUC
    # ax = axes[0]
    BLUE="tab:blue"
    ax = axes[0]
    ax.plot(log_C, auc_vec, "o", ms=4, color=BLUE, alpha=0.85, lw=0)
    ax.set_xlabel(r"$\log(C)$")
    ax.set_ylabel("Mean CV AUC")
    ax.set_ylim(0.45, None)
    ax.axvline(np.log10(c_opt), color="tab:orange", lw=1, ls="--")
    # panel B: log(C) vs n_active
    ax = axes[1]
    ax.plot(log_C, mn_vec, "o", ms=4, color=BLUE, alpha=0.85, lw=0)
    ax.set_xlabel(r"$\log(C)$")
    ax.set_ylabel("Mean active features")
    ax.set_title("B", loc="left", fontweight="bold", pad=4)
    ax.axvline(np.log10(c_opt), color="tab:orange", lw=1, ls="--")


    ## SAVE
    fig_folder= "figs/cv_sparsity/"
    os.makedirs(fig_folder, exist_ok=True)
    fig.savefig(os.path.join(fig_folder, f"fig_cv_sparsity_{dataset}_{trgt}.pdf"))  # vector for journal
    fig.savefig(os.path.join(fig_folder,f"fig_cv_sparsity_{dataset}_{trgt}.png"))
    plt.close()

