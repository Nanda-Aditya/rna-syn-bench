import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from typing import Dict, Any, List
import matplotlib.pyplot as plt
from bench_utils.plot_utils_ml import _s, MODEL_COLORS, MODEL_NAMES, _style_ax

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 15,
        "axes.labelsize": 15,
        "axes.titlesize": 15,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
    }
)

def gene_standards(gexp0, gexp1):
    # Log-normalise then z-score both matrices using real-data (gexp0) mean and std
    gexp0 = np.log(gexp0 + 1)
    gexp1 = np.log(gexp1 + 1)
    mean0 = np.mean(gexp0, axis=0)
    std0 = np.std(gexp0, axis=0)
    std0[std0 == 0] = 1  # prevent division by zero on constant columns
    g0 = (gexp0 - mean0) / std0
    g1 = (gexp1 - mean0) / std0
    return g0, g1

def de_bar_plots(de_results_list, fig_folder: str, size="large"):
    """
    Bar plot of DE gene counts (moderate stringency) for real + 3 synthetic models.
    Real gets one bar (total); each synthetic model gets total + overlap-with-real bars.
    """
    if fig_folder:
        os.makedirs(fig_folder, exist_ok=True)

    s = _s(size)
    de_real, de_dbTwin, de_mvn = de_results_list
    n_folds = len(de_real)
    key = "genes_mod"

    # --- Build per-model stats as arrays (n_folds,) ---
    real_genes = [set(de_real[i][key]) for i in range(n_folds)]

    def _counts(model_data):
        total = np.array([len(model_data[i][key]) for i in range(n_folds)])
        overlap = np.array(
            [len(set(model_data[i][key]) & real_genes[i]) for i in range(n_folds)]
        )
        return {"total": total, "overlap": overlap}

    real_de_genes   = _counts(de_real)      # overlap == total (self-overlap)
    dbtwin_de_genes = _counts(de_dbTwin)
    mvn_de_genes    = _counts(de_mvn)
    #ctg_de_genes    = _counts(de_ctg)
    model_names = ["Real", "dbTwin", "class-MVN"]
    models = [
        ("Real",      real_de_genes,   MODEL_COLORS["real"]),
        ("dbTwin",    dbtwin_de_genes, MODEL_COLORS["dbtwin"]),
        ("class-MVN", mvn_de_genes,    MODEL_COLORS["class-mvn"]),
        #("PCA-CTGAN", ctg_de_genes,    MODEL_COLORS["pca-ctgan"]),
    ]

    # --- Plot ---
    fig, ax = plt.subplots(
        figsize=(s["figsize"][0]*1.25 , s["figsize"][1]), dpi=300
    )
    x = np.arange(len(models))
    w = 0.27
    bar_kw = dict(width=w, capsize=3, edgecolor="black", linewidth=0.5)

    for i, (label, stats, color) in enumerate(models):
        mu_t, sd_t = stats["total"].mean(), stats["total"].std()
        mu_o, sd_o = stats["overlap"].mean(), stats["overlap"].std()

        if i == 0:  # Real — single centered bar
            ax.bar(x[0], mu_t, yerr=sd_t, color=color, alpha=0.8, **bar_kw)
        else:       # Synthetic — total + overlap pair
            ax.bar(
                x[i] - w / 2, mu_t, yerr=sd_t, color=color, alpha=0.8,
                label="Total" if i == 1 else "", **bar_kw,
            )
            ax.bar(
                x[i] + w / 2, mu_o, yerr=sd_o, color=color, alpha=0.5,
                label="Overlap" if i == 1 else "", hatch="//", **bar_kw,
            )
            if mu_t > 0:
                pct = (stats["overlap"] / real_de_genes["total"]).mean() * 100
                ax.text(
                    x[i] + w * 0.5 - 0.02,
                    mu_o + 0.063 * max(s["total"].mean() for _, s, _ in models),
                    f"{pct:.1f}%",
                    fontsize=s["tick"]+3.6,fontweight="bold",
                    color=color, ha="left", va="bottom")

    #ax.set_ylabel("# of genes", fontsize=s["label"])
    # ax.set_title(
    #     f"DE Gene Counts (Moderate stringency)", fontsize=s["title"]
    # )
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=s["tick"]+1)
    ax.tick_params(axis='y', labelsize=s["tick"])
    ax.legend(fontsize=s["legend"]+1, frameon=False, loc="upper left")
    _style_ax(ax)

    fig.tight_layout()
    if fig_folder:
        plt.savefig(os.path.join(fig_folder, "de_bars_mod.pdf"),
                     bbox_inches="tight", format="pdf")
    plt.close()

def coeff_plots(coeff1, coeff0, folder, num_class, sp_k=30, size="small"):
    # Scatter of top logistic-regression coefficients (real vs synthetic) with Spearman ρ annotation
    if len(folder):
        figname = os.path.join(folder, "coeffs")
    s = _s(size)
    plt.close()
    for cc in range(num_class):
        fig, ax = plt.subplots(figsize=s["figsize"])
        ix0 = np.argsort(np.abs(coeff0[cc, 1:]))[::-1] + 1
        ax.plot(
            coeff0[cc, ix0[:sp_k]],
            coeff1[cc, ix0[:sp_k]],
            "o",
            markersize=5,
            markerfacecolor="cyan",
            markeredgecolor="k",
        )
        sp_rho = np.round(
            spearmanr(coeff0[cc, ix0[:sp_k]], coeff1[cc, ix0[:sp_k]])[0], 2
        )
        ax.set_title(
            "Top logistic-regression coefficients\n"
            + rf"$\mathbf{{\rho}}$ = $\mathbf{{{sp_rho}}}$",
            fontsize=s["title"],
        )
        ax.set_xlabel("Real data", fontsize=s["label"])
        ax.set_ylabel("Synthetic data", fontsize=s["label"])
        ax.tick_params(axis="both", labelsize=s["tick"])
        _style_ax(ax)
        print(f"Top {sp_k} logistic reg. coefficients -- Spearman rho = {sp_rho}")
        fig.tight_layout()
        if len(folder):
            plt.savefig(figname + f"_{cc}.pdf", bbox_inches="tight", format="pdf")
        plt.close()
        return sp_rho


def lfc_scatter_plots(
    summ,
    fig_folder,
    lo=1,
    hi=99,
    pad=0.5,
    symmetric=False,
    size="small",
    model="dbTwin",
):
    """
    Real-vs-synthetic log2FC scatter for 'minimum' and 'moderate' gene-selection stringency.
    Saves {fig_folder}/min_lfc.png and {fig_folder}/mod_lfc.png.
    """
    if len(fig_folder):
        os.makedirs(fig_folder, exist_ok=True)

    def _to_1d(arr):
        return np.asarray(arr).ravel()

    def _auto_limits(x, y):
        # Percentile-based axis limits, optionally symmetrised around 0
        xy = np.concatenate([np.asarray(x).ravel(), np.asarray(y).ravel()])
        xy = xy[np.isfinite(xy)]
        a, b = np.percentile(xy, [lo, hi])
        a -= pad
        b += pad
        if symmetric:
            m = max(abs(a), abs(b))
            a, b = -m, m
        return (a, b), (a, b)

    s = _s(size)
    plt.close()
    rhos = {}

    # summ is a (14, 2) object array from de_computes:
    #   row 6 = [lfc_min_real, lfc_mod_real], row 7 = [lfc_min_syn, lfc_mod_syn]
    #   row 10 = [pval_min_real, pval_mod_real], row 11 = [pval_min_syn, pval_mod_syn]
    #   row 12 = [padj_min_real, padj_mod_real], row 13 = [padj_min_syn, padj_mod_syn]
    for idx, (label, out_name) in enumerate(
        [
            ("Minimum", "min_lfc.pdf"),
            ("Moderate", "mod_lfc.pdf"),
        ]
    ):
        x = _to_1d(summ[6][idx])
        y = _to_1d(summ[7][idx])
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if x.size == 0:
            continue

        sp_rho = np.round(spearmanr(x, y)[0], 2)
        fig, ax = plt.subplots(figsize=s["figsize"], dpi=300)

        xlim, ylim = _auto_limits(x, y)
        x_disp = np.clip(x, *xlim)
        y_disp = np.clip(y, *ylim)
        is_out = (x_disp != x) | (y_disp != y)

        ax.plot(
            x_disp[~is_out],
            y_disp[~is_out],
            "o",
            markersize=4,
            markerfacecolor="#56B4E9",
            markeredgecolor="0.2",
            alpha=0.85,
        )

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(
            f"log2FC ({label})\n" + rf"Sp. $\mathbf{{\rho}}$ = $\mathbf{{{sp_rho}}}$",
            fontsize=s["title"],
        )
        ax.set_xlabel("Log2FC (real)", fontsize=s["label"])
        ax.set_ylabel("Log2FC (synthetic)", fontsize=s["label"])
        ax.tick_params(axis="both", labelsize=s["tick"])
        _style_ax(ax)
        print(f"{label} logFC -- Spearman rho = {sp_rho}")
        fig.tight_layout()

        if len(fig_folder):
            plt.savefig(
                os.path.join(fig_folder, model + "_" + out_name),
                bbox_inches="tight",
                format="pdf",
            )
        plt.close()
        rhos[label.lower()] = sp_rho

    return rhos

def volcano_plotter(
    lfc_all,
    padj_all,
    genes_all,
    fig_folder="",
    lfc_thresh=1.0,
    padj_thresh=0.05,
    size="large",
    model="dbTwin",
    de_corrs=[0, 0, 0, 0],
    is_are_legend=False,
):
    """
    Create a single volcano plot from DE results.

    Parameters
    ----------
    lfc_all : array-like
        Log2 fold change values for all genes.
    genes_all : array-like
        Gene names corresponding to lfc_all and padj_all.
    padj_all : array-like
        Adjusted p-values (-log10 scale) for all genes.
    fig_folder : str
        Output directory for figures. If empty, plot is not saved.
    lfc_thresh : float
        Log2 fold change threshold for coloring significant genes.
    padj_thresh : float
        Adjusted p-value threshold for significance.
    size : str
        'small' or 'large' figure preset.
    model : str
        Model name for output filename.
    de_corrs : list, optional
        Correlation statistics [js_div, lfc_median, padj_median, lfc_std, padj_std] for title display.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes objects
    """
    lfc_all = np.asarray(lfc_all)
    padj_all = np.asarray(padj_all)
    genes_all = np.asarray(genes_all)

    # Filter out invalid values
    valid = np.isfinite(lfc_all) & np.isfinite(padj_all)
    lfc_all = lfc_all[valid]
    padj_all = padj_all[valid]
    genes_all = genes_all[valid]

    s = _s(size)
    fig, ax = plt.subplots(figsize=s["figsize"], dpi=600)

    # Clip outliers to symmetric limit (min of 5 or max_abs_lfc*1.12)
    max_abs_lfc = np.max(np.abs(lfc_all))
    clip_limit = min(4.5, max_abs_lfc)
    lfc_all = np.clip(lfc_all, -clip_limit, clip_limit)
    pdj_all = np.clip(padj_all, 0, 27)

    # Define significance categories (after clipping)
    is_sig_lfc = np.abs(lfc_all) >= lfc_thresh
    is_sig_padj = padj_all >= -np.log10(padj_thresh)

    sig_up = is_sig_lfc & is_sig_padj & (lfc_all > 0)
    sig_down = is_sig_lfc & is_sig_padj & (lfc_all < 0)
    not_sig = ~(sig_up | sig_down)

    # Plot points - NS first (bottom layer), then colored points on top
    ax.scatter(
        lfc_all[not_sig],
        padj_all[not_sig],
        c="gray",
        s=5,
        alpha=0.2,
        label="NS",
        zorder=1,
    )
    ax.scatter(
        lfc_all[sig_up],
        padj_all[sig_up],
        c="#db8d34",
        s=10,
        alpha=0.36,
        label=f"Up (|LFC|>={lfc_thresh})",
        zorder=2,
    )
    ax.scatter(
        lfc_all[sig_down],
        padj_all[sig_down],
        c="#34d0db",
        s=10,
        alpha=0.36,
        label=f"Down (|LFC|>={lfc_thresh})",
        zorder=2,
    )

    # Symmetric xlim matching clip limits
    ax.set_xlim(-1.12 * clip_limit, 1.12 * clip_limit)
    ax.set_ylim(-0.1, 31)
    ax.set_yticks([0,10,20,30])
    # Threshold lines
    ax.axhline(
        -np.log10(padj_thresh), color="black", linestyle="--", linewidth=0.8, alpha=0.5
    )
    ax.axvline(-lfc_thresh, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(lfc_thresh, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    # Labels and styling
    plot_title=f"\n \n"
    if not model == "real":
        plot_title = (
            f"r(log$_2$FC): {de_corrs[0]:.2f}±{de_corrs[2]:.2f}\n"
            "r(-log$_{10}$(padj)):"+f" {de_corrs[1]:.2f}±{de_corrs[3]:.2f}"
        )

    ax.set_title(plot_title, fontsize=s["title"]-0.5)
    ax.set_xlabel("log$_2$ Fold Change", fontsize=s["label"]-1.5)
    ax.set_ylabel("-log$_{10}$ (adjusted p value)", fontsize=s["label"]-1.5)
    ax.tick_params(axis="both", labelsize=s["tick"])
    _style_ax(ax)
    if is_are_legend:
        legend = ax.legend(
            fontsize=s["legend"],
            frameon=False,
            handletextpad=0.1,
            labelspacing=0.3,
            borderpad=0,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),  # just outside the right edge
        )
        legend.set_zorder(10)  # bring legend above the dashed lines

    fig.tight_layout()  # move tight_layout to AFTER legend


    if fig_folder:
        os.makedirs(fig_folder, exist_ok=True)
        fname = os.path.join(fig_folder, f"volcano_{model}.png")
        plt.savefig(fname, bbox_inches="tight", format="png", dpi=800)
        plt.close()

    return fig, ax
