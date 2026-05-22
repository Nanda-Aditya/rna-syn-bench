import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from bench_utils.plot_utils_ml import _s, MODEL_COLORS, MODEL_NAMES, _style_ax

def spearman_top_m(coeff0, coeff1, cc: int = 0, topk: int = 100):
    # Spearman ρ between real and synthetic coefficients restricted to the top-k features of class cc
    ix = np.argsort(np.abs(coeff0[cc, 1:]))[::-1][:topk] + 1
    rho, pval = spearmanr(coeff0[cc, ix], coeff1[cc, ix])
    return rho, pval

# ---------------------------------------------------------------------------
# Shared palette / theme
# ---------------------------------------------------------------------------
_PALETTE = {
    "train": "#3366CC",
    "test": "#E8833A",
    "thresh": "#CC3333",
    "p5": "#E8833A",
    "grid": "#E0E0E0",
    "bg": "#FAFAFA",
    "text": "#333333",
}


def _apply_theme(ax, fontsize_tick=12):
    """Minimal, print-friendly axis styling."""
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AAAAAA")
    ax.spines["bottom"].set_color("#AAAAAA")
    ax.tick_params(axis="both", labelsize=fontsize_tick, colors=_PALETTE["text"])
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=False))
    ax.grid(False)


# ---------------------------------------------------------------------------
# Color blending helper for DCR plots
# ---------------------------------------------------------------------------
def _blend_colors(base_hex: str, shade_hex: str, intensity: float = 0.25) -> str:
    """
    Blend a base color with a shade color.

    Parameters:
    -----------
    base_hex : str
        Base hex color (e.g., "#35de92")
    shade_hex : str
        Shade color to blend in (e.g., "#3366CC" for blue)
    intensity : float
        Amount of shade to blend in (0-1). Default 0.25 for 25% shade.
        0 = pure base color, 1 = pure shade color.

    Returns:
    --------
    str : Blended hex color
    """
    # Remove # and convert to RGB
    base_rgb = tuple(int(base_hex[i:i+2], 16) for i in (1, 3, 5))
    shade_rgb = tuple(int(shade_hex[i:i+2], 16) for i in (1, 3, 5))

    # Linear interpolation
    blended = tuple(
        int(base_rgb[i] * (1 - intensity) + shade_rgb[i] * intensity)
        for i in range(3)
    )

    return f"#{blended[0]:02x}{blended[1]:02x}{blended[2]:02x}"


def _get_dcr_colors(model_name: str, shade_intensity: float = 0.25) -> tuple:
    """
    Get DCR plot colors for a given model.

    Returns (trn_syn_color, tst_syn_color) where:
    - trn_syn: base color blended with blue shade
    - tst_syn: base color blended with red shade
    """
    BLUE_SHADE = "#3366CC"
    RED_SHADE = "#CC3333"

    base_color = MODEL_COLORS.get(model_name, "#666666")

    trn_syn = _blend_colors(base_color, BLUE_SHADE, intensity=shade_intensity)
    tst_syn = _blend_colors(base_color, RED_SHADE, intensity=shade_intensity)

    return trn_syn, tst_syn


# ---------------------------------------------------------------------------
# Figure 1 – DCR distributions  (violin of per-sample DCR, not per-run p5)
# ---------------------------------------------------------------------------
def plot_dcr_distributions(
    expr_dists, model_name, fig_folder="", shade_intensity=0.25, ylimset=None, size="large"
):
    """
    Single-panel DCR violin plot for expression only.
    Shows side-by-side violins for syn→train vs syn→test
    from a single representative run.
    """
    s = _s(size)
    fig, ax = plt.subplots(1, 1,
            figsize=(s["figsize"][0] * 1.2, s["figsize"][1])
                           , dpi=300)

    # Get model-specific colors with blue/red shading
    trn_syn_color, tst_syn_color = _get_dcr_colors(model_name, shade_intensity)

    dists = expr_dists[2] # take 2nd sample
    dist_strn, dist_stst = dists["d_strn"].ravel(), dists["d_stst"].ravel()  # full DCR arrays
    dcr_ratio = np.mean([exp["p5_ratio"] for exp in expr_dists])
    dcr_ratio_std = np.std([exp["p5_ratio"] for exp in expr_dists])

    # --- side-by-side violins ---
    parts = ax.violinplot(
        [dist_strn, dist_stst],
        positions=[0, 1],
        showmedians=True,
        showextrema=False,
        widths=0.6,
    )
    for i, body in enumerate(parts["bodies"]):
        c = trn_syn_color if i == 0 else tst_syn_color
        body.set_facecolor(c)
        body.set_alpha(0.35)
    parts["cmedians"].set_color("black")

    # --- annotation ---
    ax.text(
        0.65,
        0.95,
        f"p5 ratio = {dcr_ratio:.2f} ± {dcr_ratio_std:.2f}",
        transform=ax.transAxes,
        fontsize=s["text"]+0.2,
        fontweight="bold",
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc"),
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["syn→train", "syn→test"], fontsize=s["tick"]+2.5)
    ax.set_ylabel("DCR on scaled counts", fontsize=s["label"])
    _apply_theme(ax, fontsize_tick=s["tick"])

    fig.tight_layout()
    if ylimset is not None:
        ax.set_ylim(ylimset)
    if fig_folder:
        fig.savefig(
            os.path.join(fig_folder, f"dcr_{model_name}.pdf"),
            bbox_inches="tight",
            format="pdf",
            facecolor="white",
        )

def get_coexpression_gene_order(gexp0, n_genes=300):
    """
    Compute hierarchical clustering order from real gene expression data.

    Parameters:
    -----------
    gexp0 : DataFrame
        Real gene expression data (samples x genes)
    n_genes : int
        Number of top variable genes to use

    Returns:
    --------
    top_genes : Index
        Selected gene names (top n_genes by std)
    gene_order : ndarray
        Integer indices that sort genes by hierarchical clustering leaf order
    """
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    from bench_utils.ml_wrapper import mad

    gene_mads = mad(gexp0)
    ix_trn = np.argsort(-gene_mads)[0:n_genes]
    top_genes = gexp0.columns[ix_trn].tolist()

    c_real = np.corrcoef(gexp0.loc[:, top_genes].T)

    # Convert correlation to distance, clip for numerical safety
    dist = np.clip(1 - c_real, 0, 2)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    gene_order = leaves_list(Z)

    return top_genes, gene_order


def plot_coexpression(
    gexp0,
    gexp1,
    model_name="dbTwin",
    n_genes=300,
    size="large",
    fig_folder=None,
):
    """
    Plot correlation matrix heatmap comparing real vs synthetic co-expression.
    Lower triangle shows real data correlations, upper triangle shows synthetic.
    If top_genes and gene_order are provided (from get_coexpression_gene_order),
    genes are reordered by hierarchical clustering for block structure visibility.

    Send gexp0 and gexp1 as precomputed top genes in clustering order! !
    Parameters:
    -----------
    gexp0 : DataFrame
        Real gene expression data (samples x genes)
    gexp1 : DataFrame
        Synthetic gene expression data (samples x genes)
    model_name : str
        Name of synthetic model for title (e.g., "dbTwin", "PCA-CTGAN", ""class-mvn"")
    n_genes : int
        Number of top variable genes to plot (default 300)
    size : str
        Figure size preset ("small" or "large")
    fig_folder : str, optional
        Output folder path """

    # Compute correlation matrices for top genes
    gexp_cols = np.array(gexp0.columns)
    gexpl0=np.log1p(gexp0)
    gexpl1=np.log1p(gexp1)

    c_real = np.corrcoef(gexpl0.T)
    c_syn = np.corrcoef(gexpl1.T)

    # Create combined matrix: lower triangle = real, upper triangle = synthetic
    combined = np.zeros_like(c_real)
    n = c_real.shape[0]

    mask_lower = np.tril(np.ones_like(c_real, dtype=bool), k=-1)
    mask_upper = np.triu(np.ones_like(c_real, dtype=bool), k=1)

    combined[mask_lower] = c_real[mask_lower]
    combined[mask_upper] = c_syn[mask_upper]
    combined[np.eye(n, dtype=bool)] = c_real[np.eye(n, dtype=bool)]

    # Spearman rho between real and synthetic upper triangles
    rho, _ = spearmanr(c_real[mask_upper], c_syn[mask_upper])

    # Fixed symmetric colormap range from real off-diagonal values
    vabs = np.percentile(np.abs(c_real[mask_upper]), 85)
    vmin, vmax = -vabs, vabs

    s = _s(size)
    plt.close()

    fig, ax = plt.subplots(figsize=s["figsize"], dpi=400)

    im = ax.imshow(combined, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="equal")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.ax.tick_params(labelsize=s["tick"])

    ax.plot([0, n - 1], [0, n - 1], "k-", linewidth=0.5, alpha=0.5)

    ax.set_xlabel("Genes", fontsize=s["label"])
    ax.set_ylabel("Genes", fontsize=s["label"])
    ax.set_title( rf"Sp. $\rho$ = {rho:.2g} ", fontsize=s["title"] )

    ax.tick_params(axis="both", labelsize=s["tick"])
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    if fig_folder is not None:
        fname = f"coexp_{model_name.lower().replace(' ', '_').replace('-', '_')}.png"
        plt.savefig(os.path.join(fig_folder, fname), bbox_inches="tight", dpi=500)
    plt.close()


def umap_plots(
    df_trn,
    df_syn_dbt,df_syn_mvn,df_syn_ctg,
    trgt0,
    trgt_dbt,trgt_mvn,trgt_ctg,
    gexp_cols,
    fig_folder,
    comp,
    size="large",legend_str=None,
    model_name=["dbTwin", "class-mvn","pca-ctgan",],
):
    """
    UMAP1 vs UMAP2 scatter for real (df_trn) vs synthetic (df_syn).
    Colors: real = tab:blue, synthetic = tab:orange. Markers encode target class.
    Supports multi-class targets with descriptive string labels in legend.
    Saves {fig_folder}/umap{comp}_{model_name}.pdf.
    Pearson's r of Procrustes-transformed components is shown in the title.

    Parameters
    ----------
    model_name : str, optional
        Model name suffix for output filename (e.g., "dbTwin", ""class-mvn"").
        If empty, saves as umap{comp}.pdf.
    """
    import umap
    trgt0 = pd.Series(trgt0, index=df_trn.index)

    trgt_dbt = pd.Series(trgt_dbt, index=df_syn_dbt.index)
    trgt_mvn = pd.Series(trgt_mvn, index=df_syn_mvn.index)
    trgt_ctg = pd.Series(trgt_ctg, index=df_syn_ctg.index)

    missing = [ c for c in gexp_cols if c not in df_trn.columns or
                                c not in df_syn_dbt.columns
                                or c not in df_syn_ctg.columns
                                or c not in df_syn_mvn.columns]
    if missing:
        raise ValueError(f"Some gexp_cols are missing: {missing[:10]}")

    from bench_utils.ml_wrapper import mad
    gene_mad= mad(df_trn[gexp_cols])
    trn_ix= [int(i) for i in np.where(gene_mad> np.percentile(gene_mad, 40))[0] ]  # top 60 percentile genes
    gexp_cols=np.array(gexp_cols, copy=True)

    X_trn = np.nan_to_num(
        df_trn.loc[:, gexp_cols[trn_ix]].to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    X_syn_dbt = np.nan_to_num(
        df_syn_dbt.loc[:, gexp_cols[trn_ix]].to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    X_syn_ctg = np.nan_to_num( df_syn_ctg.loc[:, gexp_cols[trn_ix]].to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    X_syn_mvn= np.nan_to_num(df_syn_mvn.loc[:, gexp_cols[trn_ix]].to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)

    X_all = np.vstack([X_trn, X_syn_dbt, X_syn_mvn, X_syn_ctg])

    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        n_jobs=8)
    Z_all = reducer.fit_transform(X_all)
    pairs = {"12": (0, 1), "23": (1, 2), "13": (0, 2)}
    i, j = pairs[comp]
    Z_trn = Z_all[: X_trn.shape[0], [i, j]]
    Z_syn_dbt = Z_all[X_trn.shape[0]: X_trn.shape[0] + X_syn_dbt.shape[0], [i, j]]
    Z_syn_mvn = Z_all[X_trn.shape[0] + X_syn_dbt.shape[0]: X_trn.shape[0] +
                                X_syn_dbt.shape[0] + X_syn_mvn.shape[0], [i, j]]
    Z_syn_ctg = Z_all[X_trn.shape[0] + X_syn_dbt.shape[0] + X_syn_mvn.shape[0]:, [i, j]]

    #
    emb_trn = pd.DataFrame(Z_trn, columns=["UMAP1", "UMAP2"], index=df_trn.index)
    emb_dbt = pd.DataFrame(Z_syn_dbt, columns=["UMAP1", "UMAP2"], index=df_syn_dbt.index)
    emb_mvn = pd.DataFrame(Z_syn_mvn, columns=["UMAP1", "UMAP2"], index=df_syn_mvn.index)
    emb_ctg = pd.DataFrame(Z_syn_ctg, columns=["UMAP1", "UMAP2"], index=df_syn_ctg.index)

    # Multi-class marker map: assign different markers for each class
    def _marker_map(series):
        vals = list(pd.unique(series.dropna()))
        # Use distinct markers for up to 10 classes, cycle if more
        markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]
        return {val: markers[idx % len(markers)] for idx, val in enumerate(vals)}

    for model_name, trgt1, emb_syn in zip(["dbTwin","class-mvn",  "pca-ctgan",],
                                          [trgt_dbt, trgt_mvn, trgt_ctg],
                                          [emb_dbt, emb_mvn, emb_ctg]):

        colors = {"real":MODEL_COLORS["real"], "syn": MODEL_COLORS[model_name.lower()]}

        m0 = _marker_map(trgt0)
        m1 = _marker_map(trgt1)
        s = _s(size)
        fig, ax = plt.subplots(figsize=s["figsize"], dpi=300)

        for cls_val in pd.unique(trgt0.dropna()):
            mask = (trgt0 == cls_val).to_numpy()
            ax.scatter(
                emb_trn.loc[mask, "UMAP1"],
                emb_trn.loc[mask, "UMAP2"],
                s=25,
                c=colors["real"],
                marker=m0.get(cls_val, "o"),
                alpha=0.54,
                linewidths=0.5,
                label=f"Real {cls_val}" if not legend_str else f"Real {legend_str} {cls_val}",
            )

        for cls_val in pd.unique(trgt1.dropna()):
            mask = (trgt1 == cls_val).to_numpy()
            ax.scatter(
                emb_syn.loc[mask, "UMAP1"],
                emb_syn.loc[mask, "UMAP2"],
                s=25,
                c=colors["syn"],
                marker=m1.get(cls_val, "o"),
                alpha=0.72,
                linewidths=0.5,
                label=f" {model_name} {legend_str} {cls_val}",
            )

        fig.tight_layout()
        ax.set_xlabel("UMAP" + comp[0], fontsize=s["label"])
        ax.set_ylabel("UMAP" + comp[1], fontsize=s["label"])
        from sklearn.metrics.pairwise import rbf_kernel;
        mmd = np.sqrt(np.abs(rbf_kernel(Z_trn).mean() +
                            rbf_kernel(emb_syn.to_numpy()).mean() -
                    2*rbf_kernel(Z_trn, emb_syn.to_numpy()).mean()))
        print(f"UMAP: Real vs Synthetic\n MMD= {mmd:.2g}")
        ax.set_title(
            f"MMD= {mmd:.2g}",
            fontsize=s["title"],
        )
        ax.tick_params(axis="both", which="major", labelsize=s["tick"])
        ax.legend(
            loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, fontsize=s["legend"]
        )
        _style_ax(ax)

        if len(fig_folder):
            suffix = f"_{model_name}" if model_name else ""
            plt.savefig(
                os.path.join(fig_folder, f"umap_{comp}{suffix}.pdf"),
                bbox_inches="tight",
                format="pdf",
            )
        plt.close()
