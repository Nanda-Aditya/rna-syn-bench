import numpy as np
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from joblib import parallel_backend


def de_computes(G, trgt):
    """Compute differential expression for a single dataframe.

    Args:
        G: DataFrame with gene expression + trgt column
        trgt: Target column name for DE contrast

    Returns:
        Tuple of (genes_min, genes_mod, lfc_all, padj_all, genes_all)
    """
    unique_vals = sorted(G[trgt].unique())  # sorting is important for consistency
    contrast_class = [
        trgt,
        str(unique_vals[1]),  # convert to string
        str(unique_vals[0]),  # convert to string
    ]
    gexp= G.drop(columns=trgt).round().astype(int)

    genes_min, genes_mod, res = gexp_de(
        gexp, G[[trgt]], trgt, contrast_class.copy() )

    genes_min = sorted(genes_min)
    genes_mod = sorted(genes_mod)
    genes_all = sorted(res.index)

    lfc_all = res.loc[genes_all, "log2FoldChange"].values if genes_all else np.array([])
    padj_all = res.loc[genes_all, "padj"].values if genes_all else np.array([])

    return {"genes_min":genes_min,
            "genes_mod":genes_mod,
            "lfc_all":lfc_all,
            "padj_all":padj_all,
            "genes_all":genes_all}


def gexp_de(G, meta, trgt, contrast_class, padj=0.05, lfc_t=1, bmt=10):
    # G is a r x c gene_exp
    # meta is metdata datafrmae
    # trgt is a metadata column with TARGET
    # contrast_class are uniquee values in trgt where the diff expression would be done
    meta=meta.copy()
    meta[trgt] = meta[trgt].astype(str).astype("category")

    dds = DeseqDataSet(counts=G, metadata=meta, design=f"~ {trgt}", refit_cooks=False)
    import time
    tic= time.time()
    with parallel_backend('loky', n_jobs=6):
        dds.deseq2()
    print("time for deseq2 (loky backend)",time.time()-tic)

    stats = DeseqStats(dds, contrast=contrast_class)
    stats.summary()
    res1 = stats.results_df

    # Compute DEG sets (minimum and moderate stringency)
    df = res1.dropna(subset=["padj"]).copy()
    min_mask = df["padj"] < padj
    genes_min = set(df.index[min_mask])

    mod_mask = min_mask
    if lfc_t is not None:
        mod_mask = mod_mask & (df["log2FoldChange"].abs() >= lfc_t)
    if bmt is not None:
        mod_mask = mod_mask & (df["baseMean"] >= bmt)
    genes_mod = set(df.index[mod_mask])

    return genes_min, genes_mod, res1