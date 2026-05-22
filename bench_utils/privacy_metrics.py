"""
Privacy / re-identification risk metrics for synthetic clinico-transcriptomic data.

Returns numpy arrays throughout — no dicts.

Functions
---------
compute_privacy_metrics  – caller; returns dcr_ratio_clinical, dcr_clinical_stats,
                           dcr_expr_stats, nndr_p5
dcr_clinical             – Gower-based DCR for mixed clinical columns
_knn_euclidean           – Euclidean k=2 NN distances (used for DCR + NNDR on counts)
_dcr_bootstrap           – bootstrap-subsampled DCR with p5/p50 aggregation
"""


import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors


# ──────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────

def _classify_cols(df, cols):
    """Split cols into (numeric, categorical).  Drop constant or all-unique."""
    num, cat = [], []
    for c in cols:
        s = df[c]
        nuniq = s.nunique(dropna=True)
        if nuniq <= 1 or nuniq == len(s):          # constant or all-unique → skip
            continue
        if pd.api.types.is_numeric_dtype(s):
            num.append(c)
        else:
            cat.append(c)
    return num, cat

def _prep_expression(df, gexp_cols, fit_stats=None):
    """log1p → z-score in gene-space.  Returns (matrix, fit_stats)."""
    X = np.log1p(df[gexp_cols].fillna(0).values.astype(float))
    if fit_stats is None:
        mu, sd = X.mean(0), X.std(0)
        sd[sd == 0] = 1.0
        fit_stats = (mu, sd)
    mu, sd = fit_stats
    return (X - mu) / sd, fit_stats

def _knn_euclidean(synth, ref, k=2):
    """Return (n_synth, k) matrix of Euclidean distances to k nearest neighbours."""
    nn = NearestNeighbors(n_neighbors=min(k, len(ref)), metric="euclidean", algorithm="auto")
    nn.fit(ref)
    dists, _ = nn.kneighbors(synth)
    return dists                                                  # (n_synth, k)

def compute_expression_privacy_metrics(df_trn, df_syn, df_tst,
            gexp_cols, n_sub_rounds=5, seed=42):

    E_trn, estats = _prep_expression(df_trn, gexp_cols)
    E_syn, _      = _prep_expression(df_syn, gexp_cols, fit_stats=estats)
    E_tst, _      = _prep_expression(df_tst, gexp_cols, fit_stats=estats)

    n = len(E_tst)
    rng = np.random.RandomState(seed)
    k_dcr=1
    d_strn_runs, d_stst_runs = [], []
    for _ in range(n_sub_rounds):
        idx = rng.choice(len(E_trn), size=n, replace=False)
        E_trn_sub = E_trn[idx]
        d_strn_runs.append(_knn_euclidean(E_syn, E_trn_sub, k=k_dcr))
        d_stst_runs.append(_knn_euclidean(E_syn, E_tst, k=k_dcr))

    d_strn = np.mean(d_strn_runs, axis=0)
    d_stst = np.mean(d_stst_runs, axis=0)

    expr_dists = {"d_strn": d_strn, "d_stst": d_stst,
                "p5_ratio": np.percentile(d_strn,5)/ np.percentile(d_stst, 5)}

    return expr_dists