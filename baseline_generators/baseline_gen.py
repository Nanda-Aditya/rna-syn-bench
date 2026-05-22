import numpy as np
from sklearn.decomposition import PCA
from sklearn.utils.extmath import randomized_svd
from ctgan import CTGAN
import time
import pandas as pd

def pca_ctgan(phtp, gexp, pca_thres=0.5, ctgan_epochs=300):
    """
    Wrapper for SDV's CTGAN. You need to install CTGAN via SDV to use this.
    phtp and gexp must be matched phenotype and expression data-frames

    Performs synthetic data generation by combining principal component analysis (PCA) and
    CTGAN. This function first reduces
    dimensionality of gene expression data using PCA, combines it with clinical data, and
    generates synthetic data using the CTGAN model. The resulting synthetic data is then
    transformed back to gene-space.
    """
    n_samples = gexp.shape[0]
    n_components = max(2, int(n_samples * pca_thres))

    # PCA-reduce gene expression
    gexpl1 = np.log(gexp + 1)
    pca = PCA(n_components=n_components)
    gexp_pca_ = pca.fit_transform(gexpl1)

    pca_cols = [f"PC{i + 1}" for i in range(n_components)]
    gexp_pca = pd.DataFrame(gexp_pca_, index=gexp.index, columns=pca_cols)

    # combine clinical + PCA
    df_combo = phtp.join(gexp_pca, how="inner")

    # fit CTGAN
    discrete_cols = [
        c
        for c in phtp.columns
        if phtp[c].dtype == "object" or phtp[c].dtype.name == "category"
    ]
    for c in df_combo.columns:
        df_combo[c] = df_combo[c].fillna(
            df_combo[c].mode()[0]
            if df_combo[c].dtype == "object"
            else df_combo[c].median()
        )

    synth = CTGAN(epochs=ctgan_epochs, verbose=True)
    synth.fit(df_combo, discrete_columns=discrete_cols)
    df_out = synth.sample(n_samples)

    # split synthetic output
    phtp_syn = df_out[phtp.columns]
    gexp_pca_syn = df_out[pca_cols].values

    # inverse PCA + inverse scale
    gexpl1_syn = pca.inverse_transform(gexp_pca_syn)
    gexp_syn = np.exp(gexpl1_syn) - 1
    gexp_syn = np.round(np.clip(gexp_syn, 0, None)).astype(int)

    df_gexp_syn = pd.DataFrame(gexp_syn, columns=gexp.columns)
    df_syn = pd.concat(
        [phtp_syn.reset_index(drop=True), df_gexp_syn.reset_index(drop=True)], axis=1
    )

    return df_syn


def class_mvn(
    gexp: np.ndarray,          # (n_samples, n_genes) raw counts
    trgt: np.ndarray,             # (n_samples,) class labels
    n_components: int = None,  # SVD rank; defaults to 0.6 * n_samples
    noise_level: float = 0.0,  # scales additive noise relative to class std
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Functional MVN synthetic data generator with low-rank SVD approximation.

    Steps
    -----
    1. log1p transform raw counts
    2. Global centering
    3. Randomized SVD -> project all samples into reduced space
    4. Per class: fit Gaussian in reduced space, sample via
       standard normal + rotation (z ~ N(0,I), x = mu + V diag(sqrt(lam)) z)
    5. Back-project synthetic samples to gene space

    Parameters
    ----------
    gexp        : (n_samples, n_genes) raw counts matrix
    trgt           : (n_samples,) integer or string class labels
    n_components: SVD rank to retain. Defaults to min(0.6 * n_samples,
                  min_class_size - 1) to avoid rank deficiency within classes.
    noise_level : additive noise scale (0 = no noise).
    random_state: for reproducibility of randomized SVD and sampling

    Returns
    -------
    X_synth_gene : (n_samples, n_genes) synthetic data in log1p-gene space
    y_synth      : (n_samples,) class labels matching rows of X_synth_gene
    """
    rng = np.random.default_rng(random_state)
    classes, class_counts = np.unique(trgt, return_counts=True)

    # ------------------------------------------------------------------
    # 1. log1p transform
    # ------------------------------------------------------------------
    X = np.log1p(gexp.astype(np.float64))

    # ------------------------------------------------------------------
    # 2. Global centering (required for SVD to behave like PCA)
    # ------------------------------------------------------------------
    global_mean = X.mean(axis=0)          # (n_genes,)
    X_centered = X - global_mean

    # ------------------------------------------------------------------
    # 3. Randomized SVD on the full (all-class) centered matrix
    # ------------------------------------------------------------------
    n_samples, n_genes = X_centered.shape
    min_class_size = class_counts.min()

    if n_components is None:
        # cap at min_class_size - 1 to keep within-class cov full rank
        n_components = min(int(0.6 * n_samples), min_class_size - 1)

    if n_components >= min_class_size:
        print(
            f"Warning: n_components ({n_components}) >= smallest class size "
            f"({min_class_size}). Within-class covariance will be rank-deficient. "
            f"Consider reducing n_components or using covariance shrinkage."
        )

    # U: (n_samples, k), S: (k,), Vt: (k, n_genes)
    U, S, Vt = randomized_svd(
        X_centered,
        n_components=n_components,
        random_state=random_state,
    )

    # Project all samples: coords in reduced space, shape (n_samples, k)
    X_reduced = U * S   # equivalent to X_centered @ Vt.T

    # ------------------------------------------------------------------
    # 4. Per-class: fit Gaussian, sample via z -> rotation trick
    # ------------------------------------------------------------------
    synth_reduced_list = []
    synth_labels_list  = []

    for cls in classes:
        mask   = (trgt == cls)
        X_cls  = X_reduced[mask]        # (n_cls, k)
        n_cls  = X_cls.shape[0]

        mu     = X_cls.mean(axis=0)     # (k,)
        cov    = np.cov(X_cls, rowvar=False)  # (k, k)

        # Ensure symmetry + positive semi-definiteness via eigen-decomposition
        cov         = (cov + cov.T) / 2
        eigvals, eigvecs = np.linalg.eigh(cov)   # eigvecs: columns are eigenvectors
        eigvals     = np.clip(eigvals, 0, None)   # zero out numerical negatives

        std_devs    = np.sqrt(eigvals)            # (k,)  sqrt of eigenvalues

        # Standard normal samples, then rotate + scale to match class covariance:
        z = rng.standard_normal((n_cls, n_components))
        X_synth_cls = mu + (z * std_devs) @ eigvecs.T

        # Optional additive noise proportional to per-axis std
        if noise_level > 0.0:
            z_noise     = rng.standard_normal((n_cls, n_components))
            X_synth_cls = X_synth_cls + (z_noise * std_devs * noise_level) @ eigvecs.T

        synth_reduced_list.append(X_synth_cls)
        synth_labels_list.append(np.full(n_cls, cls))

    # ------------------------------------------------------------------
    # 5. Back-project to gene space: X_gene = X_reduced @ Vt + global_mean
    # ------------------------------------------------------------------
    X_synth_reduced = np.vstack(synth_reduced_list)   # (n_samples, k)
    y_synth         = np.concatenate(synth_labels_list)

    # invert log1p and clip at 0
    X_synth_gene = np.expm1(X_synth_reduced @ Vt + global_mean)
    X_synth_gene = np.rint(X_synth_gene).clip(0).astype(int)

    # Shuffle to avoid class-ordered output
    shuffle_idx     = rng.permutation(len(y_synth))
    return X_synth_gene[shuffle_idx], y_synth[shuffle_idx]


if __name__ == "__main__":

    import pandas as pd
    import os
    demo_run=False
    n_samp = 5
    datavec = ["tcga_luad", "tcga_luad", "risk_ped", "sepsis"]

    trgt_vec = [
        "clinical_egfr_mut",
        "clinical_stage_clean",
        "sample_disease",
        "Classification"
    ]
    ctg_gen_time = np.zeros((n_samp, len(datavec)))
    #datavec = ["risk_ped"]
    # (repo root)
    pwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_gan=False
    for dataset,dd in zip(datavec, range(len(datavec))):
        dfolder = os.path.join(pwd, "data", dataset)
        trgt= trgt_vec[dd]
        phtp0 = pd.read_csv(
            os.path.join(dfolder, "full_datasets", "metadata.csv"), index_col=0
        )
        ctgan_folder = os.path.join(dfolder, "pca_ctgan")
        os.makedirs(ctgan_folder, exist_ok=True)
        ## CLASS MVN GENERATION
        mvn_folder = os.path.join(dfolder, "class_mvn")
        os.makedirs(mvn_folder, exist_ok=True)

        gen_time= {}
        for ss in np.arange(1, n_samp + 1):
            tic = time.time()
            df_trn = pd.read_csv(os.path.join(dfolder, "df_trn",
                                              f"df_trn_{trgt}_{ss}.csv"), index_col=0)
            if run_gan:
                df_syn_ctg = pca_ctgan(
                    phtp=df_trn[phtp0.columns],
                    gexp=df_trn.drop(columns=phtp0.columns),
                    pca_thres=0.6,
                    ctgan_epochs=2 if demo_run else 240,
                )
                gen_time[ss,dd]=time.time() - tic
                print(f"{time.time() - tic} seconds is are takes for CTGAN")
                df_syn_ctg.to_csv(os.path.join(ctgan_folder,
                        f"df_syn_ctg_{trgt}_{str(ss)}.csv"))

            ## CLASS MVN generates
            gexp_cols= df_trn.drop(columns=phtp0.columns).columns
            gexp_syn, y_trgt1 = class_mvn(df_trn[gexp_cols].to_numpy(),
                                    trgt=df_trn[trgt].values, random_state=ss,
                                    noise_level=0.5)
            df_syn_mvn = pd.DataFrame(columns=gexp_cols, data=gexp_syn)
            df_syn_mvn[trgt] = y_trgt1
            df_syn_mvn.to_csv(os.path.join(mvn_folder,
                                           f"df_syn_mvn_{trgt}_{str(ss)}.csv"))
        if run_gan:
            np.savez(os.path.join(ctgan_folder, f"ctg_gen_time_{trgt}.npz"),
                 ctg_gen_time=ctg_gen_time)


