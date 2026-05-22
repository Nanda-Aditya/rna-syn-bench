"""
This script performs benchmarking of synthetic data using TSTR machine learning models.
See ml_compute_CV_params.py for hyperparameter tuning.

For each dataset and target combination, the script loads the real and synthetic datasets,
calls tune_and_train() to obtain TSTR results, and saves the results.

"""

import time
import numpy as np
import importlib
from bench_utils.ml_wrapper import tune_and_train
reload_mod = lambda name: importlib.reload(importlib.import_module(name))
import pandas as pd
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Path of the script itself (repo location)
pwd = os.path.dirname(os.path.abspath(__file__))
dt_vec = [
        ("risk_ped", "sample_disease"),
           ("tcga_luad", "clinical_egfr_mut"),
           ("sepsis", "Classification"),
           ("tcga_luad", "clinical_stage_clean")
        ]
n_samp = 5
demo_run = 0

tmp_cv= np.load("data/bench_cv_trn.npz", allow_pickle=True)
hyp_params = tmp_cv["hyp_params"].item()

for dataset, trgt in dt_vec:
    dfolder = os.path.join(pwd, "data", dataset)

    # Determine number of classes based on target
    if trgt in ["clinical_egfr_mut", "Classification"]:
        n_class = 2
    else:
        n_class = 3

    phtp0 = pd.read_csv(
        os.path.join(dfolder, "full_datasets", "metadata.csv"), index_col=0
    )
    phtp_cols = phtp0.columns
    ml_real = []
    ml_dbtwin = []
    ml_mvn = []
    ml_ctg = []
    gexp_cols_trn=[]

    for ss in np.arange(1, n_samp + 1):
        tic = time.time()

        df_trn = pd.read_csv(
            os.path.join(dfolder, "df_trn", f"df_trn_{trgt}_{ss}.csv"), index_col=0
        )
        df_tst = pd.read_csv(
            os.path.join(dfolder, "df_tst", f"df_tst_{trgt}_{ss}.csv"), index_col=0
        )

        phtp_cols = phtp0.columns
        gexp_cols = df_trn.drop(columns=phtp0.columns).columns

        df_syn_dbt = pd.read_csv(os.path.join(dfolder, "dbTwin",
                    f"df_syn_{trgt}_{ss}.csv"), index_col=0)
        df_syn_ctg = pd.read_csv(
            os.path.join(dfolder, "pca_ctgan", f"df_syn_ctg_{trgt}_{ss}.csv"),
            index_col=0)
        df_syn_mvn = pd.read_csv(
            os.path.join(dfolder, "class_mvn", f"df_syn_mvn_{trgt}_{ss}.csv"),
            index_col=0)

        num_features = min(270, int(df_trn.shape[0] * 0.9))
        tmp_real,tmp_dbtwin, tmp_mvn, tmp_ctg, tmp_cols= tune_and_train(
            df_trn,
            df_tst,
            df_syn_dbt, df_syn_mvn, df_syn_ctg,
            trgt,
            gexp_cols=gexp_cols,num_f=num_features,
            hyp_params=hyp_params[trgt])
        ml_real.append(tmp_real)
        ml_dbtwin.append(tmp_dbtwin)
        ml_mvn.append(tmp_mvn)
        ml_ctg.append(tmp_ctg)
        gexp_cols_trn.append(tmp_cols)
        print(
            f"d ML for data: {trgt} and sample: {ss} took {time.time() - tic} seconds"
        )

    data_file = os.path.join(dfolder, f"bench_results_ml_{trgt}.npz")
    if os.path.exists(data_file):
        os.remove(data_file)

    np.savez(
        data_file,
        ml_real=ml_real,
        ml_dbtwin=ml_dbtwin,
        ml_mvn=ml_mvn,
        ml_ctg=ml_ctg,
        hyp_params=hyp_params[trgt],
        gexp_cols_trn=np.array(gexp_cols_trn, dtype=object),
    )
