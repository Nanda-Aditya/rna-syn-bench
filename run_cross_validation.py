"""
This module performs cross-validation to find the optimal
hyperparameters for Elastic Net Logistic Regression.
Functions:
    run_CV: Main function to load datasets, process data, run cross-validation,
            and store results.

Attributes:
    dt_vec (list): List of dataset and target tuples.
    n_samp (int): Number of samples to process for each dataset.
    C_values (numpy.ndarray): Hyperparameter grid for the Elastic Net Logistic Regression.
    l1_ratio (float): Fixed l1_ratio value for Elastic Net Logistic Regression.
"""
import numpy as np
import importlib
import pandas as pd
import os
from joblib import Parallel, delayed
from bench_utils.ml_wrapper import mad
from bench_utils.ml_wrapper import _fit_one_C
# Reload modules for development
reload_mod = lambda name: importlib.reload(importlib.import_module(name))
os.environ["LOKY_MAX_CPU_COUNT"] = "5"
# Dataset configuration
dt_vec = [
        ("risk_ped", "sample_disease"),
           ("tcga_luad", "clinical_egfr_mut"),
           ("sepsis", "Classification"),
           ("tcga_luad", "clinical_stage_clean")
        ]

n_samp = 5

# Hyperparameter grid for elastic net logistic regression - only C values, fixed l1_ratio
C_values = np.logspace(-4, -1, 60)
l1_ratio = 0.05  # Fixed as requested

def run_CV():
    """
    Executes cross-validation for multiple datasets and targets utilizing predefined splits.

    This function performs cross-validation on the data-trgt pairs in provided in `dt_vec`.
    We only use a grid of C values for the Elastic Net Logistic Regression. For each C,
    mean_auc and mean_n_active (number of active genes) are calculated.
     Finally, the collected results are saved to a `.npz` file.

    :return: Saves the collected cross-validation results into an `.npz` file named `bench_cv_trn.npz`.
    :rtype: None
    """
    # Path of the script itself (repo location)
    pwd = os.path.dirname(os.path.abspath(__file__))

    # Cross-validation configuration
    cv_folds = 5
    bench_cv_trn= {}
    for dataset, trgt in dt_vec:
        print(f"Processing dataset: {dataset}, target: {trgt}")

        dfolder = os.path.join(pwd, "data", dataset)

        # Determine number of classes based on target
        if trgt in ["clinical_egfr_mut", "Classification"]:
            n_class = 2
        else:
            n_class = 3

        # Load metadata
        phtp0 = pd.read_csv(
            os.path.join(dfolder, "full_datasets", "metadata.csv"), index_col=0
        )
        phtp_cols = phtp0.columns

        # Store results for all samples
        all_results = []

        # Load all 5 splits (ss=1 through 5) for tune_on_predef_splits
        df_trn_list = []
        df_tst_list = []
        bench_cv_trn[trgt]= []
        for ss in np.arange(1, n_samp + 1):
            print(f"Loading sample {ss}/{n_samp}")

            df_trn = pd.read_csv(os.path.join(dfolder, "df_trn",
                    f"df_trn_{trgt}_{ss}.csv"), index_col=0 )
            df_tst = pd.read_csv(os.path.join(dfolder, "df_tst",
                    f"df_tst_{trgt}_{ss}.csv"), index_col=0 )
            df_trn_list.append(df_trn)
            df_tst_list.append(df_tst)

        # Determine number of features
        num_features = min(270, int(df_trn_list[0].shape[0] * 0.9))
        gexp_cols = np.array([col for col in df_trn_list[0].columns if col not in phtp_cols])

        ix_trn = np.argsort(-mad(pd.concat([np.log(df[gexp_cols] + 1)
                                            for df in df_trn_list], ignore_index=True,
                                           axis=0)))[:num_features]
        top_genes = gexp_cols[ix_trn].tolist()
        trn_list= []
        tst_list= []
        for i, (df_trn, df_tst) in enumerate(zip(df_trn_list, df_tst_list)):
            trn1 = np.log(df_trn[top_genes] + 1)
            tst1 = np.log(df_tst[top_genes] + 1)

            trn1[trgt] = df_trn[trgt].values
            tst1[trgt] = df_tst[trgt].values
            trn_list.append(trn1)
            tst_list.append(tst1)

        l1 = 0.05
        results = Parallel(n_jobs=-1)(
            delayed(_fit_one_C)(C, l1, trn_list, tst_list, trgt) for C in C_values
        )

        bench_cv_trn[trgt].append(results)
    print("done")

    np.savez("data/bench_cv_trn.npz",
             bench_cv_trn=bench_cv_trn)

if __name__ == "__main__":

    if not os.path.exists("data/bench_cv_trn.npz"):
       run_CV()

    hyp_params = {}  # set based on 0.4 or 0.5 * df_trn.num_samples for each dataset
    # binary was 0.4 and 0.5 for multiclass
    tmp= np.load("data/bench_cv_trn.npz", allow_pickle=True)
    bench_cv_trn = tmp["bench_cv_trn"].item()
    n_thres= {"Classification":48, "sample_disease": 100,
            "clinical_egfr_mut":160, "clinical_stage_clean": 160}

    for data, trgt in dt_vec:
        best = max([r for r in bench_cv_trn[trgt][0] if (r['mean_n_active'] < n_thres[trgt])
                   & (r['mean_n_active']> n_thres[trgt]*0.4)], key=lambda x: x['mean_auc'])
        best1= {'clf__C': best['C'], "clf__l1_ratio": best['l1_ratio'],
                "mean_auc":best['mean_auc'],
                "mean_n_active": best['mean_n_active']}
        hyp_params[trgt]= best1
        print(f"Best hyperparameters for {trgt} obtained from CV is {best1["clf__C"]}")

    np.savez("data/bench_cv_trn.npz",
             hyp_params=hyp_params, bench_cv_trn=bench_cv_trn) # add hyp_params

    # optional figs
    reload_mod("bench_utils.plot_utils_ml")
    from bench_utils.plot_utils_ml import fig_cv_sparsity
    for data, trgt in dt_vec:
        fig_cv_sparsity(bench_cv_trn, C_values, trgt,
                data, c_opt=hyp_params[trgt]["clf__C"])