#!/usr/bin/env python3
"""
Privacy metrics pipeline for computing expression-level privacy metrics.

Iterates over the same trgt-data pairs as make_de_ml and compute_benchmarks,
runs compute_expression_privacy_metrics for each synthetic data type, and appends
the new privacy results to existing .npz files.

All existing variables in the npz file are retained; only privacy metrics are added.
"""

import numpy as np
import pandas as pd
import os
import sys
os.environ["LOKY_MAX_CPU_COUNT"]=8
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench_utils.make_obj_array import make_obj_array

# Path of the script (repo location)
pwd = os.path.dirname(os.path.abspath(__file__))

datavec = [
    "risk_ped",
    "tcga_luad",
    "sepsis",
    "tcga_luad",
]

n_samp = 5

trgt_vec = [
    "sample_disease",
    "clinical_egfr_mut",
    "Classification",
"clinical_stage_clean",
]

for dataset, trgt in zip(datavec, trgt_vec):
    print(f"\nProcessing {dataset} - {trgt}...")

    dfolder = os.path.join(pwd, "data", dataset)

    # Check if results file exists
    results_file = os.path.join(dfolder, f"bench_results_dcr_{trgt}.npz")
    # if not os.path.exists(results_file):
    #     print(f"  Warning: {results_file} not found, skipping...")
    #     continue

    # Expression-level privacy metrics for each synthetic data type
    expr_dists_dbt = []
    expr_dists_ctg = []
    expr_dists_mvn = []


    # Load metadata for phenotype column names
    phtp0 = pd.read_csv(
        os.path.join(dfolder, "full_datasets", "metadata.csv"), index_col=0
    )
    phtp_cols = phtp0.columns

    for ss in np.arange(1, n_samp + 1):
        print(f"  Processing sample {ss}/{n_samp}...")
        # Load training data
        df_trn = pd.read_csv(
            os.path.join(dfolder, "df_trn", f"df_trn_{trgt}_{ss}.csv"), index_col=0)
        df_tst= pd.read_csv(os.path.join(dfolder, "df_tst", f"df_tst_{trgt}_{ss}.csv"),
                            index_col=0)

        # Get gene expression columns
        gexp_cols = df_trn.drop(columns=phtp_cols).columns
        # Load synthetic data conditionally based on flags
        df_syn_dbt = pd.read_csv(
            os.path.join(dfolder, "dbTwin", f"df_syn_{trgt}_{ss}.csv"), index_col=0)

        df_syn_ctg = pd.read_csv(
            os.path.join(dfolder, "pca_ctgan", f"df_syn_ctg_{trgt}_{ss}.csv"),
            index_col=0,)
        df_syn_mvn = pd.read_csv(
            os.path.join(dfolder, "class_mvn", f"df_syn_mvn_{trgt}_{ss}.csv"),
            index_col=0, )

        ## privacy:
        from bench_utils.privacy_metrics import compute_expression_privacy_metrics

        expr_dists_ss_dbt = compute_expression_privacy_metrics(
            df_trn=df_trn, df_syn=df_syn_dbt, df_tst=df_tst,gexp_cols=gexp_cols)
        expr_dists_dbt.append(expr_dists_ss_dbt)

        expr_dists_ss_ctg = compute_expression_privacy_metrics(
            df_trn=df_trn, df_syn=df_syn_ctg, df_tst=df_tst,gexp_cols=gexp_cols)
        expr_dists_ctg.append(expr_dists_ss_ctg)


        expr_dists_ss_mvn = compute_expression_privacy_metrics(
            df_trn=df_trn, df_syn=df_syn_mvn, df_tst=df_tst,gexp_cols=gexp_cols)
        expr_dists_mvn.append(expr_dists_ss_mvn)


    np.savez(results_file,
             expr_dists_dbt=make_obj_array(expr_dists_dbt),
             expr_dists_mvn=make_obj_array(expr_dists_mvn),
             expr_dists_ctg=make_obj_array(expr_dists_ctg))

    print(f"  Done with {dataset} - {trgt}")

print("\nAll privacy metrics computed successfully!")
