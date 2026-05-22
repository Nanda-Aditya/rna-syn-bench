import time
import numpy as np
import pandas as pd
from sklearn.model_selection import PredefinedSplit
from bench_utils.ml_logreg_utils import (
    _classifier_wrapper_en, preprocess_targets, split_karte_the)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
# column wise mad
mad = lambda df: np.median(
        np.absolute(df.values - np.median(df.values, axis=0)), axis=0 )

def column_fixer(df, gexp_cols):
    df.columns = df.columns.map(str)
    df[gexp_cols]=df[gexp_cols].astype(int)
    return df


def tune_and_train(
    df_trn,
    df_tst,
    df_syn_dbt, df_syn_mvn, df_syn_ctg,
    trgt,
    gexp_cols,
    num_f, hyp_params=None):

    """
    Tunes and trains models on the provided datasets. The function performs feature
    selection based on Mean Absolute Deviation (MAD), logarithmic transformation of
    expression data, target assignment for classification, and data preparation for training
    and testing. Models are trained using real data as well as synthetic datasets
    generated dbTwin, class_mvn, and PCA-CTGAN.

    :return: A tuple containing the TSTR results for each model:
             - (ml_real) ML outputs-  trained/tested on real training data/real test data..
             - (ml_dbtwin) trained on synthetic dbTwin data/tested on real test data.
             - (ml_mvn) trained on class-MVN/tested on real test data.
             - (ml_ctg) trained on synthetic PCA-CTGan data/tested on real test data.
             - gexp_cols : list of selected features for this fold
    """

    ## use MAD for filtering
    ix_trn = np.argsort(-mad(df_trn[gexp_cols]))[0:num_f]

    trn1 = np.log1p(df_trn.loc[:, gexp_cols[ix_trn]])
    syn_dbt = np.log1p(df_syn_dbt.loc[:, gexp_cols[ix_trn]])
    syn_mvn = np.log1p(df_syn_mvn.loc[:, gexp_cols[ix_trn]])
    syn_ctg = np.log1p(df_syn_ctg.loc[:, gexp_cols[ix_trn]])
    tst1 = np.log1p(df_tst.loc[:, gexp_cols[ix_trn]])

    #assign targets for classification
    trn1[trgt] = df_trn[trgt]
    syn_dbt[trgt] = df_syn_dbt[trgt]
    syn_mvn[trgt] = df_syn_mvn[trgt]
    syn_ctg[trgt] = df_syn_ctg[trgt]
    tst1[trgt] = df_tst[trgt]

    # Split dataframes into X and y
    xtrn_syn_dbt, ytrn_syn_dbt, xtrn_real, ytrn_real, xtst, ytst = split_karte_the(
        trn1, syn_dbt, tst1, trgt)

    xtrn_syn_mvn, ytrn_syn_mvn, _,_,_,_ = split_karte_the(
            trn1, syn_mvn, tst1, trgt)

    xtrn_syn_ctg, ytrn_syn_ctg, _, _, _, _ = split_karte_the(
        trn1, syn_ctg, tst1, trgt)
    num_class= len(ytrn_real[ytrn_real.columns[0]].unique())

    # map y vector to numeric values for training
    mapping = dict(zip(ytrn_real[ytrn_real.columns[0]].unique(), range(num_class)))
    inv_mapping = {v: k for k, v in mapping.items()}  #
    # run classifier on real data
    ml_real= wrap_classfier(xtrn_real, ytrn_real, xtst, ytst, mapping, inv_mapping,hyp_params )

    ml_dbtwin = wrap_classfier(xtrn_syn_dbt, ytrn_syn_dbt,xtst,ytst, mapping, inv_mapping,hyp_params )

    ml_mvn= wrap_classfier(xtrn_syn_mvn, ytrn_syn_mvn, xtst,ytst,mapping, inv_mapping,hyp_params)

    ml_ctg= wrap_classfier(xtrn_syn_ctg, ytrn_syn_ctg, xtst, ytst,mapping, inv_mapping,hyp_params)

    return ml_real, ml_dbtwin, ml_mvn, ml_ctg, gexp_cols[ix_trn]

def wrap_classfier(xtrn, ytrn,xtst,ytst, mapping, inv_mapping,hyp_params ):
    """
    Wraps a classifier while aligning target values to integer mappings, ensuring compatibility
    with the classifier and mapping predicted indices back to original class labels.

    :return: A dictionary containing evaluation metrics and outputs of the classifier, which includes:
        - auc: Area Under the Curve (AUC) score
        - prauc: Precision-Recall AUC score
        - classes_: Predicted class indices
        - orig_classes_: Original class labels corresponding to predicted indices
        - shap: SHAP values for model interpretability
        - coeff: Coefficients of the logistic regression model
        - pcr: Per-class Recall
    :rtype: dict
    """

    # align ytrm amd ytst to same integers
    ytrn_m = ytrn.loc[:, ytrn.columns[0]].map(mapping)
    ytst_m = ytst.loc[:, ytst.columns[0]].map(mapping)

    if np.any(np.isnan(ytst_m)) or np.any(np.isnan(ytrn_m)):
        raise ValueError("Unmapped class labels found in ytrn or ytst") # stratified k-folds will prevent this

    auc, prauc, prdcls, shap, coeff, pcr = _classifier_wrapper_en(
        xtrn,
        xtst,
        ytrn_m.to_frame(),
        ytst_m,
        hyp_params=hyp_params)
    # Map predicted class indices back to original class labels
    prdcls_orig = np.array([inv_mapping[int(i)] for i in prdcls])

    return {"auc": auc, "prauc": prauc, "classes_": prdcls, "orig_classes_": prdcls_orig,
            "shap": shap, "coeff": coeff, "pcr": pcr}

def _fit_one_C(C, l1, trn_list, tst_list, trgt):
    """
    Fits a logistic regression model with elastic net penalty for a given combination of
    penalty strength (C) and L1 ratio (l1). The function iteratively computes performance
    metrics (e.g., mean AUC and number of active features) across multiple training and
    testing folds provided as input.

    This function is used by ml_compute_CV_params.py to do hyperparameter optimization.

    :param C: Regularization strength for the logistic regression model. Must be a positive number.
    :param l1: L1 regularization ratio. Determines the balance between L1 and L2 penalties.
    :param trn_list: A list of training datasets, where each element represents a DataFrame
        containing the training data for a specific fold.
    :param tst_list: A list of testing datasets, where each element represents a DataFrame
        containing the testing data for a specific fold.
    :param trgt: The column name representing the target variable in the data. It is used
        to separate features from the label.

    :return: A dictionary containing the following metrics:
        - 'C': Input value of the C regularization parameter.
        - 'l1_ratio': Input value of the L1 regularization ratio.
        - 'mean_auc': Mean AUC computed across all folds.
        - 'std_auc': Standard deviation of the AUC across folds.
        - 'mean_n_active': Mean number of active features across folds.
        - 'std_n_active': Standard deviation of the number of active features across folds.
    :rtype: dict
    """
    fold_aucs = []
    fold_n_active = []
    for trn, tst in zip(trn_list, tst_list):
        clf = LogisticRegression(penalty='elasticnet', C=C, l1_ratio=l1,
                                 solver='saga', max_iter=5000, random_state=42)
        clf.fit(trn.drop(columns=trgt).values, trn[trgt].values)
        if trgt in ["clinical_egfr_mut", "Classification"]:
            auc = roc_auc_score(tst[trgt].values,
                                clf.predict_proba(tst.drop(columns=trgt).values)[:, 1])
            n_active = np.sum(np.abs(clf.coef_) > 1e-6)
        else:
            auc = roc_auc_score(tst[trgt].values,
                                clf.predict_proba(tst.drop(columns=trgt).values),
                                multi_class='ovr', average='macro')
            n_active = np.sum(np.any(np.abs(clf.coef_) > 1e-6, axis=0))
        fold_aucs.append(auc)
        fold_n_active.append(n_active)
    return {
        'C': C, 'l1_ratio': l1,
        'mean_auc': np.mean(fold_aucs), 'std_auc': np.std(fold_aucs),
        'mean_n_active': np.mean(fold_n_active), 'std_n_active': np.std(fold_n_active)
    }
