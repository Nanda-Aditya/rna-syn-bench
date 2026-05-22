import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import re
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from bench_utils import config

def preprocess_targets(ytrn, ytst):
    # map ytrn and ytst to 0 and 1

    ytst = ytst.astype("float32") if ytst.dtypes.iloc[0] == "bool" else ytst
    ytrn = ytrn.astype("float32") if ytrn.dtypes.iloc[0] == "bool" else ytrn

    unique_values = ytrn[ytrn.columns[0]].unique()
    num_class = len(unique_values)
    dict1 = dict(zip(unique_values, np.float32(range(num_class))))

    ytrn1 = ytrn.loc[:, ytrn.columns[0]].map(dict1)
    ytst1 = ytst.loc[:, ytst.columns[0]].map(dict1)
    return ytrn1, ytst1, num_class


def classifier_metrics_en(ytst1, y_score, num_class, prdcls):

    # Computations of metrics starts
    cls_freq = [np.count_nonzero(ytst1 == x) for x in prdcls]  # frequencies of classes
    pos_ind = np.argmin(cls_freq)  # index of minority class
    pos_class = prdcls[pos_ind]  # find minority class
    auroc = np.zeros(num_class)  # metric 1
    prauc = np.zeros(num_class)  # metric 2
    # compute roc_auc
    for kk in range(num_class):
        cls_type = type(prdcls[kk]).__name__
        fpr, tpr, _ = roc_curve(
            ytst1.astype(cls_type), y_score[:, kk].flatten(), pos_label=prdcls[kk]
        )
        auroc[kk] = auc(fpr, tpr)
    # computes pr_auc (important for imbalanced datasets)
    for kk in range(num_class):
        # precision, recall, thresholds = precision_recall_curve(ytst, y_score[:,kk].flatten(), pos_label= prdcls[kk])
        cls_type = type(prdcls[kk]).__name__
        prc, rec, _ = precision_recall_curve(
            ytst1.astype(cls_type), y_score[:, kk].flatten(), pos_label=prdcls[kk]
        )
        dx = np.concatenate(([True], np.diff(prc) > 0))
        prauc[kk] = trapz(prc[dx], rec[dx])
    return auroc, prauc


def _classifier_wrapper_en(xtrn, xtst, ytrn, ytst, hyp_params):
    """Logistic regression classifier wrapper."""

    #print(f" fitting logistic model with xtrn shape: {xtrn.shape} and xtst shape: {xtst.shape}")
    xtst = rename_columns(xtst)
    xtrn = rename_columns(xtrn)
    preproc, num_cols, cat_cols, drop_cols = _build_preprocessor(xtrn)

    clf = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=hyp_params["clf__l1_ratio"],
        C=hyp_params["clf__C"],
        class_weight="balanced",
        max_iter=5 if config.demo_run else 6000,
        n_jobs=-6,
    )
    pipe = Pipeline([("prep", preproc), ("clf", clf)])
    pipe.fit(xtrn, ytrn.values.ravel())

    # Predictions/metrics
    y_proba = pipe.predict_proba(xtst)
    prdcls = pipe.named_steps["clf"].classes_
    auroc, prauc = classifier_metrics_en(ytst, y_proba, len(prdcls), prdcls)

    # Coefficients (align shapes; one row per class, first col = intercept)
    clf = pipe.named_steps["clf"]
    intercept = np.atleast_2d(clf.intercept_).reshape(-1, 1)  # (K,1)
    coef = np.atleast_2d(clf.coef_)  # (K,p')
    coeffs = np.hstack([intercept, coef])  # (K, 1+p')

    # SHAP: explain the *transformed* representation with matching names
    Xb_trn = pipe.named_steps["prep"].transform(xtrn)
    Xb_tst = pipe.named_steps["prep"].transform(xtst)

    # Optional: make dense if your OHE is sparse and your plots expect dense
    if hasattr(Xb_trn, "toarray"):
        Xb_trn = Xb_trn.toarray()
        Xb_tst = Xb_tst.toarray()

    feat_names = getattr(
        pipe.named_steps["prep"], "get_feature_names_out", lambda: None
    )()
    expl = shap.LinearExplainer(clf, Xb_trn, feature_names=feat_names)
    shap_vals = expl.shap_values(
        Xb_tst
    )  # for logistic: values are on log-odds by default
    # per class recall
    y_pred = prdcls[np.argmax(y_proba, axis=1)]
    from sklearn.metrics import recall_score

    per_class_recall = np.array(
        recall_score(ytst, y_pred, labels=prdcls, average=None)
    )  # (3,) vector

    return auroc, prauc, prdcls, shap_vals, coeffs, per_class_recall

def rename_columns(df):
    df = df.rename(columns=lambda x: re.sub(r"[^\w]+", "_", str(x)).strip())
    return df

def split_karte_the(df_trn, df_syn, df_tst, trgt):

    trgt_not = df_trn.columns[df_trn.columns != trgt]
    xtst = df_tst[trgt_not]  # x is predictors
    ytst = df_tst[[trgt]]  # y is predicted
    xtrn_syn = df_syn[trgt_not]
    ytrn_syn = df_syn[[trgt]]
    xtrn_real = df_trn[trgt_not]
    ytrn_real = df_trn[[trgt]]
    return xtrn_syn, ytrn_syn, xtrn_real, ytrn_real, xtst, ytst


def _build_preprocessor(xdf: pd.DataFrame):

    cols = list(xdf.columns)
    # Identify categorical-like columns
    cat_cols = [
        c
        for c in xdf.columns
        if pd.api.types.is_object_dtype(xdf[c])
        or isinstance(xdf[c].dtype, pd.CategoricalDtype)
        or pd.api.types.is_string_dtype(xdf[c])
    ]

    # Identify numeric-like columns (including bools and datetimes)
    num_cols = [
        c
        for c in xdf.columns
        if pd.api.types.is_numeric_dtype(xdf[c])
        or pd.api.types.is_bool_dtype(xdf[c])
        or pd.api.types.is_datetime64_any_dtype(xdf[c])
    ]
    # Drop identifiers (string) from modeling
    drop_cols = (xdf.nunique() > 6000) & (xdf.columns.isin(cat_cols))
    # ColumnTransformer handles selection; dropped columns are ignored via remainder='drop'

    num_transformer = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=True, with_std=True)),
        ]
    )

    cat_transformer = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
        ],
        remainder="drop",
    )

    return preprocessor, num_cols, cat_cols, drop_cols


def trapz(x, y):
    """
    Perform trapezoidal integration of y=f(x) over x.
    x (array-like): The x-coordinates of the points
    y (array-like): The y-coordinates of the points (y = f(x))
    returns AUC
    """
    # Ensure x and y are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    # Check if x and y have the same shape
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    # Calculate the spacings between x values
    dx = np.diff(x)
    # Calculate the average of adjacent y values
    y_avg = (y[:-1] + y[1:]) / 2
    # Calculate the area using the trapezoidal rule
    area = sum(dx * y_avg)
    return area
