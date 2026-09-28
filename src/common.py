"""
common.py
=========
Shared utilities for Part 1 (paper reproduction) and Part 2 (proposed solution).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              roc_curve)

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
TARGET = "target"

RANDOM_STATE = 42


def load_data(path):
    df = pd.read_csv(path)
    return df


def get_base_models(random_state=RANDOM_STATE):
    """
    Instantiate the six base classifiers named in the paper (Sec. 3.3 / Table 8):
    Logistic Regression, Decision Tree, Random Forest, XGBoost, Naive Bayes, KNN.

    The paper does not report specific hyperparameters for any classifier
    (no learning rate, tree depth, n_estimators, K, etc. are given anywhere
    in the text, tables, or algorithm boxes). We therefore use each
    classifier's well-established scikit-learn / xgboost defaults, which is
    the standard assumption made in reproduction studies when a paper is
    silent on hyperparameters. This choice is documented and justified in
    the accompanying report. A light, justified deviation from pure
    defaults is made only where the default would be clearly unreasonable
    for this data size (e.g. XGBoost's eval_metric to silence a warning).
    """
    models = {
        "LR": LogisticRegression(max_iter=1000, random_state=random_state),
        "DT": DecisionTreeClassifier(random_state=random_state),
        "RF": RandomForestClassifier(random_state=random_state),
        "XGB": XGBClassifier(random_state=random_state, eval_metric="logloss"),
        "NB": GaussianNB(),
        "KNN": KNeighborsClassifier(),
    }
    return models


def compute_metrics(y_true, y_pred, y_proba):
    """Binary classification metrics matching the paper's Table 10/11/12."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    mcc_num = (tp * tn - fp * fn)
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = mcc_num / mcc_den if mcc_den > 0 else np.nan
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "mcc": mcc,
        "auc": roc_auc_score(y_true, y_proba) if len(set(y_true)) > 1 else np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def metrics_table(results_dict):
    """results_dict: {model_name: metrics_dict} -> tidy DataFrame."""
    rows = []
    for name, m in results_dict.items():
        row = {"model": name}
        row.update(m)
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")
