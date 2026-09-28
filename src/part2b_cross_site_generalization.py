"""
part2b_cross_site_generalization.py
=====================================
A second, complementary Part-2 contribution: an honest reconstruction of the
"four-database merge" the paper claims (Sec. 4: "the dataset is made up of
four databases: Cleveland, Hungary, Switzerland, and Long Beach V") and a
genuine cross-hospital generalization test that the paper never performs.

WHY THIS MATTERS (see report Part 2, Sec. 2.4):
  Part 1 showed that the 1025-row file used by the paper is, on the balance
  of evidence, a duplicated copy of the single-source Cleveland dataset
  (302 unique rows) rather than an authentic four-site merge. Here we fetch
  the four ORIGINAL raw UCI site files (processed.cleveland.data,
  processed.hungarian.data, processed.switzerland.data, processed.va.data)
  and show precisely why a real merge is difficult: `ca` and `thal` -- two
  of the features the paper's own Fig. 2 ranks as most important -- are
  83-99% missing in every site except Cleveland, and `slope` is 14-65%
  missing outside Cleveland. This is presented as further, independent
  evidence for the Part-1 leakage diagnosis (a genuine merge would be
  dominated by missingness, which the "clean", fully-populated 1025-row
  Kaggle file is not).

  We then build the best genuine merge that IS honestly possible (dropping
  the three high-missingness columns, imputing the remainder) and run:
    (i)  a pooled random-split evaluation (the "easy" setting most similar
         in spirit to the paper's protocol), and
    (ii) a leave-one-site-out (LOSO) cross-validation, training on three
         hospitals and testing on the held-out fourth -- a genuine
         cross-population generalization test that neither the original
         paper nor our Part-1/Part-2a work attempts.

This quantifies the gap between "same-source" and "cross-hospital"
performance, which is directly relevant to the paper's claimed clinical
motivation (a model meant for real-world deployment must generalize
beyond the hospital it was trained in).
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve, auc)

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw_sites")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "part2")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

RANDOM_STATE = 42
COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "num"]
SITES = {"cleveland": "cleveland.data", "hungarian": "hungarian.data",
          "switzerland": "switzerland.data", "va": "va.data"}

# Features retained for the honest multi-site merge (drop slope/ca/thal:
# see missingness_report.csv -- these are 42-99% missing outside Cleveland,
# so imputing them for 3 of 4 sites would mean inventing most of the
# clinical signal rather than genuinely testing generalization).
NOMINAL_CATEGORICAL = ["cp", "restecg"]
BINARY = ["sex", "fbs", "exang"]
CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
USE_COLS = NOMINAL_CATEGORICAL + BINARY + CONTINUOUS


def load_sites():
    dfs = {}
    for name, fn in SITES.items():
        df = pd.read_csv(os.path.join(RAW_DIR, fn), header=None,
                          names=COLS, na_values="?")
        df["site"] = name
        dfs[name] = df
    return dfs


def missingness_report(dfs):
    miss = pd.DataFrame({name: df[COLS[:-1]].isna().mean().round(3)
                          for name, df in dfs.items()})
    miss.to_csv(os.path.join(OUT_DIR, "cross_site_missingness_report.csv"))
    print("=== Feature missingness by site (fraction of rows) ===")
    print(miss)
    return miss


def build_merged_df(dfs):
    all_df = pd.concat(dfs.values(), ignore_index=True)
    all_df["target"] = (all_df["num"] > 0).astype(int)
    # drop rows missing any of the *retained* features' target-critical info
    # (fbs/restecg/exang/trestbps/chol/oldpeak have <=28% missingness; we
    # impute these rather than drop, to retain sample size - see pipeline)
    return all_df


def get_base_models(random_state=RANDOM_STATE):
    return {
        "LR": LogisticRegression(max_iter=1000, random_state=random_state),
        "DT": DecisionTreeClassifier(random_state=random_state),
        "RF": RandomForestClassifier(random_state=random_state, n_estimators=200),
        "XGB": XGBClassifier(random_state=random_state, eval_metric="logloss", n_estimators=150),
        "NB": GaussianNB(),
        "KNN": KNeighborsClassifier(),
    }


def build_pipeline(random_state=RANDOM_STATE):
    pre = ColumnTransformer([
        ("nominal", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), NOMINAL_CATEGORICAL),
        ("binary", SimpleImputer(strategy="most_frequent"), BINARY),
        ("continuous", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), CONTINUOUS),
    ])
    estimators = [(name, clone) for name, clone in get_base_models(random_state).items()]
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=random_state),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
        stack_method="predict_proba", n_jobs=1)
    return Pipeline([("pre", pre), ("stack", stack)])


def eval_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_proba) if len(set(y_true)) > 1 else np.nan,
        "n_test": len(y_true),
    }


def pooled_random_cv(df, n_splits=10):
    X = df[USE_COLS]
    y = df["target"].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = []
    oof_true, oof_proba = [], []
    for tr_idx, te_idx in skf.split(X, y):
        pipe = build_pipeline()
        pipe.fit(X.iloc[tr_idx], y[tr_idx])
        y_pred = pipe.predict(X.iloc[te_idx])
        y_proba = pipe.predict_proba(X.iloc[te_idx])[:, 1]
        fold_metrics.append(eval_metrics(y[te_idx], y_pred, y_proba))
        oof_true.extend(y[te_idx]); oof_proba.extend(y_proba)
    return pd.DataFrame(fold_metrics), (np.array(oof_true), np.array(oof_proba))


def leave_one_site_out(df):
    results = {}
    oof = {}
    for test_site in SITES:
        train_df = df[df["site"] != test_site]
        test_df = df[df["site"] == test_site]
        if test_df["target"].nunique() < 2:
            print(f"Skipping {test_site}: only one class present in test site.")
            continue
        X_train, y_train = train_df[USE_COLS], train_df["target"].values
        X_test, y_test = test_df[USE_COLS], test_df["target"].values
        pipe = build_pipeline()
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        results[test_site] = eval_metrics(y_test, y_pred, y_proba)
        oof[test_site] = (y_test, y_proba)
        print(f"LOSO test site={test_site}: n={len(y_test)}, "
              f"acc={results[test_site]['accuracy']:.3f}, "
              f"auc={results[test_site]['auc']:.3f}")
    return pd.DataFrame(results).T, oof


def main():
    dfs = load_sites()
    miss = missingness_report(dfs)
    merged = build_merged_df(dfs)
    print(f"\nMerged multi-site dataset (honest, all 4 sites): n={len(merged)}")
    print(merged["site"].value_counts())
    print(merged["target"].value_counts())
    merged.to_csv(os.path.join(OUT_DIR, "..", "..", "data", "heart_4site_merged_raw.csv"), index=False)

    print("\n--- Protocol (i): pooled random 10-fold CV (easy, same-distribution) ---")
    pooled_metrics, pooled_oof = pooled_random_cv(merged)
    print(pooled_metrics.mean().round(4))

    print("\n--- Protocol (ii): leave-one-site-out CV (hard, cross-hospital) ---")
    loso_metrics, loso_oof = leave_one_site_out(merged)
    print(loso_metrics.round(4))

    pooled_metrics.to_csv(os.path.join(OUT_DIR, "pooled_random_cv_per_fold.csv"))
    loso_metrics.to_csv(os.path.join(OUT_DIR, "leave_one_site_out_per_site.csv"))

    summary = {
        "pooled_random_cv_mean": pooled_metrics.mean().to_dict(),
        "pooled_random_cv_std": pooled_metrics.std().to_dict(),
        "leave_one_site_out_mean": loso_metrics.mean().to_dict(),
        "leave_one_site_out_per_site": loso_metrics.to_dict(orient="index"),
    }
    with open(os.path.join(OUT_DIR, "cross_site_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    # --- Figure: pooled (easy) vs LOSO (hard) comparison ---
    metrics_order = ["accuracy", "precision", "recall", "f1", "auc"]
    x = np.arange(len(metrics_order))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, [pooled_metrics.mean()[m] for m in metrics_order], width,
           yerr=[pooled_metrics.std()[m] for m in metrics_order], capsize=3,
           label="Pooled random 10-fold CV (same-distribution)")
    ax.bar(x + width/2, [loso_metrics.mean()[m] for m in metrics_order], width,
           yerr=[loso_metrics.std()[m] for m in metrics_order], capsize=3,
           label="Leave-one-site-out CV (cross-hospital)")
    ax.set_xticks(x); ax.set_xticklabels([m.upper() for m in metrics_order])
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title("Honest 4-site merge: same-distribution vs. cross-hospital generalization")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "part2b_cross_site_comparison.png"), dpi=150)
    plt.close()

    # --- Figure: missingness heatmap ---
    plt.figure(figsize=(7, 5))
    plt.imshow(miss.values, cmap="Reds", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(label="Fraction missing")
    plt.xticks(range(len(miss.columns)), miss.columns, rotation=0)
    plt.yticks(range(len(miss.index)), miss.index)
    for i in range(miss.shape[0]):
        for j in range(miss.shape[1]):
            plt.text(j, i, f"{miss.values[i,j]:.2f}", ha="center", va="center",
                      color="white" if miss.values[i, j] > 0.5 else "black", fontsize=8)
    plt.title("Feature missingness across the 4 genuine UCI sites")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "part2b_missingness_heatmap.png"), dpi=150)
    plt.close()

    # --- Figure: per-site LOSO bar chart ---
    plt.figure(figsize=(8, 5))
    loso_metrics[["accuracy", "auc"]].plot(kind="bar", ax=plt.gca())
    plt.title("Leave-one-site-out performance by held-out hospital")
    plt.ylabel("Score"); plt.ylim(0, 1.05)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "part2b_loso_per_site.png"), dpi=150)
    plt.close()

    print("\nDone. Results written to results/part2/ and figures/.")


if __name__ == "__main__":
    main()
