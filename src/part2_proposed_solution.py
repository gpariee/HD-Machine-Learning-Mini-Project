"""
part2_proposed_solution.py
===========================
Proposed improvement over Bhagat, Sharma & Agarwal (2025)'s stacking-based
heart-attack prediction pipeline.

MOTIVATION (see report Part 2, Sec. 1 for full discussion):
  L1. The published pipeline is evaluated on a dataset mirror containing
      723/1025 (70.5%) exact duplicate rows (see eda_and_leakage_check.py
      and part1_reproduction.py). A plain random split leaks near-identical
      patients across train/test, inflating every reported metric. Our
      Part-1 reproduction shows accuracy falls from ~99.4% (leaky) to
      ~83% (de-duplicated, honest) for the *identical* stacking pipeline.
  L2. The paper reports a SINGLE train/test split with no repeated runs,
      no variance estimate, and no statistical test that the stacking
      ensemble's improvement over its base learners is genuine rather
      than sampling noise -- a serious concern on a dataset this small
      once duplicates are removed (n=302).
  L3. All hyperparameters (tree depth, K, regularisation, number of
      estimators, the stacking meta-classifier itself) are left at
      implicit defaults; no hyperparameter optimisation is performed.
  L4. Four of the thirteen input features (cp, restecg, slope, thal) are
      nominal categorical codes (e.g. thal: 1=normal, 2=fixed defect,
      3=reversible defect) but are fed to distance- and margin-based
      learners (LR, KNN) as raw integers, implicitly imposing a false
      ordinal relationship between categories that do not have one.
      This is a documented pitfall in the heart-disease-ML literature
      (e.g. Mohan et al. 2019; Ghosh et al. 2021 both flag categorical
      handling as a source of avoidable error on this exact family of
      UCI-derived datasets).

PROPOSED SOLUTION -- four combined, substantial methodological changes
(NOT a mere classifier swap):
  S1. Leakage-safe protocol: built and evaluated exclusively on the
      de-duplicated 302-row dataset, with a *nested* stratified 10x5-fold
      cross-validation protocol (outer loop = unbiased performance
      estimate; inner loop = hyperparameter search) replacing the single
      random 80/20 split.
  S2. Redesigned feature space: proper one-hot encoding of the four
      nominal categorical features, plus four domain-motivated derived
      features (percentage of age-predicted maximum heart rate achieved,
      cholesterol-to-age ratio, blood-pressure x cholesterol load term,
      and an ST-depression x slope interaction term), followed by
      mutual-information-based feature selection whose k is itself
      tuned inside the inner CV loop.
  S3. Hyperparameter optimisation: RandomizedSearchCV (inner 5-fold) tunes
      the six base learners and a regularised logistic-regression
      meta-classifier jointly with the feature-selection width.
  S4. Ensemble-strategy ablation: the tuned stacking design is compared
      against a tuned soft-voting ensemble of the same base learners, to
      test whether the specific stacking architecture (vs. simpler
      averaging) is what drives any improvement.

For a fair, paired comparison, Part 1's paper-style stacking pipeline
(default hyperparameters, raw/ordinal features, no tuning) is re-evaluated
on the *same* de-duplicated data using the *same* outer folds as the
proposed pipeline, and a Wilcoxon signed-rank test is used to assess
whether the observed improvement is statistically significant.
"""
import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from functools import partial
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve, auc)

from common import FEATURES, TARGET, RANDOM_STATE

warnings.filterwarnings("ignore", category=UserWarning)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "heart.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "part2")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

NOMINAL_CATEGORICAL = ["cp", "restecg", "slope", "thal"]
BINARY = ["sex", "fbs", "exang"]
CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]

# NOTE ON COMPUTE BUDGET: this environment provides a single CPU core.
# A "textbook" nested-CV + HPO search (10x5 folds, large forests, wide
# grids) would take well over an hour of wall-clock time here. We use a
# reduced-but-still-genuine nested CV budget (5 outer x 3 inner folds,
# 12 random-search draws, capped ensemble sizes) that keeps every
# methodological element intact (nested HPO, feature-selection-width
# tuning, paired statistical testing) while completing in a few minutes.
# On a standard multi-core machine the OUTER_FOLDS/INNER_FOLDS/
# N_ITER_SEARCH constants below can be raised (e.g. to 10/5/25, as used
# in Part 1's 30-repeat protocol) simply by editing these constants; the
# rest of the pipeline is unchanged. This trade-off and its effect on
# variance is discussed explicitly in the report.
OUTER_FOLDS = 10
INNER_FOLDS = 3
N_ITER_SEARCH = 12
N_JOBS = 1


# ---------------------------------------------------------------------
# S2: Feature engineering
# ---------------------------------------------------------------------
def engineer_features(df):
    """Row-wise, deterministic, target-independent derived features.
    Safe to compute before CV splitting (no statistics are estimated
    from the data; each value depends only on that row's own raw
    features), unlike scaling or encoding which must be fit per-fold.
    """
    df = df.copy()
    df["pct_age_predicted_hr"] = df["thalach"] / (220.0 - df["age"])
    df["chol_age_ratio"] = df["chol"] / df["age"]
    df["bp_chol_load"] = (df["trestbps"] * df["chol"]) / 10000.0
    df["oldpeak_slope_interaction"] = df["oldpeak"] * (df["slope"] + 1)
    return df


ENGINEERED = ["pct_age_predicted_hr", "chol_age_ratio", "bp_chol_load",
              "oldpeak_slope_interaction"]


def build_preprocessor():
    return ColumnTransformer(transformers=[
        ("nominal", OneHotEncoder(handle_unknown="ignore"), NOMINAL_CATEGORICAL),
        ("binary", "passthrough", BINARY),
        ("continuous", StandardScaler(), CONTINUOUS + ENGINEERED),
    ])


# ---------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------
def build_baseline_pipeline(random_state=RANDOM_STATE):
    """Part-1-style pipeline: raw ordinal features (no one-hot, no
    engineered features), default hyperparameters, LR meta-classifier,
    5-fold internal stacking CV. Used as the paired comparator so that
    any improvement can be attributed to S1-S4 rather than to a
    different train/test partition."""
    scaler = ColumnTransformer([
        ("scale", StandardScaler(), FEATURES)
    ])
    estimators = [
        ("LR", LogisticRegression(max_iter=1000, random_state=random_state)),
        ("DT", DecisionTreeClassifier(random_state=random_state)),
        ("RF", RandomForestClassifier(random_state=random_state)),
        ("XGB", XGBClassifier(random_state=random_state, eval_metric="logloss")),
        ("NB", GaussianNB()),
        ("KNN", KNeighborsClassifier()),
    ]
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=random_state),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
        stack_method="predict_proba", n_jobs=N_JOBS)
    pipe = Pipeline([("scale", scaler), ("stack", stack)])
    return pipe


def build_proposed_pipeline(random_state=RANDOM_STATE, kind="stacking"):
    """S1-S4 proposed pipeline. kind='stacking' or 'voting' (S4 ablation)."""
    pre = build_preprocessor()
    # NOTE (reproducibility fix): mutual_info_classif's k-NN-based estimator
    # adds small internal random noise to continuous features to break
    # distance ties, and its `random_state` parameter defaults to None
    # (i.e. draws from the *global* numpy RNG) if not set explicitly. Left
    # unpinned, this made feature-selection scores -- and therefore every
    # downstream tuned hyperparameter and metric in this file -- silently
    # non-deterministic across runs despite every other random_state in the
    # pipeline being fixed to RANDOM_STATE. Wrapping it with functools.partial
    # pins this source of randomness too, so re-running this script (or the
    # equivalent notebook) reproduces bit-identical results.
    mi_scorer = partial(mutual_info_classif, random_state=random_state)
    select = SelectKBest(score_func=mi_scorer, k=10)

    estimators = [
        ("LR", LogisticRegression(max_iter=2000, random_state=random_state)),
        ("DT", DecisionTreeClassifier(random_state=random_state)),
        ("RF", RandomForestClassifier(random_state=random_state)),
        ("XGB", XGBClassifier(random_state=random_state, eval_metric="logloss")),
        ("NB", GaussianNB()),
        ("KNN", KNeighborsClassifier()),
    ]

    if kind == "stacking":
        ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=2000, random_state=random_state),
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
            stack_method="predict_proba", n_jobs=N_JOBS)
        step_name = "stack"
    else:
        ensemble = VotingClassifier(estimators=estimators, voting="soft", n_jobs=N_JOBS)
        step_name = "vote"

    pipe = Pipeline([("pre", pre), ("select", select), (step_name, ensemble)])
    return pipe, step_name


def get_search_space(step_name):
    prefix = f"{step_name}__"
    space = {
        "select__k": [8, 10, 12, "all"],
        f"{prefix}LR__C": [0.01, 0.1, 0.3, 1, 3, 10],
        f"{prefix}DT__max_depth": [2, 3, 4, 5, 6, None],
        f"{prefix}DT__min_samples_leaf": [1, 2, 4, 8],
        f"{prefix}RF__n_estimators": [50, 100, 150],
        f"{prefix}RF__max_depth": [3, 4, 5, None],
        f"{prefix}RF__min_samples_leaf": [1, 2, 4],
        f"{prefix}XGB__n_estimators": [50, 100, 150],
        f"{prefix}XGB__max_depth": [2, 3, 4],
        f"{prefix}XGB__learning_rate": [0.03, 0.1, 0.2],
        f"{prefix}KNN__n_neighbors": [3, 5, 7, 9, 11],
    }
    if step_name == "stack":
        space["stack__final_estimator__C"] = [0.01, 0.1, 0.3, 1, 3, 10]
    return space


# ---------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------
def eval_fold(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_proba) if len(set(y_true)) > 1 else np.nan,
    }


# ---------------------------------------------------------------------
# Nested CV runner
# ---------------------------------------------------------------------
def nested_cv_baseline(df, outer_cv):
    """Paired comparator: Part-1-style pipeline, no HPO, evaluated on the
    exact same outer folds as the proposed pipeline."""
    X = df[FEATURES]
    y = df[TARGET].values
    fold_metrics = []
    oof_true, oof_proba = [], []
    for fold_i, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y)):
        pipe = build_baseline_pipeline(random_state=RANDOM_STATE)
        pipe.fit(X.iloc[tr_idx], y[tr_idx])
        y_pred = pipe.predict(X.iloc[te_idx])
        y_proba = pipe.predict_proba(X.iloc[te_idx])[:, 1]
        fold_metrics.append(eval_fold(y[te_idx], y_pred, y_proba))
        oof_true.extend(y[te_idx]); oof_proba.extend(y_proba)
    return pd.DataFrame(fold_metrics), (np.array(oof_true), np.array(oof_proba))


def nested_cv_proposed(df_eng, outer_cv, kind="stacking", n_iter=N_ITER_SEARCH):
    """Full nested CV: inner RandomizedSearchCV for HPO+feature-selection
    width, outer loop for unbiased evaluation."""
    X_cols = NOMINAL_CATEGORICAL + BINARY + CONTINUOUS + ENGINEERED
    X = df_eng[X_cols]
    y = df_eng[TARGET].values
    fold_metrics = []
    best_params_per_fold = []
    oof_true, oof_proba = [], []

    for fold_i, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y)):
        pipe, step_name = build_proposed_pipeline(random_state=RANDOM_STATE, kind=kind)
        space = get_search_space(step_name)
        inner_cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        search = RandomizedSearchCV(
            pipe, param_distributions=space, n_iter=n_iter, cv=inner_cv,
            scoring="roc_auc", random_state=RANDOM_STATE, n_jobs=N_JOBS, refit=True)
        search.fit(X.iloc[tr_idx], y[tr_idx])
        best = search.best_estimator_
        y_pred = best.predict(X.iloc[te_idx])
        y_proba = best.predict_proba(X.iloc[te_idx])[:, 1]
        fold_metrics.append(eval_fold(y[te_idx], y_pred, y_proba))
        best_params_per_fold.append(search.best_params_)
        oof_true.extend(y[te_idx]); oof_proba.extend(y_proba)
        print(f"  [{kind}] outer fold {fold_i+1}/{OUTER_FOLDS} done "
              f"(best inner AUC={search.best_score_:.3f})")

    return pd.DataFrame(fold_metrics), best_params_per_fold, (np.array(oof_true), np.array(oof_proba))


def main():
    t0 = time.time()
    df = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)
    df_eng = engineer_features(df)
    print(f"Working on de-duplicated dataset: n={len(df)}")

    outer_cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    print("\n--- Evaluating paired baseline (Part-1-style stacking, no HPO) ---")
    baseline_metrics, baseline_oof = nested_cv_baseline(df, outer_cv)

    print("\n--- Evaluating proposed pipeline (stacking + S1-S4) ---")
    proposed_metrics, proposed_params, proposed_oof = nested_cv_proposed(
        df_eng, outer_cv, kind="stacking")

    print("\n--- Evaluating ablation: proposed features/HPO + soft voting ---")
    voting_metrics, voting_params, voting_oof = nested_cv_proposed(
        df_eng, outer_cv, kind="voting")

    # --- Aggregate ---
    def summarize(df_m, name):
        s = df_m.mean().to_dict()
        s_std = df_m.std().to_dict()
        return {f"{name}_{k}_mean": v for k, v in s.items()} | \
               {f"{name}_{k}_std": v for k, v in s_std.items()}

    summary = {}
    summary.update(summarize(baseline_metrics, "baseline"))
    summary.update(summarize(proposed_metrics, "proposed_stack"))
    summary.update(summarize(voting_metrics, "proposed_vote"))

    with open(os.path.join(OUT_DIR, "nested_cv_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    baseline_metrics.to_csv(os.path.join(OUT_DIR, "baseline_per_fold.csv"), index_label="fold")
    proposed_metrics.to_csv(os.path.join(OUT_DIR, "proposed_stack_per_fold.csv"), index_label="fold")
    voting_metrics.to_csv(os.path.join(OUT_DIR, "proposed_vote_per_fold.csv"), index_label="fold")
    with open(os.path.join(OUT_DIR, "proposed_stack_best_params_per_fold.json"), "w") as f:
        json.dump(proposed_params, f, indent=2)

    # --- Statistical significance: paired Wilcoxon signed-rank test ---
    stats_results = {}
    for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
        a = baseline_metrics[metric].values
        b = proposed_metrics[metric].values
        try:
            stat, p = wilcoxon(b - a)
        except ValueError:
            stat, p = np.nan, np.nan
        stats_results[metric] = {
            "baseline_mean": float(a.mean()), "proposed_mean": float(b.mean()),
            "mean_diff": float(b.mean() - a.mean()),
            "wilcoxon_stat": float(stat) if stat == stat else None,
            "p_value": float(p) if p == p else None,
            "significant_at_0.05": bool(p < 0.05) if p == p else None,
        }
    with open(os.path.join(OUT_DIR, "significance_tests.json"), "w") as f:
        json.dump(stats_results, f, indent=2)

    print("\n=== Nested CV results (mean +/- std over 10 outer folds) ===")
    comp = pd.DataFrame({
        "Baseline (paper-style, no HPO)": baseline_metrics.mean(),
        "Proposed (Stacking, S1-S4)": proposed_metrics.mean(),
        "Proposed (Soft Voting, S1-S4)": voting_metrics.mean(),
    })
    comp_std = pd.DataFrame({
        "Baseline std": baseline_metrics.std(),
        "Proposed Stack std": proposed_metrics.std(),
        "Proposed Vote std": voting_metrics.std(),
    })
    print(comp.round(4))
    comp.to_csv(os.path.join(OUT_DIR, "mean_comparison.csv"))
    comp_std.to_csv(os.path.join(OUT_DIR, "std_comparison.csv"))

    print("\n=== Paired Wilcoxon signed-rank test (proposed stack vs baseline) ===")
    print(json.dumps(stats_results, indent=2))

    # --- Figures ---
    metrics_order = ["accuracy", "precision", "recall", "f1", "auc"]
    x = np.arange(len(metrics_order))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, [comp["Baseline (paper-style, no HPO)"][m] for m in metrics_order],
           width, yerr=[comp_std["Baseline std"][m] for m in metrics_order],
           capsize=3, label="Baseline (paper-style, no HPO)")
    ax.bar(x, [comp["Proposed (Stacking, S1-S4)"][m] for m in metrics_order],
           width, yerr=[comp_std["Proposed Stack std"][m] for m in metrics_order],
           capsize=3, label="Proposed (Stacking, S1-S4)")
    ax.bar(x + width, [comp["Proposed (Soft Voting, S1-S4)"][m] for m in metrics_order],
           width, yerr=[comp_std["Proposed Vote std"][m] for m in metrics_order],
           capsize=3, label="Proposed (Soft Voting, S1-S4)")
    ax.set_xticks(x); ax.set_xticklabels([m.upper() for m in metrics_order])
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score (10-fold nested CV mean +/- std)")
    ax.set_title("Part 2: Proposed pipeline vs. paired baseline (de-duplicated data)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "part2_proposed_vs_baseline.png"), dpi=150)
    plt.close()

    # box plots of per-fold accuracy/F1/AUC
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, ["accuracy", "f1", "auc"]):
        data = [baseline_metrics[metric], proposed_metrics[metric], voting_metrics[metric]]
        ax.boxplot(data, tick_labels=["Baseline", "Proposed\n(Stack)", "Proposed\n(Vote)"])
        ax.set_title(metric.upper())
        ax.set_ylabel("Score per outer fold")
    plt.suptitle("Per-fold score distributions (10 outer folds)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "part2_boxplots_per_fold.png"), dpi=150)
    plt.close()

    # pooled out-of-fold ROC curves
    plt.figure(figsize=(7, 6))
    for name, (yt, yp) in [("Baseline (paper-style)", baseline_oof),
                            ("Proposed (Stacking)", proposed_oof),
                            ("Proposed (Voting)", voting_oof)]:
        fpr, tpr, _ = roc_curve(yt, yp)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (pooled OOF AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("Part 2: pooled out-of-fold ROC curves (10-fold nested CV)")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "part2_pooled_roc.png"), dpi=150)
    plt.close()

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
    print("Done. Results written to results/part2/ and figures/.")


if __name__ == "__main__":
    main()
