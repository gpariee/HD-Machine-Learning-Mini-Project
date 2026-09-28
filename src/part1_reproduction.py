"""
part1_reproduction.py
======================
Faithful reproduction of Bhagat, Sharma & Agarwal (2025), "An efficient
stacking-based ensemble technique for early heart attack prediction",
Multimedia Tools and Applications, 84:36351-36375.

Methodology reproduced (per Secs. 3-5 of the paper):
  1. Load the merged heart-disease dataset (1025 rows x 14 columns).
  2. Preprocess: no missing values reported/found; scale numeric features.
  3. Train six base classifiers: LR, DT, RF, XGBoost, NB, KNN.
  4. Build a 5-fold stacking ensemble (meta-classifier trained on the
     out-of-fold predictions of the six base classifiers).
  5. Evaluate with accuracy, precision, recall, F1, specificity, MCC, AUC.

ASSUMPTIONS made where the paper is silent (documented in the report,
Section 3):
  A1. Train/test split ratio: the paper states data is "divided into
      training and testing sets" but never gives a ratio. We assume an
      80/20 stratified split (the field-standard choice, and the same
      ratio explicitly used in the companion Option-1 ECG paper on this
      assignment's reading list).
  A2. Random seed: not reported. We fix random_state=42 throughout and
      additionally report mean +/- std over 30 repeated stratified splits
      to characterise the paper's likely sampling variance.
  A3. Base-classifier hyperparameters: not reported for any of the six
      models. We use scikit-learn / xgboost defaults (see common.py).
  A4. Stacking meta-classifier: not named in the paper. We use Logistic
      Regression, the standard default final estimator for classification
      stacks (and the sklearn.ensemble.StackingClassifier default choice
      pattern), with 5-fold internal cross-validation to generate
      out-of-fold base-model predictions, exactly as described textually
      in the paper's Sec. 3.4.1 and Table 8.
  A5. Feature scaling: the paper does not detail a scaling step beyond
      "preprocess ... and scale it correctly" (Sec. 3.1). We apply
      StandardScaler fit on the training fold only, inside the CV loop,
      to avoid leaking test-set statistics into training.

TWO EXPERIMENTAL PROTOCOLS are run and reported side by side:
  - Protocol A ("as-published"): uses the 1025-row file exactly as the
    paper describes it, with an ordinary random stratified split. This is
    our best-effort faithful reproduction of what the authors did.
  - Protocol B ("de-duplicated / leakage-controlled"): first removes the
    723 exact duplicate rows identified in eda_and_leakage_check.py
    (leaving the 302 unique patient records, i.e. essentially the
    original UCI Cleveland set), then repeats the identical pipeline.
    This protocol is NOT what the paper did; it is included here because
    Part 1 of the assignment requires us to critically analyse
    discrepancies between our reproduction and the published results, and
    a plain, faithful reproduction under Protocol A alone cannot show
    *why* any such discrepancy exists. Comparing A vs. B isolates the
    duplicate-row leakage as the explanatory variable.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone

from common import (load_data, get_base_models, compute_metrics,
                     metrics_table, FEATURES, TARGET, RANDOM_STATE)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "heart.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "part1")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

N_REPEATS = 30  # repeated random splits to characterise sampling variance


def build_stack(random_state=RANDOM_STATE):
    base_models = get_base_models(random_state=random_state)
    estimators = [(name, clone(model)) for name, model in base_models.items()]
    meta = LogisticRegression(max_iter=1000, random_state=random_state)
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=meta,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,
    )
    return stack


def run_single_split(df, seed, protocol_name):
    X = df[FEATURES].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}
    base_models = get_base_models(random_state=seed)
    for name, model in base_models.items():
        m = clone(model)
        m.fit(X_train_s, y_train)
        y_pred = m.predict(X_test_s)
        y_proba = m.predict_proba(X_test_s)[:, 1]
        results[name] = compute_metrics(y_test, y_pred, y_proba)

    stack = build_stack(random_state=seed)
    stack.fit(X_train_s, y_train)
    y_pred = stack.predict(X_test_s)
    y_proba = stack.predict_proba(X_test_s)[:, 1]
    results["Stacking (proposed)"] = compute_metrics(y_test, y_pred, y_proba)

    return results, (y_test, stack.predict_proba(X_test_s)[:, 1]), (X_test_s, y_test, stack, base_models, scaler)


def run_protocol(df, protocol_name):
    """Run N_REPEATS random splits; return mean/std metrics + one representative
    split's fitted models/predictions for confusion matrices & ROC curves."""
    all_results = {name: [] for name in list(get_base_models().keys()) + ["Stacking (proposed)"]}
    rep_roc_data = None
    rep_fitted = None

    for i in range(N_REPEATS):
        seed = RANDOM_STATE + i
        results, roc_data, fitted = run_single_split(df, seed, protocol_name)
        for name, m in results.items():
            all_results[name].append(m)
        if i == 0:  # keep the first (seed=42) split as the representative one
            rep_roc_data = roc_data
            rep_fitted = fitted

    summary = {}
    for name, metric_dicts in all_results.items():
        keys = [k for k in metric_dicts[0].keys() if k not in ("tn", "fp", "fn", "tp")]
        summary[name] = {}
        for k in keys:
            vals = [d[k] for d in metric_dicts]
            summary[name][f"{k}_mean"] = float(np.mean(vals))
            summary[name][f"{k}_std"] = float(np.std(vals))
        # also store the single-seed (seed=42) result as "headline" number,
        # analogous to how the paper reports one split's outcome
        headline = metric_dicts[0]
        for k in keys:
            summary[name][f"{k}_seed42"] = float(headline[k])

    return summary, rep_roc_data, rep_fitted, all_results


def plot_confusion_matrices(fitted, protocol_name):
    X_test_s, y_test, stack, base_models, scaler = fitted
    from sklearn.metrics import ConfusionMatrixDisplay
    models_to_plot = dict(base_models)
    # refit base models quickly is unnecessary; instead retrain fresh copies
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flat
    idx = 0
    # base models: refit on same data used for stack (already fit inside stack) -
    # for plotting we just retrain standalone copies for clarity
    return fig  # placeholder, replaced by full routine below


def full_confusion_and_roc(df, protocol_name, seed=RANDOM_STATE):
    X = df[FEATURES].values
    y = df[TARGET].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    base_models = get_base_models(random_state=seed)
    fitted_models = {}
    for name, model in base_models.items():
        m = clone(model)
        m.fit(X_train_s, y_train)
        fitted_models[name] = m
    stack = build_stack(random_state=seed)
    stack.fit(X_train_s, y_train)
    fitted_models["Stacking (proposed)"] = stack

    from sklearn.metrics import confusion_matrix, roc_curve, auc
    fig, axes = plt.subplots(2, 4, figsize=(19, 9))
    for ax, (name, m) in zip(axes.flat, fitted_models.items()):
        y_pred = m.predict(X_test_s)
        cm = confusion_matrix(y_test, y_pred)
        im = ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center",
                     color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["True 0", "True 1"])
        ax.set_title(name)
    for j in range(len(fitted_models), len(axes.flat)):
        axes.flat[j].axis("off")
    plt.suptitle(f"Confusion matrices - Protocol {protocol_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"confusion_matrices_{protocol_name}.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7, 6))
    for name, m in fitted_models.items():
        y_proba = m.predict_proba(X_test_s)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        lw = 2.5 if name == "Stacking (proposed)" else 1.3
        plt.plot(fpr, tpr, lw=lw, label=f"{name} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curves - Protocol {protocol_name}")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"roc_curves_{protocol_name}.png"), dpi=150)
    plt.close()


def main():
    df_full = load_data(DATA_PATH)
    df_dedup = df_full.drop_duplicates().reset_index(drop=True)

    print(f"Protocol A (as-published): n={len(df_full)}")
    print(f"Protocol B (de-duplicated): n={len(df_dedup)}")

    summary_A, roc_A, fitted_A, raw_A = run_protocol(df_full, "A_as_published")
    summary_B, roc_B, fitted_B, raw_B = run_protocol(df_dedup, "B_deduplicated")

    with open(os.path.join(OUT_DIR, "protocol_A_summary.json"), "w") as f:
        json.dump(summary_A, f, indent=2)
    with open(os.path.join(OUT_DIR, "protocol_B_summary.json"), "w") as f:
        json.dump(summary_B, f, indent=2)

    # Tidy CSV tables (seed=42 headline number, matching paper's single-run style)
    def to_headline_df(summary):
        rows = []
        for name, d in summary.items():
            rows.append({
                "model": name,
                "accuracy": d["accuracy_seed42"],
                "precision": d["precision_seed42"],
                "recall": d["recall_seed42"],
                "f1": d["f1_seed42"],
                "specificity": d["specificity_seed42"],
                "mcc": d["mcc_seed42"],
                "auc": d["auc_seed42"],
            })
        return pd.DataFrame(rows).set_index("model")

    def to_meanstd_df(summary):
        rows = []
        for name, d in summary.items():
            rows.append({
                "model": name,
                "accuracy": f"{d['accuracy_mean']:.4f} +/- {d['accuracy_std']:.4f}",
                "precision": f"{d['precision_mean']:.4f} +/- {d['precision_std']:.4f}",
                "recall": f"{d['recall_mean']:.4f} +/- {d['recall_std']:.4f}",
                "f1": f"{d['f1_mean']:.4f} +/- {d['f1_std']:.4f}",
                "auc": f"{d['auc_mean']:.4f} +/- {d['auc_std']:.4f}",
            })
        return pd.DataFrame(rows).set_index("model")

    headline_A = to_headline_df(summary_A)
    headline_B = to_headline_df(summary_B)
    meanstd_A = to_meanstd_df(summary_A)
    meanstd_B = to_meanstd_df(summary_B)

    headline_A.to_csv(os.path.join(OUT_DIR, "protocol_A_headline_seed42.csv"))
    headline_B.to_csv(os.path.join(OUT_DIR, "protocol_B_headline_seed42.csv"))
    meanstd_A.to_csv(os.path.join(OUT_DIR, "protocol_A_meanstd_30runs.csv"))
    meanstd_B.to_csv(os.path.join(OUT_DIR, "protocol_B_meanstd_30runs.csv"))

    print("\n=== Protocol A (as-published, n=1025) - seed 42 headline ===")
    print(headline_A.round(4))
    print("\n=== Protocol B (de-duplicated, n=302) - seed 42 headline ===")
    print(headline_B.round(4))
    print("\n=== Protocol A - mean +/- std over 30 random splits ===")
    print(meanstd_A)
    print("\n=== Protocol B - mean +/- std over 30 random splits ===")
    print(meanstd_B)

    # Figures: confusion matrices + ROC for both protocols
    full_confusion_and_roc(df_full, "A_as_published")
    full_confusion_and_roc(df_dedup, "B_deduplicated")

    # Comparison bar chart: paper-reported vs our two protocols (stacking model only).
    # NOTE: uses the mean-over-30-repeated-splits statistic (summary_A/B[*_mean]),
    # NOT a single seed's headline number, so that this figure is directly
    # consistent with Table 1 / the report's main narrative (which is built
    # around the repeated-split mean precisely because a single split is an
    # unreliable, high-variance estimate -- see assumption A2). A single split
    # (e.g. seed=42) can show Protocol A hitting a literal 100% by chance,
    # which would be a dramatic but statistically unrepresentative number to
    # headline.
    paper_reported = {"accuracy": 0.9853, "precision": 1.0000, "recall": 0.9853,
                       "f1": 0.9861, "auc": 0.9880}
    ours_A = {k: summary_A["Stacking (proposed)"][f"{k}_mean"] for k in paper_reported}
    ours_B = {k: summary_B["Stacking (proposed)"][f"{k}_mean"] for k in paper_reported}

    metrics_order = ["accuracy", "precision", "recall", "f1", "auc"]
    x = np.arange(len(metrics_order))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, [paper_reported[m] for m in metrics_order], width, label="Paper (reported, single split)")
    ax.bar(x, [ours_A[m] for m in metrics_order], width, label="Ours - Protocol A (as-published, mean of 30 splits)")
    ax.bar(x + width, [ours_B[m] for m in metrics_order], width, label="Ours - Protocol B (de-duplicated, mean of 30 splits)")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics_order])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Stacking ensemble: paper-reported vs. reproduced results")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "paper_vs_reproduction_comparison.png"), dpi=150)
    plt.close()

    comparison_df = pd.DataFrame({
        "Paper (reported)": paper_reported,
        "Ours - Protocol A (as-published)": ours_A,
        "Ours - Protocol B (de-duplicated)": ours_B,
    })
    comparison_df.to_csv(os.path.join(OUT_DIR, "paper_vs_reproduction_comparison.csv"))
    print("\n=== Paper vs. Reproduction comparison (Stacking model) ===")
    print(comparison_df.round(4))

    print("\nDone. All results written to results/part1/ and figures/.")


if __name__ == "__main__":
    main()
