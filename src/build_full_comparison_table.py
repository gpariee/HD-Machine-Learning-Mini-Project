"""
build_full_comparison_table.py
================================
Consolidates the paper's own Table 11 (all six base classifiers + the
proposed stacking ensemble) against our Protocol A (as-published) and
Protocol B (de-duplicated) reproductions, for every model, not just the
stacking ensemble. Produces results/part1/full_model_comparison.csv,
the source for the report's main Part-1 results table.
"""
import json
import os
import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "part1")

# Verbatim from the paper's Table 11 (Bhagat, Sharma & Agarwal, 2025, p.36368).
# NOTE: as reported in the paper, the "Sensitivity" and "Specificity" columns
# are numerically identical to "Accuracy" and "F1-Score" respectively for
# every row, and "MCC" is identical to "Precision" for every row. We reproduce
# the paper's Accuracy/Precision/Recall/F1/AUC columns (the ones our brief
# asks us to compare) verbatim; the apparent Sensitivity/Specificity/MCC
# duplication is discussed as a data-quality observation in the report
# (Section 4.4) rather than treated as ground truth for those three metrics.
PAPER_TABLE_11 = {
    "LR":       {"accuracy": 0.8439, "f1": 0.8620, "recall": 0.9090, "precision": 0.8196, "auc": 0.9204},
    "NB":       {"accuracy": 0.8439, "f1": 0.8608, "recall": 0.9000, "precision": 0.8250, "auc": 0.9185},
    "KNN":      {"accuracy": 0.8585, "f1": 0.8687, "recall": 0.8727, "precision": 0.8648, "auc": 0.9304},
    "DT":       {"accuracy": 0.9268, "f1": 0.9289, "recall": 0.8909, "precision": 0.9702, "auc": 0.9702},
    "RF":       {"accuracy": 0.9268, "f1": 0.9333, "recall": 0.9545, "precision": 0.9130, "auc": 0.9734},
    "XGB":      {"accuracy": 0.9073, "f1": 0.9132, "recall": 0.9090, "precision": 0.9174, "auc": 0.9830},
    "Stacking (proposed)": {"accuracy": 0.9853, "f1": 0.9861, "recall": 0.9727, "precision": 1.0000, "auc": 0.9880},
}

MODEL_ORDER = ["LR", "DT", "RF", "XGB", "NB", "KNN", "Stacking (proposed)"]
METRICS = ["accuracy", "precision", "recall", "f1", "auc"]


def load_meanstd(protocol):
    df = pd.read_csv(os.path.join(OUT_DIR, f"protocol_{protocol}_meanstd_30runs.csv"), index_col=0)
    return df


def load_headline(protocol):
    df = pd.read_csv(os.path.join(OUT_DIR, f"protocol_{protocol}_headline_seed42.csv"), index_col=0)
    return df


def main():
    meanstd_A = load_meanstd("A")
    meanstd_B = load_meanstd("B")

    rows = []
    for model in MODEL_ORDER:
        row = {"model": model}
        for metric in METRICS:
            row[f"paper_{metric}"] = PAPER_TABLE_11[model][metric]
            row[f"protocolA_{metric}"] = meanstd_A.loc[model, metric] if model in meanstd_A.index else None
            row[f"protocolB_{metric}"] = meanstd_B.loc[model, metric] if model in meanstd_B.index else None
        rows.append(row)
    full = pd.DataFrame(rows).set_index("model")
    full.to_csv(os.path.join(OUT_DIR, "full_model_comparison.csv"))
    print(full.to_string())
    print(f"\nWritten to {os.path.join(OUT_DIR, 'full_model_comparison.csv')}")


if __name__ == "__main__":
    main()
