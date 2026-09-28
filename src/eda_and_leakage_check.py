"""
eda_and_leakage_check.py
=========================
Exploratory data analysis for the "Heart Disease Dataset" (Kaggle: johnsmith88 /
mirrored here from a public GitHub repo) used by Bhagat, Sharma & Agarwal (2025),
"An efficient stacking-based ensemble technique for early heart attack prediction",
Multimedia Tools and Applications, 84:36351-36375.

This script verifies dataset shape/ranges against the paper's Table 9, and
quantifies the duplicate-row data-leakage issue that this specific dataset
mirror is known to contain (the "1025-row" file is the 303-row Cleveland
set replicated with repeats, not four independently merged UCI databases
as claimed in the paper).

Outputs (written to results/part1/):
  - eda_summary.json
  - duplicate_report.txt
  - class_balance.png
  - correlation_heatmap.png
  - feature_distributions.png
"""
import json
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "heart.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "part1")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.0)

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
TARGET = "target"

PAPER_RANGES = {
    "age": (29, 77), "trestbps": (94, 200), "chol": (126, 564),
    "thalach": (71, 202), "oldpeak": (0, 6.2),
}


def main():
    df = pd.read_csv(DATA_PATH)
    summary = {}

    summary["n_rows"] = int(df.shape[0])
    summary["n_cols"] = int(df.shape[1])
    summary["columns"] = list(df.columns)
    summary["missing_values_total"] = int(df.isna().sum().sum())
    summary["class_balance"] = df[TARGET].value_counts().to_dict()

    # --- Range check against paper's Table 9 ---
    range_check = {}
    for col, (lo, hi) in PAPER_RANGES.items():
        actual_lo, actual_hi = df[col].min(), df[col].max()
        range_check[col] = {
            "paper_range": [lo, hi],
            "observed_range": [float(actual_lo), float(actual_hi)],
            "matches": bool(actual_lo == lo and actual_hi == hi),
        }
    summary["range_check_vs_paper_table9"] = range_check

    # --- Duplicate analysis (the core leakage finding) ---
    n_total = df.shape[0]
    n_exact_dupes = int(df.duplicated().sum())
    df_unique = df.drop_duplicates()
    n_unique = df_unique.shape[0]

    dup_report_lines = []
    dup_report_lines.append("DUPLICATE ROW ANALYSIS")
    dup_report_lines.append("=" * 60)
    dup_report_lines.append(f"Total rows in file:              {n_total}")
    dup_report_lines.append(f"Fully duplicated rows (exact):    {n_exact_dupes} "
                             f"({100*n_exact_dupes/n_total:.1f}% of the dataset)")
    dup_report_lines.append(f"Unique rows after de-duplication: {n_unique}")
    dup_report_lines.append("")
    dup_report_lines.append("Interpretation: the paper (Sec. 4) states the dataset "
                             "'is made up of four databases: Cleveland, Hungary, "
                             "Switzerland, and Long Beach V' totalling 1025 records. "
                             "The de-duplicated row count found here (n={}) closely "
                             "matches the size of the original single-source UCI "
                             "Cleveland dataset (303 records, 1 of which is commonly "
                             "dropped due to missing values, giving 302). This is "
                             "strong evidence that the 1025-row file is the Cleveland "
                             "subset replicated with repeats rather than an "
                             "independent four-database merge, and that a plain "
                             "random train/test split on this file leaks near-"
                             "identical rows across the train and test partitions."
                             .format(n_unique))
    dup_report_lines.append("")

    # how many duplicate groups and their sizes
    dup_counts = df.groupby(FEATURES + [TARGET]).size().reset_index(name="count")
    dup_counts = dup_counts.sort_values("count", ascending=False)
    dup_report_lines.append("Top 10 most-repeated unique patient records:")
    dup_report_lines.append(dup_counts.head(10)[["count"]].to_string())
    summary["n_exact_duplicate_rows"] = n_exact_dupes
    summary["n_unique_rows"] = n_unique
    summary["max_repeat_count_single_record"] = int(dup_counts["count"].max())
    summary["mean_repeat_count"] = float(dup_counts["count"].mean())

    with open(os.path.join(OUT_DIR, "duplicate_report.txt"), "w") as f:
        f.write("\n".join(dup_report_lines))

    with open(os.path.join(OUT_DIR, "eda_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # --- Figures ---
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    df[TARGET].value_counts().sort_index().plot(
        kind="bar", ax=ax[0], color=["#4C72B0", "#C44E52"])
    ax[0].set_title("Class balance (full file, n=1025)")
    ax[0].set_xlabel("target (0=no disease, 1=disease)")
    ax[0].set_ylabel("count")
    ax[0].set_xticklabels(["0", "1"], rotation=0)

    df_unique[TARGET].value_counts().sort_index().plot(
        kind="bar", ax=ax[1], color=["#4C72B0", "#C44E52"])
    ax[1].set_title(f"Class balance (de-duplicated, n={n_unique})")
    ax[1].set_xlabel("target (0=no disease, 1=disease)")
    ax[1].set_xticklabels(["0", "1"], rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "class_balance.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(9, 7))
    corr = df[FEATURES + [TARGET]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, annot_kws={"size": 7})
    plt.title("Feature correlation matrix (full file)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()

    fig, axes = plt.subplots(3, 5, figsize=(18, 10))
    for i, col in enumerate(FEATURES):
        ax = axes.flat[i]
        sns.histplot(data=df, x=col, hue=TARGET, bins=20, kde=False,
                     multiple="stack", ax=ax, legend=(i == 0))
        ax.set_title(col)
    for j in range(len(FEATURES), len(axes.flat)):
        axes.flat[j].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "feature_distributions.png"), dpi=150)
    plt.close()

    print("EDA complete.")
    print(json.dumps(summary["range_check_vs_paper_table9"], indent=2))
    print(f"n_exact_duplicate_rows = {n_exact_dupes} / {n_total}")
    print(f"n_unique_rows = {n_unique}")


if __name__ == "__main__":
    main()
