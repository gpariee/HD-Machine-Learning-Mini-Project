"""
Generates the three Jupyter notebooks (01, 02, 03) that mirror the src/
scripts, cell by cell, so markers can run and read the workflow
interactively. Each notebook imports the corresponding module from
../src and calls its main() function, then displays the key result
files inline.
"""
import nbformat as nbf
import os

NB_DIR = os.path.join(os.path.dirname(__file__), "..", "notebooks")
os.makedirs(NB_DIR, exist_ok=True)


def make_notebook(cells, filename):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    path = os.path.join(NB_DIR, filename)
    with open(path, "w") as f:
        nbf.write(nb, f)
    print(f"wrote {path}")


md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

# ---------------------------------------------------------------
# Notebook 1: EDA and leakage check
# ---------------------------------------------------------------
nb1_cells = [
    md("# 01 - Exploratory Data Analysis & Data-Leakage Verification\n"
       "**Paper reproduced:** Bhagat, A., Sharma, A., & Agarwal, S. (2025). "
       "\"An efficient stacking-based ensemble technique for early heart "
       "attack prediction.\" *Multimedia Tools and Applications*, "
       "84, 36351-36375.\n\n"
       "This notebook loads the dataset used in Part 1, verifies it "
       "against the paper's Table 9, and quantifies the duplicate-row "
       "issue that underlies the central finding of our reproduction "
       "(see the technical report, Section 3.4)."),
    code("import sys, os\n"
         "sys.path.append(os.path.abspath('../src'))\n"
         "import pandas as pd\n"
         "pd.set_option('display.max_columns', 20)"),
    md("## Load data and run the EDA / leakage-check script"),
    code("from eda_and_leakage_check import main as run_eda\n"
         "run_eda()"),
    md("## Inspect the summary JSON"),
    code("import json\n"
         "with open('../results/part1/eda_summary.json') as f:\n"
         "    summary = json.load(f)\n"
         "summary"),
    md("## Duplicate report (plain text)"),
    code("print(open('../results/part1/duplicate_report.txt').read())"),
    md("## Figures"),
    code("from IPython.display import Image, display\n"
         "display(Image('../figures/class_balance.png'))\n"
         "display(Image('../figures/correlation_heatmap.png'))\n"
         "display(Image('../figures/feature_distributions.png'))"),
    md("### Key finding\n"
       "723 of 1025 rows (70.5%) are exact duplicates; only 302 unique "
       "rows remain after de-duplication -- essentially the size of the "
       "original single-source UCI Cleveland dataset. This is the "
       "evidentiary basis for the two-protocol reproduction strategy "
       "used in notebook 02."),
]
make_notebook(nb1_cells, "01_eda_and_leakage_check.ipynb")

# ---------------------------------------------------------------
# Notebook 2: Part 1 reproduction
# ---------------------------------------------------------------
nb2_cells = [
    md("# 02 - Part 1: Faithful Reproduction of the Stacking Ensemble\n"
       "Reproduces the paper's six base classifiers (LR, DT, RF, XGBoost, "
       "NB, KNN) plus the 5-fold stacking ensemble, under two protocols:\n"
       "- **Protocol A (as-published):** the 1025-row file, random 80/20 "
       "stratified split -- our best-effort faithful reproduction of what "
       "the paper describes.\n"
       "- **Protocol B (de-duplicated):** the same pipeline on the 302 "
       "unique rows, to isolate the effect of the duplicate-row leakage "
       "identified in notebook 01.\n\n"
       "See `src/part1_reproduction.py` docstring for the full list of "
       "documented assumptions (A1-A5) made where the paper is silent on "
       "implementation detail."),
    code("import sys, os\n"
         "sys.path.append(os.path.abspath('../src'))\n"
         "import pandas as pd\n"
         "pd.set_option('display.max_columns', 20)"),
    md("## Run the full reproduction (both protocols, 30 repeated splits each)\n"
       "This takes a few minutes."),
    code("from part1_reproduction import main as run_part1\n"
         "run_part1()"),
    md("## Headline results (single split, seed=42) - matches the paper's single-run reporting style"),
    code("pd.read_csv('../results/part1/protocol_A_headline_seed42.csv', index_col=0).round(4)"),
    code("pd.read_csv('../results/part1/protocol_B_headline_seed42.csv', index_col=0).round(4)"),
    md("## Mean +/- std over 30 repeated random splits (our recommended, more robust estimate)"),
    code("pd.read_csv('../results/part1/protocol_A_meanstd_30runs.csv', index_col=0)"),
    code("pd.read_csv('../results/part1/protocol_B_meanstd_30runs.csv', index_col=0)"),
    md("## Full model-by-model comparison table (this is Table 1 in the technical report)\n"
       "Consolidates the paper's own Table 11 against both protocols, for all seven "
       "models -- not just the stacking ensemble."),
    code("from build_full_comparison_table import main as run_full_comparison\n"
         "run_full_comparison()"),
    code("pd.read_csv('../results/part1/full_model_comparison.csv', index_col=0)"),
    md("## Paper-reported vs. reproduced (stacking ensemble only)"),
    code("pd.read_csv('../results/part1/paper_vs_reproduction_comparison.csv', index_col=0).round(4)"),
    code("from IPython.display import Image, display\n"
         "display(Image('../figures/paper_vs_reproduction_comparison.png'))"),
    md("## Confusion matrices and ROC curves"),
    code("display(Image('../figures/confusion_matrices_A_as_published.png'))\n"
         "display(Image('../figures/roc_curves_A_as_published.png'))"),
    code("display(Image('../figures/confusion_matrices_B_deduplicated.png'))\n"
         "display(Image('../figures/roc_curves_B_deduplicated.png'))"),
    md("### Interpretation\n"
       "Protocol A's stacking ensemble (99.4% mean accuracy over 30 "
       "splits) closely matches the paper's reported 98.53% accuracy. "
       "Protocol B, identical in every other respect, drops to ~83%. "
       "Since the only difference between A and B is whether duplicate "
       "rows are allowed to appear on both sides of the train/test split, "
       "this ~15-16 percentage point gap is best explained by data "
       "leakage rather than by any property of the stacking architecture "
       "itself. See the technical report Section 4 for the full critical "
       "analysis."),
]
make_notebook(nb2_cells, "02_part1_reproduction.ipynb")

# ---------------------------------------------------------------
# Notebook 3: Part 2 proposed solution
# ---------------------------------------------------------------
nb3_cells = [
    md("# 03 - Part 2: Proposed Solution\n"
       "Two complementary contributions, both built on the de-duplicated, "
       "leakage-safe dataset (or, for 03b, the genuine multi-site data):\n\n"
       "**(a) Leakage-safe, feature-engineered, hyperparameter-optimised "
       "pipeline** (`part2_proposed_solution.py`): nested 10x3-fold CV, "
       "proper one-hot encoding of nominal features, four derived "
       "clinical features, mutual-information feature selection tuned "
       "in the inner loop, joint hyperparameter optimisation of all six "
       "base learners plus the meta-classifier, and a soft-voting "
       "ensemble ablation -- evaluated with a paired Wilcoxon "
       "signed-rank test against an untuned baseline on identical folds.\n\n"
       "**(b) Genuine multi-site merge & cross-hospital generalisation "
       "test** (`part2b_cross_site_generalization.py`): the four original "
       "raw UCI site files (Cleveland/Hungary/Switzerland/VA) are used to "
       "quantify (i) why an authentic four-site merge is dominated by "
       "missingness, and (ii) how much performance degrades under a "
       "leave-one-site-out cross-hospital evaluation versus a pooled "
       "random split."),
    code("import sys, os\n"
         "sys.path.append(os.path.abspath('../src'))\n"
         "import pandas as pd, json\n"
         "pd.set_option('display.max_columns', 20)"),
    md("## Part 2a: nested CV, feature engineering, HPO, statistical testing\n"
       "NOTE: this cell reproduces the full nested-CV search and can take "
       "several minutes on a single CPU core (see the compute-budget note "
       "in `part2_proposed_solution.py`)."),
    code("from part2_proposed_solution import main as run_part2a\n"
         "run_part2a()"),
    md("### Mean comparison table (10-fold nested CV)"),
    code("pd.read_csv('../results/part2/mean_comparison.csv', index_col=0).round(4)"),
    md("### Paired significance tests (Wilcoxon signed-rank, proposed stack vs. baseline)"),
    code("with open('../results/part2/significance_tests.json') as f:\n"
         "    print(json.dumps(json.load(f), indent=2))"),
    code("from IPython.display import Image, display\n"
         "display(Image('../figures/part2_proposed_vs_baseline.png'))\n"
         "display(Image('../figures/part2_boxplots_per_fold.png'))\n"
         "display(Image('../figures/part2_pooled_roc.png'))"),
    md("### Interpretation\n"
       "On this small (n=302), honest, de-duplicated dataset, the fully "
       "engineered/tuned pipeline does **not** significantly outperform "
       "a simple untuned stacking baseline (all p > 0.05). We report this "
       "honestly: the real contribution of 2a is a trustworthy, "
       "statistically validated *evaluation protocol* that corrects the "
       "paper's ~15-point leakage-driven overestimate, not a further "
       "numeric improvement that the sample size cannot support. See "
       "report Section 5 for discussion."),
    md("## Part 2b: genuine multi-site merge & cross-hospital generalisation"),
    code("from part2b_cross_site_generalization import main as run_part2b\n"
         "run_part2b()"),
    md("### Missingness across the four genuine UCI sites"),
    code("pd.read_csv('../results/part2/cross_site_missingness_report.csv', index_col=0)"),
    code("display(Image('../figures/part2b_missingness_heatmap.png'))"),
    md("### Pooled (same-distribution) vs. leave-one-site-out (cross-hospital)"),
    code("pd.read_csv('../results/part2/leave_one_site_out_per_site.csv', index_col=0).round(4)"),
    code("display(Image('../figures/part2b_cross_site_comparison.png'))\n"
         "display(Image('../figures/part2b_loso_per_site.png'))"),
    md("### Interpretation\n"
       "`ca` and `thal` -- two of the paper's own top-ranked predictive "
       "features -- are 83-99% missing in every site except Cleveland, "
       "which is strong independent evidence that the paper's dataset is "
       "not a genuine four-site merge. Using the reduced, genuinely "
       "mergeable 10-feature subset, cross-hospital (leave-one-site-out) "
       "AUC falls to as low as 0.72-0.76 for Switzerland/VA, versus 0.89 "
       "for a pooled random split -- a real and clinically relevant "
       "generalisation gap that the original paper's single-source "
       "evaluation protocol cannot reveal."),
]
make_notebook(nb3_cells, "03_part2_proposed_solution.ipynb")

print("All notebooks written.")
