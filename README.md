# Reproduction and Extension of a Stacking-Based Heart-Attack Prediction Model

**Paper reproduced (Part 1):** Bhagat, A., Sharma, A., & Agarwal, S. (2025).
"An efficient stacking-based ensemble technique for early heart attack
prediction." *Multimedia Tools and Applications*, 84, 36351-36375.

**Assignment components covered:**
- Part 1: faithful reproduction of the paper's stacking ensemble
- Part 2: proposed original solution (leakage-safe nested-CV pipeline +
  genuine multi-site cross-hospital generalisation study)
- Full technical report: `report/technical_report.pdf`

---

## 1. Project structure

```
.
├── data/
│   ├── heart.csv                  # 1025-row Kaggle mirror used by the paper (Part 1)
│   └── raw_sites/                 # 4 genuine raw UCI site files (Part 2b)
│       ├── cleveland.data
│       ├── hungarian.data
│       ├── switzerland.data
│       └── va.data
├── src/
│   ├── common.py                          # shared utilities (Part 1)
│   ├── eda_and_leakage_check.py           # EDA + duplicate-row leakage diagnosis
│   ├── part1_reproduction.py              # Part 1: faithful reproduction, Protocols A & B
│   ├── part2_proposed_solution.py         # Part 2a: nested CV, feature eng., HPO, stats test
│   ├── part2b_cross_site_generalization.py# Part 2b: genuine 4-site merge, LOSO CV
│   └── make_notebooks.py                  # (re)generates the notebooks below
├── notebooks/
│   ├── 01_eda_and_leakage_check.ipynb
│   ├── 02_part1_reproduction.ipynb
│   └── 03_part2_proposed_solution.ipynb
├── results/
│   ├── part1/   (CSV/JSON metrics tables produced by part1_reproduction.py)
│   └── part2/   (CSV/JSON metrics tables produced by part2*.py)
├── figures/     (all PNG figures referenced in the report)
├── report/
│   └── technical_report.pdf
├── requirements.txt
└── README.md   (this file)
```

## 2. Installation

Requires Python 3.10+.

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. How to reproduce every result in the report

Run from the `src/` directory, in this order. All outputs are written to
`results/` and `figures/` (already populated in this archive with the
results referenced in the report — re-running will overwrite them with a
fresh run; expect minor variation from stochastic model fitting, discussed
in the report Section 4.3).

```bash
cd src

# 1. EDA + duplicate-row / leakage diagnosis (Part 1, ~10 seconds)
python eda_and_leakage_check.py

# 2. Part 1: faithful reproduction of the paper (Protocols A & B, ~2-3 minutes)
python part1_reproduction.py

# 3. Build the consolidated paper-vs-reproduction comparison table used as
#    Table 1 in the report (~1 second; requires step 2's output files)
python build_full_comparison_table.py

# 4. Part 2a: leakage-safe nested-CV pipeline with feature engineering,
#    HPO and paired significance testing (~5-6 minutes on 1 CPU core;
#    faster on a multi-core machine — see the compute-budget note at the
#    top of part2_proposed_solution.py to raise OUTER_FOLDS/INNER_FOLDS/
#    N_ITER_SEARCH back to a full-scale search if you have more cores)
python part2_proposed_solution.py

# 5. Part 2b: genuine multi-site merge + cross-hospital generalisation test (~30 seconds)
python part2b_cross_site_generalization.py

# (optional) regenerate the notebooks from these scripts
python make_notebooks.py
```

Alternatively, open the three notebooks in `notebooks/` in order — each
notebook imports and runs the corresponding script(s) and displays the
resulting tables/figures inline. Launch with:

```bash
jupyter notebook notebooks/
```

## 4. Where each report section's evidence comes from

| Report section | Source file(s) |
|---|---|
| Sec. 3: dataset & leakage diagnosis | `results/part1/eda_summary.json`, `duplicate_report.txt`, `figures/class_balance.png` |
| Sec. 4: Part 1 reproduction results | `results/part1/protocol_[A/B]_*.csv`, `paper_vs_reproduction_comparison.csv`, `figures/roc_curves_*.png`, `confusion_matrices_*.png` |
| Sec. 5: Part 2a proposed pipeline | `results/part2/mean_comparison.csv`, `significance_tests.json`, `figures/part2_*.png` |
| Sec. 6: Part 2b cross-hospital study | `results/part2/cross_site_*.csv/json`, `leave_one_site_out_per_site.csv`, `figures/part2b_*.png` |

## 5. Data provenance

- `data/heart.csv`: the exact 1025-row, 14-column file used to reproduce
  the paper. We verified it against the paper's own Table 9 (feature
  ranges for age, trestbps, chol, thalach and oldpeak all match exactly).
  Mirrored from a public GitHub repository (originally sourced from the
  Kaggle "Heart Disease Dataset" by user johnsmith88); included here so
  the reproduction is self-contained.
- `data/raw_sites/*.data`: the four original, unmodified UCI Heart
  Disease Data Set files (Cleveland, Hungarian, Switzerland, VA Long
  Beach), donated 1988, sourced via a public GitHub mirror of the UCI
  Machine Learning Repository archive (https://archive.ics.uci.edu/dataset/45/heart+disease).

## 6. Video presentation & hosting

- Video presentation link: https://drive.google.com/file/d/1HLwDioGNV-u2mArWC496cAxhsx5nvSmZ/view?usp=drivesdk
- GitHub repository: https://github.com/gpariee/HD-Machine-Learning-Mini-Project

## 7. Notes on reproducibility and stochasticity

All scripts fix `random_state=42` and, where feasible, additionally
report results averaged over multiple repeated splits/folds (30 repeats
for Part 1, 10-fold nested CV for Part 2a) specifically so that a single
lucky/unlucky split cannot be mistaken for a genuine effect — this is
itself one of the paper's identified shortcomings (Section 2, limitation
L2). Minor (<1-2 percentage point) variation between runs on different
machines is expected and normal, arising from floating-point
non-determinism in XGBoost/BLAS across platforms; this does not affect
any of the report's conclusions, which are based on differences an order
of magnitude larger.
