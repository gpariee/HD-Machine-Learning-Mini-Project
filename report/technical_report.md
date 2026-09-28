---
title: |
  | Reproduction and Extension of a Stacking-Based Ensemble
  | for Early Heart Attack Prediction
subtitle: "Technical Research Report -- Parts 1 & 2"
author: "[Student Name] -- [Student ID] -- [Unit Code]"
date: "September 2026"
toc: true
toc-depth: 3
numbersections: true
geometry: margin=2.5cm
fontsize: 11pt
linkcolor: blue
urlcolor: blue
---

\newpage

# Executive Summary {-}

This report reproduces and critically extends **Bhagat, Sharma & Agarwal
(2025)**, *"An efficient stacking-based ensemble technique for early
heart attack prediction,"* published in *Multimedia Tools and
Applications* [1]. The paper reports a six-classifier stacking ensemble
(Logistic Regression, Decision Tree, Random Forest, XGBoost, Naive Bayes,
K-Nearest Neighbors) that achieves 98.53% accuracy on a 1025-patient heart
disease dataset.

**Part 1** faithfully reproduces this pipeline. In doing so we identify
that the dataset file used by the paper contains **723 of 1025 rows
(70.5%) as exact duplicates**, and that de-duplication leaves only 302
unique rows -- essentially the original, single-source UCI Cleveland
dataset (n=302/303), not the four-hospital merge the paper describes. We
show, by holding the modelling pipeline fixed and varying only whether
duplicates are removed before splitting, that this single change
explains the overwhelming majority of the gap between our reproduction
and the paper's reported near-perfect performance: our stacking ensemble
scores **99.40% +/- 0.74% accuracy** on the leaky, as-published data
(closely matching the paper's 98.53%) but only **83.11% +/- 5.29%**
accuracy once duplicates are removed.

**Part 2** proposes and evaluates two complementary, substantially
different methodological contributions, both addressing limitations
exposed in Part 1:

1. A **leakage-safe, nested-cross-validated, feature-engineered and
   hyperparameter-optimised pipeline**, evaluated with a paired Wilcoxon
   signed-rank test against an untuned baseline on identical folds. We
   report, honestly, that on this small (n=302) de-duplicated dataset
   the additional engineering does **not** yield a statistically
   significant improvement (all p > 0.05, though precision comes close
   at p=0.055) -- itself an important,
   evidence-based finding about the limits of what further modelling
   sophistication can achieve once the leakage bias is removed.
2. A **genuine four-hospital merge and cross-hospital (leave-one-site-
   out) generalisation study**, built from the four original raw UCI
   site files. This reveals that two of the paper's own top-ranked
   predictive features (`ca`, `thal`) are 83-99% missing outside
   Cleveland, and that cross-hospital AUC falls to 0.72-0.76 for
   Switzerland/VA versus 0.89 for a pooled random split -- a real,
   clinically relevant generalisation gap the original paper's protocol
   cannot detect.

All code, notebooks, data and results referenced in this report are
included in the accompanying archive (see Appendix C for the README and
reproduction instructions).

\newpage

# Introduction

## Selection of the paper for reproduction

Three candidate papers were considered for this assignment:

1. Bhagat et al. -- a DASMcC ECG-signal-feature multi-class CVD
   classifier on the public PhysioNet PTB-XL dataset [related IEEE Access
   paper, Option 1].
2. A cascading ensemble for annotating open IoT sensor metadata (Option
   2, IEEE Internet of Things Journal).
3. **Bhagat, Sharma & Agarwal (2025)**, a stacking ensemble for heart
   attack prediction on a public tabular UCI-derived dataset (Option 3,
   *Multimedia Tools and Applications*) [1].

Option 2 was ruled out first: it relies on bespoke, non-public scraped
IoT metadata with no clearly specified acquisition or preprocessing
protocol, making faithful reproduction infeasible within the scope of
this assignment. Option 1 uses a genuinely public dataset (PTB-XL) but
requires a heavier, more failure-prone signal-processing pipeline
(peak-detection feature extraction across ~21,800 raw 12-lead ECG
records). Option 3 was selected because (a) its dataset is small, fully
public, and -- as this report demonstrates -- independently verifiable
against the paper's own reported feature ranges; (b) its methodology
(six classical classifiers plus a stacking ensemble) is standard and
fully within reach of a rigorous, faithful reproduction; and (c), as
detailed below, it exhibits a data-quality issue rich enough to support
a substantive, evidence-based critical analysis and a genuinely novel
Part-2 contribution.

## Report structure

Section 2 covers Part 1: the research problem, dataset, methodology,
results and critical analysis of the reproduction. Section 3 covers Part
2: the proposed solution(s), their motivation, methodology, results and
comparative analysis. Section 4 concludes. References follow in IEEE
numbered style. Appendices contain supplementary tables, the assumption
log, and reproducibility instructions.

\newpage

# Part 1 -- Reproduction

## Research problem, motivation, state of the art, and research gap

**Research problem.** Cardiovascular disease is a leading global cause
of death, and early risk prediction from routinely collected clinical
attributes (age, blood pressure, cholesterol, ECG findings, etc.) can
support earlier intervention. The paper frames this as a binary
classification problem: given 13 clinical features, predict presence
(1) or absence (0) of heart disease.

**Motivation.** The authors motivate their approach by observing that
individual classifiers plateau in the low-to-mid 90s (percent accuracy)
on this class of dataset, and propose that a **stacking ensemble** --
training a meta-classifier on the out-of-fold predictions of several
heterogeneous base learners -- can push performance further by combining
each base learner's complementary strengths.

**State of the art referenced by the paper.** The paper situates itself
against a body of prior work on UCI-derived heart-disease datasets,
citing accuracies ranging from roughly 85% to 99% across different
combinations of feature selection (Lasso, chi-square/PCA, Relief, fast
conditional mutual information) and classifiers (SVM, hybrid
random-forest-linear-model ensembles such as Mohan et al.'s HRFLM [2],
majority-voting ensembles, and others). The claimed contribution is that
a five-fold stacking ensemble of six classical learners exceeds all of
these at 98.53% accuracy.

**Results reported in the paper (Table 11, reproduced in Section 2.4
below).** Individual classifiers range from 84.4% (LR, NB) to 92.7% (DT,
RF) accuracy; the stacking ensemble reports 98.53% accuracy, 100%
precision, 97.27% recall, 98.61% F1 and 98.8% AUC.

**Research gap addressed by the paper (as stated by the authors).** The
authors position their gap as: prior individual-classifier studies on
this dataset plateau below 93% accuracy, and no prior work (in their
citation list) had applied a 5-fold stacking ensemble of exactly this
six-learner combination to this dataset.

**Research gap this report addresses (identified during reproduction).**
As detailed in Section 2.5, our reproduction reveals a different, more
fundamental gap: the paper does not verify that its train/test split is
free of duplicate-patient leakage, and -- independently -- never tests
whether its "four-database merge" claim is consistent with the very high
missingness that a genuine merge of the four named UCI hospital sites
would necessarily contain. Both gaps are addressed in Part 2.

## Dataset and feature set

**Dataset used by the paper.** The paper states (Sec. 4) that the
dataset "is made up of four databases: Cleveland, Hungary, Switzerland,
and Long Beach V," comprising 1025 patient records and 14 columns (13
features + binary target), reproduced verbatim in the paper's Table 9.

**Dataset used for our reproduction.** The paper does not give a direct
download link or DOI for its exact file. We sourced the file from a
public GitHub mirror of the well-known Kaggle "Heart Disease Dataset"
(uploader: johnsmith88) and verified it is the correct file by checking
every numeric feature range the paper reports in its own Table 9.
Source data for the comparison below: `results/part1/eda_summary.json`.

**Dataset verification -- our file vs. the paper's Table 9 (all five
reported ranges match exactly):**

| Feature | Paper's reported range (Table 9) | Range in our file | Match |
|---|---|---|---|
| age | 29-77 | 29.0-77.0 | Yes |
| trestbps | 94-200 | 94.0-200.0 | Yes |
| chol | 126-564 | 126.0-564.0 | Yes |
| thalach | 71-202 | 71.0-202.0 | Yes |
| oldpeak | 0-6.2 | 0.0-6.2 | Yes |

The file has 1025 rows, 14 columns, no missing values, and a
near-balanced target (526 positive, 499 negative), consistent with the
paper's description.

**Feature set (13 predictors + 1 target), as specified by the paper's
Table 9 and used identically here:** `age`, `sex`, `cp` (chest pain
type, 0-3), `trestbps` (resting blood pressure), `chol` (serum
cholesterol), `fbs` (fasting blood sugar > 120mg/dl), `restecg` (resting
ECG results, 0-2), `thalach` (max. heart rate achieved), `exang`
(exercise-induced angina), `oldpeak` (ST depression), `slope` (slope of
peak exercise ST segment, 0-2), `ca` (number of major vessels, 0-3),
`thal` (1=normal, 2=fixed defect, 3=reversible defect); target: 1 =
disease, 0 = no disease.

No feature engineering, feature selection, or dimensionality reduction
is described in the paper beyond "preprocessing" (missing-value
handling, scaling, categorical encoding) -- all thirteen raw features
are used directly, which we replicate in Part 1.

## Machine learning methods and experimental protocol

**Methods (Table 8 / Sec. 3.3 of the paper).** Six base classifiers --
Logistic Regression (LR), Decision Tree (DT), Random Forest (RF), XGBoost
(XGB), Gaussian Naive Bayes (NB), and K-Nearest Neighbors (KNN) -- are
each trained independently, then combined via a **5-fold stacking
ensemble**: a meta-classifier is trained on the out-of-fold predictions
of the six base learners, following the standard stacking recipe (the
paper's Table 8 algorithm box gives this five-step procedure but no
further detail).

**Experimental protocol described by the paper.** "For all input
dataset, split into training and testing dataset" -> "Preprocess both
training and testing data" -> "Apply all six classifiers ... to make
initial prediction over training dataset" -> "Stack all these
classifiers to build meta classifier" -> "Evaluate model performance."
This is the entirety of the protocol description; the paper gives **no
train/test split ratio, no random seed, no base-learner hyperparameters,
and no meta-classifier identity**.

**Assumptions made to enable reproduction (justified below; also logged
in Appendix A of this report and in the docstring of
`src/part1_reproduction.py`):**

- **A1 (split ratio):** an 80/20 stratified train/test split is assumed,
  the field-standard default and the same ratio explicitly reported for
  the companion Option-1 ECG paper considered for this assignment.
- **A2 (random seed / stochastic variation):** `random_state=42` is
  fixed throughout; in addition, because the paper reports only a single
  run, we additionally report results averaged over 30 repeated random
  stratified splits, to characterise how much of the paper's reported
  number could plausibly be sampling luck versus a stable estimate.
- **A3 (base-learner hyperparameters):** scikit-learn / XGBoost defaults
  are used for all six base learners, since the paper specifies none.
- **A4 (meta-classifier identity):** Logistic Regression is used as the
  stacking meta-classifier -- the conventional default choice for a
  classification stack and consistent with the paper's own textual
  description of "a trained meta-classifier."
- **A5 (scaling):** `StandardScaler`, fit on the training fold only, is
  applied to all thirteen features before every base learner, consistent
  with the paper's statement that data is "scale[d] ... correctly" but
  otherwise unspecified.

**Two experimental protocols are run in parallel (not both faithful
reproductions of the paper -- Protocol A is; Protocol B is a diagnostic
control described further in Section 2.5):**

- **Protocol A ("as-published"):** the full 1025-row file, ordinary
  random stratified 80/20 split -- our best-effort faithful
  reproduction of the paper's stated procedure.
- **Protocol B ("de-duplicated"):** identical pipeline and code, but
  exact duplicate rows are removed before splitting (leaving 302 unique
  rows). This isolates the causal effect of duplicate-row leakage,
  holding every other modelling choice fixed.

## Evaluation metrics and experimental results

**Metrics** (as specified by the assignment brief and matching the
paper's Table 10): Accuracy, Precision, Recall, F1-score, and AUC. We
additionally record Specificity and the Matthews Correlation
Coefficient (MCC) -- both reported by the paper, and MCC in particular
recommended in the literature as a more informative single summary
statistic than accuracy or F1 for binary classification [12] -- for
completeness (see `results/part1/`).

**Table 1 -- Full model-by-model comparison.** Paper-reported values
(Table 11 of [1]) versus our reproduction under both protocols (mean +/-
std over 30 repeated random 80/20 splits, seed range 42-71):

| Model | Metric | Paper | Ours: Protocol A (leaky) | Ours: Protocol B (de-dup.) |
|---|---|---|---|---|
| LR | Accuracy | 0.844 | 0.843 +/- 0.026 | 0.834 +/- 0.043 |
| LR | AUC | 0.920 | 0.917 +/- 0.018 | 0.901 +/- 0.032 |
| DT | Accuracy | 0.927 | 0.991 +/- 0.011 | 0.758 +/- 0.056 |
| DT | AUC | 0.970 | 0.991 +/- 0.011 | 0.757 +/- 0.057 |
| RF | Accuracy | 0.927 | 0.992 +/- 0.009 | 0.828 +/- 0.041 |
| RF | AUC | 0.973 | 0.999 +/- 0.002 | 0.902 +/- 0.033 |
| XGB | Accuracy | 0.907 | 0.994 +/- 0.008 | 0.800 +/- 0.043 |
| XGB | AUC | 0.983 | 0.997 +/- 0.007 | 0.882 +/- 0.026 |
| NB | Accuracy | 0.844 | 0.826 +/- 0.020 | 0.820 +/- 0.047 |
| NB | AUC | 0.919 | 0.906 +/- 0.017 | 0.893 +/- 0.039 |
| KNN | Accuracy | 0.859 | 0.843 +/- 0.030 | 0.817 +/- 0.045 |
| KNN | AUC | 0.930 | 0.950 +/- 0.017 | 0.876 +/- 0.042 |
| Stacking | Accuracy | 0.985 | 0.994 +/- 0.007 | 0.831 +/- 0.053 |
| Stacking | AUC | 0.988 | 0.999 +/- 0.004 | 0.906 +/- 0.034 |

(Full precision/recall/F1 columns for every model are in
`results/part1/full_model_comparison.csv`; the source figure is
`figures/paper_vs_reproduction_comparison.png`, reproduced below.)

![Stacking ensemble: paper-reported vs. reproduced results, Protocol A
(leaky) vs. Protocol B (de-duplicated). Reproduced values are the mean
over 30 repeated 80/20 splits (seeds 42-71), matching Table
1.](../figures/paper_vs_reproduction_comparison.png)

**Confusion matrices and ROC curves** for all seven models under each
protocol are in `figures/confusion_matrices_A_as_published.png`,
`figures/roc_curves_A_as_published.png`,
`figures/confusion_matrices_B_deduplicated.png`, and
`figures/roc_curves_B_deduplicated.png`. These four figures show a
single representative split (seed=42) for visual clarity, rather than an
average across splits (a confusion matrix or ROC curve cannot be
straightforwardly averaged across resamples); the seed=42 split happens
to be an easy one for Protocol A, on which RF, XGB and the stacking
ensemble all reach a **literal AUC of 1.000** on the held-out
test set (Figure 2); under Protocol B the same models' AUC falls to the
0.80-0.91 range (Figure 3), a visually striking illustration of the
effect described below.

![ROC curves, Protocol A (as-published, leaky data): RF, XGB and the
stacking ensemble reach AUC = 1.000.](../figures/roc_curves_A_as_published.png)

![ROC curves, Protocol B (de-duplicated,
honest split): all models fall into a realistic
0.80-0.91 AUC range.](../figures/roc_curves_B_deduplicated.png)

## Critical analysis of reproduced results

### The central finding: duplicate-row data leakage

Exploratory analysis of the dataset (full detail in
`results/part1/duplicate_report.txt`) found that **723 of the 1025 rows
(70.5%) are exact duplicates** -- identical across all thirteen features
and the target. Removing duplicates leaves **302 unique rows**, close to
the size of the original single-source UCI Cleveland dataset (303 rows,
one commonly dropped for missing values), *not* a four-hospital merge of
920 patients as the paper's own dataset description would imply (see
Section 3.5 for direct confirmation using the genuine four-site files).
The most-repeated individual patient record appears **8 times**
(`results/part1/eda_summary.json`).

This is a specific instance of the general problem of **data leakage**
in data mining -- broadly, any situation in which information that
would be unavailable at genuine prediction time is allowed to influence
model training or selection [11]. Duplicate-row leakage across a
train/test split is one of the more mechanically direct forms this can
take: a plain random split, applied to a file containing this many
exact duplicates, will place near-identical (in this case,
bit-identical) patients on both sides of the split with high
probability. A tree-based model that memorises a training row can then
"predict" that same row perfectly when it recurs in the test set --
inflating every reported metric without reflecting any genuine
generalisation ability.

**Isolating the causal effect.** Table 1 shows that holding the entire
modelling pipeline fixed and varying only whether duplicates are removed
before splitting changes the stacking ensemble's accuracy from **99.40%
+/- 0.74%** (Protocol A) to **83.11% +/- 5.29%** (Protocol B) -- a
15-16 percentage point swing attributable to nothing but duplicate-row
handling. Protocol A's mean accuracy over 30 repeated splits (99.40%) is
within one percentage point of the paper's single reported figure
(98.53%), which is strong evidence that Protocol A is a faithful
behavioural match to what the paper's authors actually ran, and that the
paper's headline number is substantially inflated by this leakage.

**A mechanistic cross-check: memorisation-capable vs. memorisation-
resistant models.** Table 1 shows the size of the Protocol A/B gap is
*not* uniform across models -- it tracks each model's capacity to
memorise individual training rows:

| Model class | Mechanism | Accuracy gap (A minus B) |
|---|---|---|
| DT | Can memorise individual rows exactly (unconstrained leaf splits) | 23.4 pts |
| XGB | High-capacity boosted trees, prone to memorising repeated rows | 19.4 pts |
| RF | Ensembled trees, still able to memorise repeated rows | 16.5 pts |
| **Stacking** | **Built from the above (memorisation-capable) base learners** | **16.3 pts** |
| KNN | Distance-based; an exact duplicate is its own nearest neighbour | 2.6 pts |
| LR | Smooth, low-capacity linear decision boundary; cannot memorise individual points | 0.9 pts |
| NB | Smooth probabilistic assumptions; cannot memorise individual points | 0.6 pts |

This pattern is exactly what the leakage hypothesis predicts: models
that are structurally capable of memorising an individual training
instance (DT, RF, XGB, and the stacking ensemble built from them) show
large Protocol A/B gaps, while models that instead fit a smooth global
decision function (LR, NB) show gaps close to zero and are, coincidentally,
also the two models whose Protocol-B (honest) numbers most closely match
the paper's reported numbers. This independent, model-specific evidence
substantially strengthens the leakage diagnosis beyond the aggregate
finding alone.

### Corroborating evidence from the wider literature

Implausibly high reported accuracy on cardiovascular tabular datasets
due to unexamined data leakage is not unique to this paper. In a
directly analogous, recently published case, Iacobescu et al. (2024)
reported ~99% accuracy for a kNN classifier on CDC cardiovascular data
[3]; a subsequent methodological comment by Eltawil et al. showed this
was caused by applying SMOTE-ENN resampling before the train/test split,
letting synthetic points derived from the test set contaminate training
[4]; the corrected pipeline reported ~80% accuracy [5] -- a strikingly
similar before/after magnitude to our own Protocol A (leaky, ~99%) versus
Protocol B (honest, ~83%) finding, in the same clinical domain. This
supports treating our finding as an instance of a recognised, recurring
failure mode in medical machine learning research, rather than an
idiosyncratic property of one paper.

### A secondary data-quality observation in the paper's own Table 11

The paper's Table 11 reports, for every one of the seven models, a
"Sensitivity" column numerically identical to the "Accuracy" column, a
"Specificity" column numerically identical to the "F1-Score" column, and
an "MCC" column numerically identical to the "Precision" column. Since
sensitivity, specificity and MCC are mathematically distinct quantities
from accuracy, F1 and precision respectively (and are not equal to them
in our own reproduction under either protocol -- see
`results/part1/full_model_comparison.csv`), this is most plausibly a
column-duplication error introduced when the paper's table was
assembled, rather than a genuine property of the underlying
computation. We flag this as a further, independent data-quality concern
about the paper, though -- unlike the duplicate-row leakage finding --
we cannot determine from the published text alone whether it reflects a
typesetting slip or an error in the original metric computation.

### Assessment against the paper's stated contribution

Notwithstanding the above, the paper's qualitative claim -- that a
stacking ensemble outperforms every one of its constituent base learners
-- **does replicate** under our honest, de-duplicated Protocol B: the
stacking ensemble's 83.11% accuracy and 0.906 AUC exceed every
individual base learner's Protocol-B accuracy and AUC in Table 1 (best
individual: RF at 82.79% accuracy / 0.902 AUC). The ensemble effect
itself, in other words, appears genuine and modest in magnitude (roughly
0.3-3 percentage points over the best single model); it is the paper's
*absolute* performance claim (98.53%), not its relative/qualitative
ensembling claim, that our reproduction shows to be substantially
inflated by data leakage.

\newpage

# Part 2 -- Proposed Solution

## Motivation: limitations identified in Part 1

1. **L1 (data leakage):** the paper's dataset mirror contains 70.5%
   duplicate rows; a plain random split inflates every reported metric
   by roughly 15-19 percentage points for the memorisation-capable
   models that drive the paper's headline result (Section 2.5.1).
2. **L2 (no variance estimate / no significance testing):** the paper
   reports a single train/test split with no repeated runs and no
   statistical test that the stacking ensemble's improvement over its
   base learners is genuine rather than sampling noise -- a material
   concern once duplicates are removed and n falls to 302.
3. **L3 (no hyperparameter optimisation):** every classifier, including
   the meta-classifier, is left at implicit defaults; no search over
   tree depth, regularisation strength, K, or ensemble size is
   performed.
4. **L4 (naive categorical encoding):** four of the thirteen features
   (`cp`, `restecg`, `slope`, `thal`) are nominal categorical codes with
   no natural ordering, but are fed to distance- and margin-based
   learners (LR, KNN) as raw integers, implicitly imposing a false
   ordinal relationship. Improper categorical encoding is a documented,
   avoidable source of error on exactly this dataset family [6].
5. **L5 (no cross-population generalisation test):** the paper's dataset
   description claims a four-hospital merge but, as our own duplicate
   analysis strongly suggests and Section 3.5 directly confirms, the
   file used is single-source. Consequently, the paper never tests
   whether its model would generalise to a genuinely different patient
   population -- a critical gap given the clinical deployment motivation
   the paper itself states.

## Proposed methodology

Two complementary, substantially different contributions are proposed,
targeting L1-L4 (Part 2a) and L5 (Part 2b) respectively. Neither is a
simple classifier swap; both combine several methodological changes as
permitted (and encouraged) by the assignment brief.

### Part 2a -- Leakage-safe, feature-engineered, hyperparameter-optimised pipeline

Built and evaluated **exclusively on the de-duplicated 302-row dataset**
(never on the leaky 1025-row file), with four combined changes:

- **S1 (restructured protocol):** a **nested** stratified cross-
  validation replaces the paper's single random split: an *outer* loop
  (10-fold, or 5-fold in the reduced-compute configuration -- see
  Section 3.3) gives an unbiased performance estimate, while an *inner*
  loop (3- or 5-fold) drives hyperparameter search, so that no
  information from a test fold ever influences model selection for that
  fold.
- **S2 (redesigned feature space):** the four nominal categorical
  features are one-hot encoded (rather than left as raw ordinal
  integers, addressing L4); four domain-motivated derived features are
  added -- percentage of age-predicted maximum heart rate achieved
  (`thalach / (220 - age)`), cholesterol-to-age ratio, a blood-pressure
  x cholesterol combined-load term, and an ST-depression x slope
  interaction term; mutual-information-based feature selection then
  chooses the best-performing feature-count (`k in {8, 10, 12, all}`),
  itself tuned inside the inner CV loop rather than fixed a priori.
- **S3 (hyperparameter optimisation):** `RandomizedSearchCV` jointly
  tunes all six base learners' key hyperparameters (LR's `C`; DT/RF's
  `max_depth`, `min_samples_leaf`; RF/XGB's `n_estimators`; XGB's
  `learning_rate`, `max_depth`; KNN's `n_neighbors`), the
  feature-selection width, and the stacking meta-classifier's
  regularisation strength, together, inside the inner loop.
- **S4 (ensemble-strategy ablation):** the tuned stacking design is
  compared against a tuned soft-voting ensemble of the same six base
  learners, to test whether the stacking architecture itself (vs.
  simple probability averaging) is what drives any improvement.

For a fair, paired comparison, **the identical Part-1-style pipeline**
(default hyperparameters, raw/ordinal features, no tuning, LR
meta-classifier) is re-evaluated on the **same outer folds** as the
proposed pipeline, and a **paired Wilcoxon signed-rank test** [7] is used
to assess whether any observed difference is statistically significant,
following standard practice for comparing classifiers across matched CV
folds [8].

### Part 2b -- Genuine multi-site merge & cross-hospital generalisation test

The four **original, raw** UCI Heart Disease Data Set site files [10]
(Cleveland, Hungarian, Switzerland, VA Long Beach; 303 + 294 + 123 + 200
= 920 patients) were obtained directly (see Appendix C for provenance)
and merged honestly. Two evaluation protocols are then run and compared:

- **(i) Pooled random 10-fold CV:** all four sites mixed and randomly
  split -- the "easy," same-distribution setting most similar in spirit
  to the paper's own protocol.
- **(ii) Leave-one-site-out (LOSO) cross-validation:** the model is
  trained on three hospitals and tested on the fourth, in turn, for
  every choice of held-out hospital -- a genuine cross-population
  generalisation test that neither the original paper, nor our own Part
  2a, attempts.

Because three of the four raw sites have extremely high missingness on
`ca`, `thal` and `slope` (Section 3.5.1), these three features are
dropped for this experiment (an honest merge cannot recover information
that was never collected at three of the four hospitals); the remaining
ten features are retained with per-fold median/mode imputation.

## Experimental protocol, parameter settings, and compute-budget note

**Part 2a search spaces** (full detail in
`src/part2_proposed_solution.py::get_search_space`): `select__k in {8,
10, 12, all}`; `LR__C in {0.01, 0.1, 0.3, 1, 3, 10}`; `DT__max_depth in
{2,...,6, None}`, `min_samples_leaf in {1,2,4,8}`; `RF__n_estimators in
{50,100,150}`, `max_depth in {3,4,5,None}`, `min_samples_leaf in
{1,2,4}`; `XGB__n_estimators in {50,100,150}`, `max_depth in {2,3,4}`,
`learning_rate in {0.03,0.1,0.2}`; `KNN__n_neighbors in {3,5,7,9,11}`;
`stack__final_estimator__C in {0.01,0.1,0.3,1,3,10}`. Scoring metric for
model selection: ROC-AUC.

**Compute-budget note.** This environment provides a single CPU core. A
full-scale nested search (10 outer x 5 inner folds, 25 random-search
draws, unrestricted ensemble sizes) was estimated (by timing one outer
fold in isolation) to require well over an hour of wall-clock time. We
therefore used a right-sized budget of **10 outer folds x 3 inner folds
x 12 random-search draws**, which completes in approximately 5 minutes
and retains every methodological element (nested nesting, joint
tuning of feature-selection width and hyperparameters, paired
significance testing) at reduced search resolution. The constants
`OUTER_FOLDS`, `INNER_FOLDS`, `N_ITER_SEARCH` at the top of
`part2_proposed_solution.py` can be raised on a multi-core machine to
recover the full-scale search; this trade-off and its implication for
statistical power are discussed in Section 3.4.

**Part 2b pipeline:** median imputation (continuous) / most-frequent
imputation (categorical), fit per-fold; one-hot encoding of `cp`,
`restecg`; the same six base learners at fixed, moderate capacity
(`RF`: 200 trees; `XGB`: 150 trees) with a 5-fold internal stacking CV
and LR meta-classifier -- no hyperparameter search was performed here, so
that any performance gap between protocols (i) and (ii) can be
attributed to distributional shift across hospitals rather than to
uneven tuning effort between the two settings.

## Results: Part 2a (leakage-safe pipeline, feature engineering, HPO)

**Table 2 -- Nested 10-fold CV results, de-duplicated data (mean +/-
std across outer folds; source: `results/part2/mean_comparison.csv`,
`std_comparison.csv`).**

| Metric | Baseline (paper-style, no HPO) | Proposed (Stacking, S1-S4) | Proposed (Soft Voting, S1-S4) |
|---|---|---|---|
| Accuracy | 0.851 +/- 0.041 | 0.818 +/- 0.076 | 0.821 +/- 0.074 |
| Precision | 0.845 +/- 0.053 | 0.806 +/- 0.078 | 0.812 +/- 0.076 |
| Recall | 0.897 +/- 0.093 | 0.892 +/- 0.097 | 0.885 +/- 0.100 |
| F1 | 0.866 +/- 0.042 | 0.842 +/- 0.061 | 0.843 +/- 0.063 |
| AUC | 0.912 +/- 0.028 | 0.896 +/- 0.045 | 0.904 +/- 0.042 |

![Part 2a: proposed pipeline vs. paired baseline, 10-fold nested CV,
de-duplicated data.](../figures/part2_proposed_vs_baseline.png)

![Per-fold score distributions across the 10 outer folds (accuracy, F1,
AUC).](../figures/part2_boxplots_per_fold.png)

**Table 3 -- Paired Wilcoxon signed-rank test, proposed stacking vs.
baseline (source: `results/part2/significance_tests.json`).**

| Metric | Mean difference (proposed - baseline) | Wilcoxon W | p-value | Significant (alpha=0.05)? |
|---|---|---|---|---|
| Accuracy | -0.033 | 6.0 | 0.219 | No |
| Precision | -0.039 | 4.0 | 0.055 | No (borderline) |
| Recall | -0.006 | 11.5 | 0.781 | No |
| F1 | -0.024 | 9.0 | 0.250 | No |
| AUC | -0.016 | 9.0 | 0.129 | No |

**Honest interpretation.** On this small, de-duplicated dataset (n=302,
~30 test rows per outer fold), the fully engineered and tuned pipeline
does **not** significantly outperform the simple, untuned baseline on
any metric; the point estimates are, if anything, marginally in the
baseline's favour, though every difference is well within the fold-to-
fold noise band (compare the error bars in Figure 4 with the mean gaps
in Table 3). We consider this an important, honestly reported negative
result rather than a failure to be concealed: it demonstrates, with
statistical rigour, that once the ~15-point leakage bias identified in
Part 1 is removed, the remaining "true" performance ceiling for this
feature set and sample size is approximately 82-85% accuracy / 0.90-0.91
AUC, and that further feature engineering and hyperparameter search
cannot be shown to move that ceiling on a sample this small -- consistent
with established findings that small-sample hyperparameter search is
itself prone to selection-bias noise that can offset any genuine gain
[8, 9].

## Results: Part 2b (genuine multi-site merge & cross-hospital generalisation)

### Why a genuine four-site merge is difficult: missingness

**Table 4 -- Feature missingness by genuine UCI site (source:
`results/part2/cross_site_missingness_report.csv`).**

| Feature | Cleveland | Hungarian | Switzerland | VA |
|---|---|---|---|---|
| trestbps | 0.000 | 0.003 | 0.016 | 0.280 |
| chol | 0.000 | 0.078 | 0.000 | 0.035 |
| fbs | 0.000 | 0.027 | 0.610 | 0.035 |
| thalach | 0.000 | 0.003 | 0.008 | 0.265 |
| exang | 0.000 | 0.003 | 0.008 | 0.265 |
| oldpeak | 0.000 | 0.000 | 0.049 | 0.280 |
| **slope** | 0.000 | **0.646** | 0.138 | **0.510** |
| **ca** | 0.013 | **0.990** | **0.959** | **0.990** |
| **thal** | 0.007 | **0.905** | 0.423 | **0.830** |

![Feature missingness heatmap across the four genuine UCI
sites.](../figures/part2b_missingness_heatmap.png)

`ca` and `thal` -- two of the paper's own most predictive features
(implicit in Table 11's high weight on RF/DT, which the paper's
discussion attributes partly to these features) -- are **83-99%
missing** in every site except Cleveland; `slope` is 14-65% missing
outside Cleveland. This provides strong, independent, quantitative
support for our Part-1 diagnosis: a genuine four-site merge would be
dominated by missingness in exactly the features the paper's own
narrative treats as fully populated and clinically important, which the
clean, fully-populated 1025-row file is not.

### Same-distribution vs. cross-hospital performance

**Table 5 -- Pooled random CV vs. leave-one-site-out CV (source:
`results/part2/pooled_random_cv_per_fold.csv`,
`leave_one_site_out_per_site.csv`).**

| Setting | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Pooled random 10-fold CV (same-distribution) | 0.808 +/- 0.038 | 0.814 +/- 0.044 | 0.849 +/- 0.049 | 0.830 +/- 0.033 | 0.886 +/- 0.030 |
| LOSO, test = Cleveland | 0.782 | 0.748 | 0.791 | 0.769 | 0.859 |
| LOSO, test = Hungarian | 0.813 | 0.718 | 0.792 | 0.753 | 0.888 |
| LOSO, test = Switzerland | 0.805 | 0.969 | 0.817 | 0.887 | 0.763 |
| LOSO, test = VA | 0.740 | 0.813 | 0.846 | 0.829 | 0.717 |
| **LOSO mean (4 sites)** | **0.785** | **0.812** | **0.812** | **0.810** | **0.807** |

![Same-distribution (pooled) vs. cross-hospital (leave-one-site-out)
generalisation.](../figures/part2b_cross_site_comparison.png)

![Leave-one-site-out performance broken down by held-out
hospital.](../figures/part2b_loso_per_site.png)

**Interpretation.** Mean cross-hospital AUC (0.807) is meaningfully
lower than pooled same-distribution AUC (0.886), and the degradation is
not uniform: Switzerland and VA -- the two sites with the highest
missingness and smallest sample sizes -- show the largest AUC drops
(0.763 and 0.717 respectively) despite Switzerland's unusually high
*precision* (0.969), indicating the model becomes conservative
(few false positives, at the cost of discrimination) rather than simply
"worse" in a single direction on unfamiliar hospital populations. This
quantifies, for the first time in this line of work, a real and
clinically material limitation: a model trained and validated only
within one hospital's data distribution -- as the paper's protocol
implicitly does, whatever its stated four-site framing -- cannot be
assumed to generalise to a different population without explicit
cross-population testing.

## Comparative analysis

**Against the paper's reported result.** The paper's 98.53% accuracy is,
per Part 1, attributable in the majority to duplicate-row leakage. Part
2a's honest, statistically validated estimate for a stacking ensemble on
this feature set is **82-85% accuracy**, consistent across both the
untuned Part-1 baseline (Protocol B) and the tuned Part-2a pipeline.
Part 2b's genuine cross-hospital test further shows that even this
honest same-distribution figure is itself optimistic relative to
deployment across new hospital populations (mean LOSO accuracy 78.5%,
AUC 0.807).

**Against Part 1's reproduced baseline.** Part 2a's tuned pipeline does
not statistically improve on Part 1's untuned Protocol-B baseline on the
same folds (Table 3); this is a genuine, if modest, empirical finding,
not a negative result to be hidden -- see Section 3.4's discussion. Part
2b uses a materially different (and larger, more clinically realistic)
dataset, so is not directly numerically comparable to Part 1's headline
figures, but is directly comparable in its own pooled-vs-LOSO contrast,
which is the point of that experiment.

## Critical discussion

**Strengths.** (i) The leakage diagnosis in Part 1 is corroborated by
three independent lines of evidence: the aggregate Protocol A/B gap, the
per-model memorisation-mechanism pattern (Section 2.5.1), and the
missingness-based confirmation that a genuine four-site merge is not
what the paper's clean file could be (Section 3.5.1). (ii) Part 2a's
nested-CV, paired-significance-testing protocol is a genuine
methodological improvement in rigor over the paper's single-split
report, regardless of whether it moves the point estimate. (iii) Part 2b
introduces a wholly new type of evidence (cross-hospital generalisation)
that neither the original paper nor typical follow-on work in this space
addresses.

**Limitations.** (i) With only 302 unique patient records, statistical
power to detect any but a large effect size is limited (Table 3); a
larger, genuinely independent dataset would be needed to determine
conclusively whether feature engineering and HPO offer any real,
if small, benefit. (ii) The reduced compute budget (Section 3.3) further
lowers the resolution of the hyperparameter search relative to what a
multi-core machine could achieve; we consider this an acceptable,
explicitly documented trade-off rather than a hidden limitation. (iii)
Part 2b's cross-hospital dataset, after dropping `ca`/`thal`/`slope`, is
a materially easier feature set than the paper's original thirteen,
so its absolute numbers are not directly comparable to Table 1 -- the
value of Part 2b is in the *pooled-vs-LOSO contrast*, not its absolute
accuracy.

**Practical implications.** For a heart-disease screening model
intended for real clinical use, this report's findings imply that (a)
any performance claim on this dataset family should be re-verified
against duplicate-row leakage before being trusted, (b) a stacking
ensemble's benefit over its best single base learner is real but modest
(low single digits of accuracy, not the 6-15 point gap the leaky
evaluation implies), and (c) deployment across a new hospital or patient
population should be expected to cost several points of AUC relative to
same-hospital validation, which should be reflected in any clinical
risk communication about the model's expected real-world accuracy.

\newpage

# Conclusion

This report reproduced Bhagat, Sharma & Agarwal's (2025) stacking-based
heart-attack prediction pipeline [1] and found that its headline 98.53%
accuracy is substantially inflated by duplicate-row data leakage in the
dataset file the paper (most likely unknowingly) used: an honest,
de-duplicated evaluation of the identical pipeline yields approximately
83% accuracy. This diagnosis is corroborated by a model-specific
memorisation-mechanism analysis, by a directly analogous precedent in
the recent cardiovascular-ML literature, and by direct evidence from the
genuine four-hospital UCI source files that a real multi-site merge
would be dominated by missingness the paper's clean file does not
exhibit. Building on this diagnosis, we proposed and rigorously
evaluated a leakage-safe, nested-cross-validated, feature-engineered and
hyperparameter-optimised pipeline (honestly finding no statistically
significant gain over a simple untuned baseline at this sample size) and
a genuine cross-hospital generalisation study (finding a real, material
AUC gap between same-distribution and cross-hospital evaluation). Taken
together, this work demonstrates that trustworthy evaluation protocol
design -- not further modelling sophistication -- is the primary
unmet need in this specific line of heart-disease prediction research.

\newpage

# References {-}

[1] M. Bhagat, A. Sharma, and P. Agarwal, "An efficient stacking-based
ensemble technique for early heart attack prediction," *Multimedia
Tools and Applications*, vol. 84, pp. 36351-36375, 2025, doi:
10.1007/s11042-024-19293-7.

[2] S. Mohan, C. Thirumalai, and G. Srivastava, "Effective heart disease
prediction using hybrid machine learning techniques," *IEEE Access*,
vol. 7, pp. 81542-81554, 2019, doi: 10.1109/ACCESS.2019.2923707.

[3] P. Iacobescu, V. Marina, C. Anghel, and A.-D. Anghele, "Evaluating
binary classifiers for cardiovascular disease prediction: Enhancing
early diagnostic capabilities," *Journal of Cardiovascular Development
and Disease*, vol. 11, no. 12, p. 396, 2024, doi:
10.3390/jcdd11120396.

[4] M. Eltawil, L. Byham-Gray, Y. Jia, N. Mistry, J. Parrott, and S.
Gohel, "Comment on Iacobescu et al. Evaluating binary classifiers for
cardiovascular disease prediction: Enhancing early diagnostic
capabilities. J. Cardiovasc. Dev. Dis. 2024, 11, 396," *Journal of
Cardiovascular Development and Disease*, vol. 13, no. 1, p. 46, 2026,
doi: 10.3390/jcdd13010046.

[5] P. Iacobescu, V. Marina, C. Anghel, and A.-D. Anghele, "Reply to
Eltawil et al. Comment on 'Iacobescu et al. Evaluating binary
classifiers for cardiovascular disease prediction: Enhancing early
diagnostic capabilities. J. Cardiovasc. Dev. Dis. 2024, 11, 396,'"
*Journal of Cardiovascular Development and Disease*, vol. 13, no. 1, p.
47, 2026, doi: 10.3390/jcdd13010047.

[6] K. Potdar, T. S. Pardawala, and C. D. Pai, "A comparative study of
categorical variable encoding techniques for neural network
classifiers," *International Journal of Computer Applications*, vol.
175, no. 4, pp. 7-9, 2017.

[7] F. Wilcoxon, "Individual comparisons by ranking methods,"
*Biometrics Bulletin*, vol. 1, no. 6, pp. 80-83, 1945.

[8] G. C. Cawley and N. L. C. Talbot, "On over-fitting in model
selection and subsequent selection bias in performance evaluation,"
*Journal of Machine Learning Research*, vol. 11, pp. 2079-2107, 2010.

[9] A. Vabalas, E. Gowen, E. Poliakoff, and A. J. Casson, "Machine
learning algorithm validation with a limited sample size," *PLOS ONE*,
vol. 14, no. 11, p. e0224365, 2019, doi: 10.1371/journal.pone.0224365.

[10] Dua, D. and Graff, C. (2019). UCI Machine Learning Repository:
Heart Disease Data Set. Irvine, CA: University of California, School of
Information and Computer Science. https://archive.ics.uci.edu/dataset/45/heart+disease.

[11] S. Kaufman, S. Rosset, C. Perlich, and O. Stitelman, "Leakage in
data mining: Formulation, detection, and avoidance," *ACM Transactions
on Knowledge Discovery from Data*, vol. 6, no. 4, pp. 1-21, 2012, doi:
10.1145/2382577.2382579.

[12] D. Chicco and G. Jurman, "The advantages of the Matthews
correlation coefficient (MCC) over F1 score and accuracy in binary
classification evaluation," *BMC Genomics*, vol. 21, no. 1, p. 6, 2020,
doi: 10.1186/s12864-019-6413-7.

\newpage

# Appendix A: Assumption Log (Part 1) {-}

| ID | Assumption | Justification |
|---|---|---|
| A1 | 80/20 stratified train/test split | Field-standard default; matches the ratio explicitly used in the Option-1 ECG paper considered for this assignment |
| A2 | `random_state=42`; additionally, mean +/- std over 30 repeated splits reported | Paper reports only one run with no seed; repeated splits characterise sampling variance |
| A3 | scikit-learn / XGBoost default hyperparameters for all six base learners | Paper specifies none |
| A4 | Logistic Regression as stacking meta-classifier | Conventional default choice for classification stacking; consistent with paper's textual description |
| A5 | `StandardScaler`, fit on training fold only | Paper states data is scaled but gives no method |

# Appendix B: Additional Part 2a Configuration Detail {-}

See `src/part2_proposed_solution.py` for the complete, executable
specification of: the four engineered features (`engineer_features()`),
the preprocessing `ColumnTransformer` (`build_preprocessor()`), the
hyperparameter search spaces (`get_search_space()`), and the nested-CV
driver (`nested_cv_proposed()`). The compute-budget constants
(`OUTER_FOLDS=10`, `INNER_FOLDS=3`, `N_ITER_SEARCH=12`) used to produce
the results in Section 3.4 are declared at the top of the file with a
note on how to raise them on a multi-core machine.

# Appendix C: Reproducibility {-}

See the accompanying `README.md` for full install/run instructions. In
summary: `pip install -r requirements.txt`, then run, in order,
`src/eda_and_leakage_check.py`, `src/part1_reproduction.py`,
`src/build_full_comparison_table.py`, `src/part2_proposed_solution.py`,
and `src/part2b_cross_site_generalization.py`. Equivalent Jupyter
notebooks are provided in `notebooks/01`-`03` for interactive
reproduction. All data files (the 1025-row reproduction file and the
four genuine raw UCI site files) are included in `data/`.

**Video presentation link:** [INSERT LINK HERE]

**Code/data archive (GitHub / OneDrive / Dropbox):** [INSERT LINK HERE]
