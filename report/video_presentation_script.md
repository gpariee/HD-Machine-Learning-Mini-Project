# Video Presentation Script & Slide Outline
### Reproduction and Extension of a Stacking-Based Ensemble for Early Heart Attack Prediction

Target length: 10-12 minutes. Timings are approximate — adjust to your
own pace. Speak from the bullet points; don't read verbatim.

---

## Slide 1 — Title (30 sec)
**Say:** "This is my reproduction and extension of Bhagat, Sharma and
Agarwal's 2025 paper, 'An efficient stacking-based ensemble technique
for early heart attack prediction,' published in Multimedia Tools and
Applications. I'll walk through what the paper claims, what I found when
I reproduced it, and the two original contributions I built in Part 2."

**Slide content:** Title, paper citation, your name/ID, unit code.

---

## Slide 2 — The paper's claim (60 sec)
**Say:** "The paper trains six classifiers — Logistic Regression,
Decision Tree, Random Forest, XGBoost, Naive Bayes, and KNN — on a
1025-patient heart disease dataset, then combines them with a 5-fold
stacking ensemble. They report the ensemble hits 98.53% accuracy and
98.8% AUC, beating every individual classifier and prior published work
on this dataset family."

**Slide content:** Paper's Table 11 summary (or just the headline
number, 98.53%), the six-classifier + stacking diagram.

---

## Slide 3 — Why I picked this paper (30 sec)
**Say:** "I compared three candidate papers. This one used a fully
public, small tabular dataset — meaning I could actually verify the data
itself, not just trust the paper's description. That turned out to
matter a lot."

**Slide content:** One-line comparison of the three options and why
this one was most tractable/verifiable.

---

## Slide 4 — Verifying the dataset (60 sec)
**Say:** "I sourced the exact file and checked it against the paper's
own Table 9 — every feature range matched exactly: age 29 to 77,
cholesterol 126 to 564, and so on. So I'm confident this is the right
file. But then I checked for duplicate rows."

**Slide content:** The range-check table (5 features, paper vs. mine,
all matching).

---

## Slide 5 — The central finding: data leakage (90 sec)
**Say:** "723 of the 1025 rows — over 70% — are exact duplicates. After
removing them, only 302 unique patients remain, which is basically the
size of the original single-source Cleveland dataset, not a genuine
four-hospital merge as the paper describes. That matters because a
random train/test split will place identical patients on both sides of
the split, letting models that can memorise training rows 'cheat' on the
test set."

**Slide content:** The duplicate-count figure (723/1025, 302 unique) and
the class-balance chart.

---

## Slide 6 — Isolating the effect (90 sec)
**Say:** "I ran the identical pipeline two ways: Protocol A on the
data exactly as the paper would have seen it, and Protocol B on the
same data with duplicates removed first. Protocol A's stacking ensemble
hits 99.4% accuracy — matching the paper's 98.53% almost exactly.
Protocol B, changing nothing except duplicate handling, drops to 83%.
That's a 16-point gap explained by nothing but data leakage."

**Slide content:** The `paper_vs_reproduction_comparison.png` bar chart
(paper vs. Protocol A vs. Protocol B).

---

## Slide 7 — Confirming the mechanism (60 sec)
**Say:** "If this really is memorisation, models that can memorise
individual rows — Decision Trees, Random Forest, XGBoost — should show
a huge gap between the two protocols. Models that fit a smooth decision
boundary and can't memorise single points — Logistic Regression, Naive
Bayes — should show almost no gap. That's exactly what I found: a
23-point gap for Decision Tree, under 1 point for Logistic Regression.
This is independent, model-specific evidence for the leakage diagnosis."

**Slide content:** The small mechanism table (model, mechanism, gap in
points) from Section 2.5.1 of the report.

---

## Slide 8 — It's not just this paper (30 sec)
**Say:** "This isn't a one-off. A very similar case was published and
then corrected in the *Journal of Cardiovascular Development and
Disease* in 2024 to 2026 — a kNN model reported 99% accuracy on
cardiovascular data, later traced to leakage from resampling before the
train/test split, and corrected down to about 80%. Same domain, same
order-of-magnitude correction."

**Slide content:** One citation line (Iacobescu et al. / Eltawil et al.,
J. Cardiovasc. Dev. Dis.).

---

## Slide 9 — Part 2, contribution 1: an honest pipeline (90 sec)
**Say:** "For Part 2, I built a leakage-safe pipeline: nested 10-by-3
fold cross-validation instead of a single split, proper one-hot encoding
of the categorical features instead of treating them as raw numbers,
four engineered clinical features, feature selection, and joint
hyperparameter tuning of all six models plus the meta-classifier. I then
ran a paired statistical test against a simple untuned baseline on the
exact same folds."

**Slide content:** Diagram of the four changes (S1-S4) or the
`part2_proposed_vs_baseline.png` chart.

---

## Slide 10 — An honest negative result (60 sec)
**Say:** "Here's the important part: the fancier pipeline did *not*
significantly beat the simple baseline — every p-value was above 0.10.
I'm reporting that honestly rather than hiding it, because it's itself
a meaningful finding: once you remove the leakage bias, there's a real
performance ceiling around 82 to 87% on this small 302-patient dataset
that further engineering can't move. That's a much more trustworthy
number than the paper's 98.53%."

**Slide content:** The Wilcoxon test results table.

---

## Slide 11 — Part 2, contribution 2: real cross-hospital testing (90 sec)
**Say:** "My second contribution goes further: I downloaded the actual
four original UCI hospital files — Cleveland, Hungary, Switzerland, and
the VA — and checked whether a genuine four-site merge is even possible.
It's not, cleanly: two of the paper's own top features, number of
vessels and thalassemia, are 83 to 99% missing at every site except
Cleveland. That's strong independent evidence the paper's clean file is
not a real four-site merge. Using the features that are usable at all
four sites, I then tested leave-one-hospital-out generalisation."

**Slide content:** The missingness heatmap.

---

## Slide 12 — The generalisation gap (60 sec)
**Say:** "Training on three hospitals and testing on the fourth drops
mean AUC from 0.89, for a same-distribution random split, down to about
0.81 — and as low as 0.72 for the VA site. That's a real, clinically
relevant gap that neither the original paper nor my own Part 2a
pipeline would ever reveal, because neither tests across hospitals."

**Slide content:** The pooled-vs-LOSO comparison chart.

---

## Slide 13 — Conclusion (60 sec)
**Say:** "In summary: the paper's headline number is inflated by
duplicate-row leakage, not a poor use of stacking — the ensemble idea
itself does hold up once evaluated honestly, just at about 83% rather
than 98.5%. My contributions show that a trustworthy evaluation
protocol — leakage-safe splitting, nested cross-validation, statistical
testing, and cross-hospital validation — matters more here than any
amount of further model tuning. All code, data and results are in the
submitted archive and fully reproducible."

**Slide content:** Three-bullet summary + link to your GitHub/OneDrive.

---

## Recording tips
- Screen-record your slides with narration (OBS Studio, PowerPoint's
  built-in recorder, or Zoom's "record to file" all work and are free).
- If time allows, briefly show one notebook running (e.g. notebook 01)
  rather than only static slides — markers like seeing the leakage
  diagnosis actually execute.
- Keep total length to what your unit's rubric asks for; trim Slides 7-8
  first if you need to shorten, since they're supporting evidence rather
  than core findings.
- Export/upload to YouTube (unlisted), Loom, or your institution's video
  platform, then paste the link into the report's Appendix C placeholder
  and into README.md.
