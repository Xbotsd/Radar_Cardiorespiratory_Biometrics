# Evaluation protocol

This file documents the decisions that determine every number in the paper.
It exists because most of the contribution is *protocol*, not model.

## 1. Windowing

| Task | Window | Hop | Cap per recording | Reason |
|---|---|---|---|---|
| Identification (19-D) | 15 s | 3 s | 56 | matches the original analysis, so the leaky number is reproducible |
| Cross-species / age (16-D) | 20 s | 10 s | 20 (dog), 10 (human) | long enough for stable power fractions |
| Vital signs, window level | 30 s | 15 s | — | ≥ 3 respiratory cycles, ≥ 20 cardiac cycles |
| Vital signs, recording level | whole recording | — | — | reference is a single mean |

The final partial window is always dropped (`range(0, n - win, hop)`), matching
the original code so that counts are comparable.

## 2. Splits

**Closed set, P0 (reported for comparison only).**  Stratified 5-fold over
windows with shuffling.  Overlapping windows of the same recording land in both
train and test.  This is the protocol of the earlier submission and it inflates
the score; it is kept as the upper reference.

**Closed set, P1 (headline).**  Per subject, the first 60 % of windows train
and the last 40 % test, with a **guard gap of 5 windows** (15 s = one window
length) so that no test window overlaps a training window.  The gap is swept
from 0 to 10 windows and the result is flat (46.1–47.4 %), which shows the
improvement over P0 is not an artefact of the particular gap.

**Closed set, P2 (long horizon).**  Awake dogs only.  Train on windows 0–56
(first ~3 min), test on windows starting after 25 min.  Same animals, same
session, 22 min of separation.

**Open set.**  30 dogs are randomly partitioned into 10 known / 20 unknown,
20 repeats.  Within each known dog the windows are split temporally into
fit (45 %) / calibration (15 %) / test (40 %) with guard gaps.  The rejection
threshold is a quantile of the **calibration** scores only, targeting 90 %
acceptance; the unknown dogs never influence it.  The fixed split used
previously (known = anaesthetised No1–No10) is also reported, and it is the one
that gives the misleading 89.6 % rejection.

**Cross-species.**  Stratified **grouped** 5-fold, groups = (dataset, subject),
so no subject appears in both folds.  Window-level CV is also reported; the
difference between the two is small (90.7 % vs. 90.2 %), which is expected
because species is a subject-level property.

**Age.**  Subject-level, features are the median over that subject's windows,
5-fold subject-disjoint CV, compared with the mean predictor.  A 200-repeat
subject-label permutation gives the null.

## 3. Estimators

**Vital signs.**  Per channel: breathing peak → notch every harmonic of the
breathing rate that falls in the cardiac band → cardiac peak.  Channels are
discarded when HR/BR falls outside 4–12 (dogs) or 2.5–12 (humans).  Surviving
channels are fused with an SNR-weighted median.  The ablation in
`results/vitals_summary.json` removes, one at a time, the harmonic notch, the
ratio gate, the weighted-median fusion, and the multi-bin pooling; the fusion
and the notch each matter, the ratio gate barely does.

**Classifiers.**  Random forest (300 trees), RBF-SVM (C = 10, γ = scale), LDA,
5-NN and logistic regression, all behind a training-fold `StandardScaler`.
Hyper-parameters are fixed before evaluation and are **not** tuned per fold.

## 4. Open-set scores

Four scores are compared, all oriented so that larger means "more known":

* `Mahalanobis` — negative minimum Mahalanobis distance to a class centroid,
  shared covariance, pseudo-inverse regularised by 1e-3.
* `Euclidean` — negative minimum distance to a centroid.
* `MaxProb` — maximum random-forest class probability.
* `Top2Margin` — difference between the two largest class probabilities.

## 5. What would count as a positive result

For the open-set experiment, AUROC ≥ 0.7 for known-vs-unknown.  Observing
≈ 0.53 with 20 repeats is the negative result reported in the paper: the
fixed-split 89.6 % rejection does not survive resampling of which dogs are
known.  We did not keep searching for a split that produces a high number.

For cross-species, the pre-specified positive control was 90 % accuracy, and
the negative control (human 60 GHz vs. human 120 GHz, same species, different
hardware) was run **before** interpreting the 90 % as species information.

## 6. Reproducibility

`SEED = 42` everywhere.  No GPU.  Runtime on 32 vCPU / 32 GB: full pipeline
< 15 min after the one-off CSV→NPY cache.  Feature tables are committed so the
classifier scripts can be re-run without the raw data.
