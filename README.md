# The Metric & SHAP Benchmark — Part 1: Regression

> **Project 5, Part 1 of 2.** A single, disciplined pipeline that benchmarks **8 regression
> models** on Ames Housing under one identical preprocessing protocol, evaluates each with
> **9 metrics**, and uses **SHAP** to explain *why* the models agree — and where they diverge.

**Live notebook view:** open [`part 1/index.html`](part 1/index.html) (rendered via `nbconvert`; all plots
and SHAP visuals are embedded).

---

## Contents

1. [Results at a glance](#results-at-a-glance)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
   - [EDA](#eda)
   - [Missing-value strategy — the 4 branches](#missing-value-strategy--the-4-branches)
   - [Split discipline — how leakage was eliminated](#split-discipline--how-leakage-was-eliminated)
   - [Target transform](#target-transform)
4. [Pipeline architecture](#pipeline-architecture)
5. [Models benchmarked](#models-benchmarked)
6. [Evaluation metrics](#evaluation-metrics)
7. [Results](#results)
   - [Holdout leaderboard (586 houses)](#holdout-leaderboard-586-houses)
   - [5-fold cross-validation (leakage-free)](#5-fold-cross-validation-leakage-free)
8. [Key findings](#key-findings)
   - [Finding 1 — Lasso selects, Ridge only shrinks](#finding-1--lasso-selects-ridge-only-shrinks)
   - [Finding 2 — OLS wins the holdout, but it wins by extrapolating](#finding-2--ols-wins-the-holdout-but-it-wins-by-extrapolating)
   - [Finding 3 — SHAP and Lasso agree 19/20](#finding-3--shap-and-lasso-agree-1920)
9. [Lessons learned — the bugs we caught](#lessons-learned--the-bugs-we-caught)
10. [Repository structure](#repository-structure)
11. [How to reproduce](#how-to-reproduce)
12. [Part 2 preview](#part-2-preview)

---

## Results at a glance

| | Holdout R² | Holdout RMSE | CV R² ($) | CV RMSE | Verdict |
|---|---|---|---|---|---|
| **LinearRegression** | **0.944** | **$21,136** | 0.853 | $29,675 | Best single-split score |
| **XGBoost** | 0.915 | $26,063 | **0.896** | **$24,946** | Most robust under CV |
| Ridge (α=1) | 0.937 | $22,504 | 0.857 | $29,253 | Slight shrinkage, no win |
| RandomForest | 0.897 | $28,722 | 0.879 | $26,908 | Stable, untuned |
| LassoCV | 0.915 | $26,097 | 0.855 | $29,406 | Keeps 116/328 features |
| DecisionTree | 0.788 | $41,102 | 0.736 | $39,740 | Textbook overfitting |

**Two honest winners:** OLS tops the *single split*, XGBoost tops *cross-validation*. The gap
between those two rankings is the most instructive result in this project — see
[Finding 2](#finding-2--ols-wins-the-holdout-but-it-wins-by-extrapolating).

---

## Dataset

**[Ames Housing](https://www.kaggle.com/datasets/marcinantoniak/ames-housing)** (De Cock, 2011)

| Property | Value |
|---|---|
| Raw shape | 2,930 rows × 82 columns |
| After cleaning | **2,929 rows** × 80 columns |
| Target | `SalePrice` (USD, right-skewed) |
| Raw features | 79 (after dropping `Order`, `PID`) |
| Transformed features | **328** (one-hot expansion + scaling) |
| Train / test | 2,343 / 586 (`test_size=0.2`, `random_state=42`) |

Cleaning decisions (each verified before executing):
- Dropped `Order` and `PID` — pure row identifiers, zero predictive value.
- Dropped the single `MS SubClass = 150` row (index 2730) — a code that appears exactly once
  and encodes a partial-bath anomaly inconsistent with the rest of that record.
- `MS SubClass` cast to `str` — it is a **coded category** (20, 30, 60…), not a continuous
  quantity; treating it as numeric would impose a false ordinal relationship.

---

## Methodology

### EDA

Findings that shaped every downstream decision:

- **Target skew = 1.7435.** Right-skewed; a handful of $700K+ houses dominate any squared-error
  loss. Motivated the [log transform](#target-transform).
- **`Gr Liv Area` outliers.** A scatter of living-area vs price surfaced **5 houses > 4,000 sq ft**
  that are clear anomalies (large partial sales / poor data), sitting far off the main trend.
- **27 columns contain missing values**, sorted by rate — `Pool QC` tops the list at ~99.6%
  missing, which is itself meaningful (no pool ≠ data error; it means *no pool*).

### Missing-value strategy — the 4 branches

Every one of the 79 features is routed into exactly **one** of four branches, decided by *why*
the value is missing — not by a blanket imputation rule:

| Branch | Reason for missing | Fill | Count |
|---|---|---|---|
| **A — absence** | Missing *means* the amenity doesn't exist | constant `'None'` | 15 |
| **B — categorical** | Genuinely unknown category | `most_frequent` | 29 |
| **C — zero** | Absent feature ⇒ real value of 0 (sq ft, baths) | constant `0` | 8 |
| **D — numeric** | Genuinely unknown quantity | `median` | 27 |
| | | **Total** | **79** ✅ |

A **coverage gate** asserts that these four branches are disjoint and exactly cover all 79
features — nothing double-processed, nothing silently dropped by `remainder='drop'`:

```
A absence: 15 cols    B categorical: 29 cols
C zero: 8 cols        D median: 27 cols
sum: 79 (expect 79)   overlaps: NONE   uncovered: NONE   not-a-feature: NONE
```

> **Why this matters.** `Pool QC = NaN` does **not** mean "unknown pool quality" — it means
> *there is no pool*. Filling it with the median quality would invent a pool that doesn't exist.
> Branch A preserves that semantic distinction; branch D handles true uncertainty.

### Split discipline — how leakage was eliminated

The single most common error in applied ML is fitting preprocessing on the *entire* dataset.
This project defeats it structurally, not by discipline-alone:

1. Raw `X_train` / `X_test` are split **first** (80/20, `random_state=42`).
2. Imputers, one-hot encoders and scalers live **inside** a `ColumnTransformer`, composed of
   four mini-pipelines (`impute → encode/scale`).
3. `transformer.fit_transform(X_train)` — **fit on train only.**
4. `transformer.transform(X_test)` — **transform only; the test set is never fitted.**

Consequently every learned statistic (median, most-frequent category, scaler mean/std) comes
from the training fold alone. Verification: transformed shapes `(2343, 328)` / `(586, 328)`,
same feature count, **0 NaNs** on both sides.

### Target transform

`SalePrice` is log-transformed before training, then inverted for evaluation:

```python
y_train_raw, y_test_raw = y_train.copy(), y_test.copy()   # keep raw dollars
y_train = np.log1p(y_train)
y_test  = np.log1p(y_test)
```

| | Skew |
|---|---|
| Before | **1.7435** |
| After `log1p` | **≈ −0.106** (near-symmetric) |

Predictions are inverted with `np.expm1` and **all metrics are reported in dollars** — an RMSE
of "$21,136" is interpretable to anyone; "0.19 in log-space" is not. Keeping `y_*_raw` is what
makes dollar-space scoring possible without re-reading the CSV.

---

## Pipeline architecture

```
raw X_train (79 cols, strings + NaNs)
        │
        ▼
┌─ ColumnTransformer ──────────────────────────────────────────────┐
│  A  none_pipe    : impute 'None'  → OneHotEncoder                │  15 cols
│  B  other_pipe   : most_frequent  → OneHotEncoder                │  29 cols
│  C  zero_pipe    : impute 0       → StandardScaler               │   8 cols
│  D  median_pipe  : impute median  → StandardScaler               │  27 cols
│  remainder = 'drop'                                              │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
X_train_t (2343 × 328)   /   X_test_t (586 × 328)   — dense, scaled, zero NaNs
```

`OneHotEncoder(sparse_output=False, handle_unknown='ignore')` — dense output, and unknown
categories in the test set (or in a CV fold) become all-zero columns instead of crashing.

---

## Models benchmarked

| Model | Family | Regularization |
|---|---|---|
| `LinearRegression` | Linear | none (baseline) |
| `Ridge` (α=1) | Linear | L2 |
| `Lasso` (α=1) | Linear | L1 *(kept as the collapsed-α exhibit)* |
| `LassoCV` | Linear | L1, α by 5-fold CV |
| `RidgeCV` | Linear | L2, α by 5-fold CV |
| `DecisionTreeRegressor` | Tree | none (pruning) |
| `RandomForestRegressor` | Ensemble | bagging |
| `XGBRegressor` | Ensemble | boosting |

All randomized estimators are pinned with `random_state=42` so every number in this document is
reproducible.

---

## Evaluation metrics

| Metric | Formula | Notes |
|---|---|---|
| **MSE** | `mean((y − ŷ)²)` | Squared dollars; dominated by large errors |
| **RMSE** | `√MSE` | Dollars — same unit as the target |
| **MAE** | `mean(|y − ŷ|)` | Dollars, robust to outliers |
| **MAPE** | `mean(|y − ŷ| / |y|) × 100` | % — scale-free, human-readable |
| **sMAPE** | `mean(2|y − ŷ| / (|y| + |ŷ|)) × 100` | Symmetric in y and ŷ, bounded [0, 200] |
| **R²** | `1 − SS_res / SS_tot` | Variance explained |
| **Adjusted R²** | `1 − (1−R²)(n−1)/(n−p−1)` | Penalizes feature count; computed on **train** (a property of the fit) |
| **5-fold CV** | `KFold(5, shuffle=True, random_state=42)` | Leakage-free; see below |
| **Residuals** | `y − ŷ` | Diagnostic plots for all 8 models |

> **Adjusted R² caveat.** The formula is theoretically defined for *linear* models. For the
> tree ensembles it is reported for completeness but the "p predictors" assumption does not
> hold. Also, p=328 counts one-hot dummies — whether you count 328 columns or ~44 source
> variables changes the penalty. Textbook convention counts columns, so 328 is used here.

> **MAPE vs sMAPE.** For well-fit models the two nearly coincide (7.14% vs 7.10% for OLS) —
> errors are small relative to values. They diverge only where errors are large and asymmetric,
> e.g. the collapsed `Lasso(α=1)`: 33.88% vs 31.69%.

---

## Results

### Holdout leaderboard (586 houses)

Sorted by RMSE. All metrics in dollars; `Adjusted R²` computed on train.

| Model | train R² | test R² | RMSE | MAE | MAPE | sMAPE | Adj. R² |
|---|---|---|---|---|---|---|---|
| **LinearRegression** | 0.9419 | **0.9441** | **21,136** | **12,848** | **7.14%** | **7.10%** | 0.9324 |
| Ridge (α=1) | 0.9342 | 0.9366 | 22,504 | 13,791 | 7.48% | 7.46% | 0.9234 |
| RidgeCV | 0.9122 | 0.9180 | 25,588 | 14,821 | 7.81% | 7.81% | 0.8979 |
| XGBoost | 0.9982 | 0.9149 | 26,063 | 15,199 | 8.17% | 8.15% | 0.9979 |
| LassoCV | 0.8890 | 0.9147 | 26,097 | 15,159 | 7.95% | 7.93% | 0.8710 |
| RandomForest | 0.9839 | 0.8967 | 28,722 | 16,071 | 8.41% | 8.42% | 0.9813 |
| DecisionTree | 1.0000 | 0.7884 | 41,102 | 24,511 | 13.26% | 13.11% | 1.0000 |
| Lasso (α=1) | −0.0331 | −0.0405 | 91,153 | 60,168 | 33.88% | 31.69% | −0.2013 |

**Reading the table:**
- **OLS wins** on every holdout metric. RMSE $21,136 ≈ 7.1% average error — an excellent result.
- **DecisionTree is the overfitting exhibit:** train R² = 1.0000 vs test 0.7884. An unpruned
  tree memorizes. Its predictions are leaf means — every one ends in `…001`.
- **`Lasso(α=1)` collapsed.** Default α=1.0 is far too aggressive here: your log-target has
  tiny variance (values hover 9–13), so the L1 penalty zeroes *all 328* coefficients, leaving
  only the intercept. Its constant prediction is `exp(mean(log y)) = $165,957` — the
  **geometric** mean, which sits below the arithmetic mean ($182K). R² = −0.04 means it is
  worse than predicting the mean.

### 5-fold cross-validation (leakage-free)

The preprocessing lives **inside** the CV loop, so each fold refits imputation and scaling on
its own training portion — no fold's validation rows ever influence their own preprocessing.
Log-predictions are clipped to the training range before inversion (see
[Finding 2](#finding-2--ols-wins-the-holdout-but-it-wins-by-extrapolating)).

| Model | CV log-R² | CV RMSE | CV R² ($) | max $pred |
|---|---|---|---|---|
| **XGBoost** | 0.8747 | **24,946** | **0.8959** | $570,815 |
| RandomForest | 0.8693 | 26,908 | 0.8789 | $493,799 |
| RidgeCV | 0.8890 | 29,116 | 0.8582 | $755,000 |
| Ridge (α=1) | 0.8857 | 29,253 | 0.8568 | $755,000 |
| LassoCV | 0.8894 | 29,406 | 0.8553 | $755,000 |
| LinearRegression | 0.8709 | 29,675 | 0.8527 | $755,000 |
| DecisionTree | 0.7352 | 39,740 | 0.7358 | $755,000 |
| Lasso (α=1) | −0.0008 | 78,610 | −0.0338 | $166,652 |

**Under CV the ranking flips:** XGBoost leads, OLS drops to 6th. Reporting **both** `CV log-R²`
(the model's fit, fair) and `CV R² ($)` (practical, outlier-fragile) is what makes this visible.

---

## Key findings

### Finding 1 — Lasso selects, Ridge only shrinks

The clearest possible demonstration of L1 vs L2, side by side:

| | α chosen by CV | Features kept (of 328) |
|---|---|---|
| **LassoCV** | **0.000625** (tiny) | **116** — 212 zeroed |
| **RidgeCV** | **10.0** | **328** — none zeroed |

Lasso's tiny α is required because the log-target's variance is small; even so, L1 still
*amputates* 65% of the features. Ridge's L2 penalty **shrinks all coefficients toward zero but
never produces an exact zero** — it keeps every feature alive. Selection vs. shrinkage, quantified.

### Finding 2 — OLS wins the holdout, but it wins by extrapolating

OLS tops the single split (R² 0.944) yet loses under CV (R² 0.853 vs XGBoost's 0.896). The
reason is a fundamental property difference:

> **Trees cannot extrapolate.** A tree's prediction is the average of training labels inside a
> leaf, so it is always bounded by `[min(train_y), max(train_y)]`. A linear model has no such
> bound — it will happily predict a log-price beyond anything it saw.

The evidence, caught by cross-validation and invisible in the single split:

```
log-space CV R²  = 0.871   ← the model is fine where it operates
dollar CV R²     = 0.853   ← after clipping (was 0.069 before)
max $pred (OLS)  = $755,000  = exactly max(y_train)   ← clipping bound
max $pred (XGB)  = $570,815  < $755,000               ← tree never tries to exceed
```

**Before clipping**, one CV fold's OLS model emitted a log-prediction of **15.02** against a
training max of **13.53**. An error of ~1.5 in log space becomes a 20× blow-up under `expm1`:
`expm1(15.02) = $3,330,644` for a house that actually sold for **$160,000**. Two such
predictions above $1M cratered the dollar R² to **0.069**, while log-space R² stayed a
respectable 0.846. R² squares the error, so outliers dominate.

**Takeaway:** the holdout number was the lucky one. A single split hides fragility that CV
exposes — which is exactly why CV, not a lone holdout, is the honest arbiter. (Clipping
predictions to the training range is a legitimate bounded-prediction technique; both clipped
and unclipped numbers were examined so the effect is transparent.)

### Finding 3 — SHAP and Lasso agree 19/20

The project's central question: *does SHAP's feature attribution agree with Lasso's selection?*

```
Lasso kept 116 of 328 features
XGBoost top-20 kept by Lasso: 19/20     (identical at dummy level and source-variable level)
```

The top of both rankings: `Overall Qual`, `Gr Liv Area`, `Year Built`, `Total Bsmt SF`,
`1st Flr SF`, `Garage Cars`, `Garage Area`, `Neighborhood`. Two entirely different methods —
L1 coefficient shrinkage vs. Shapley-value attribution — converge on the same features.

**The single disagreement is the real lesson: `2nd Flr SF`.** XGBoost ranks it top-20; Lasso
zeroed it. This is not noise — it follows from a structural identity in Ames:

```
Gr Liv Area  =  1st Flr SF  +  2nd Flr SF  +  Low Qual Fin SF
```

`2nd Flr SF` is **linearly derivable** from `Gr Liv Area` and `1st Flr SF` — both of which
Lasso kept. A linear model can reconstruct it exactly, so L1 marks it redundant and drops it.
XGBoost, being non-linear and interaction-hungry, *does* use the 2nd-floor component
separately — a 2nd story affects value differently than ground-floor area at equal total size.

So the disagreement cleanly demonstrates the difference:
- **Lasso (linear):** *"derivable from its parts → redundant → zero it."*
- **XGBoost (non-linear):** *"the 2nd floor's identity matters independently → keep it."*

This is precisely what SHAP is for: a coefficient of 0 says "unused," but SHAP reveals the tree
*is* using it, and why. *(Honest caveat: with correlated features SHAP splits credit among them,
so treat exact ranks — not the membership — with care.)*

---

## Lessons learned — the bugs we caught

A running theme: **most failures were silent** — the notebook ran fine and produced plausible
numbers. Catching them is the actual skill.

| # | Bug | Symptom | Fix |
|---|---|---|---|
| 1 | In-place imputation on `X_train` before split-scoping | Leakage — test stats leaked into training | Move imputation into `ColumnTransformer`, fit on train only |
| 2 | `X_text` instead of `X_test` | `NameError` | Typo |
| 3 | Set-based `input_cols` | Column order non-deterministic across runs | Order-preserving `list(train_df.columns)` |
| 4 | `np.log1p(y_train)` computed but never assigned | Skew stayed 1.6 (raw) | `y_train = np.log1p(y_train)` |
| 5 | Fitted on raw `X_train` | `could not convert string to float: 'RL'` | Use transformed `X_train_t` |
| 6 | Metrics scored vs `y_test` (log) not `y_test_raw` (dollars) | MAE ≈ `preds.mean()`; R² = −22 billion | Score against the raw target |
| 7 | Stale `sv` in linear SHAP cells | Bar plots silently showed DecisionTree's ranking for 5 models | Distinct names per model |
| 8 | `results` vs `cv_results` | CV table displayed holdout numbers (identical output = the tell) | Reference the right dict |
| 9 | CV cell reset `preds_store` | Holdout predictions clobbered by CV predictions | Separate `cv_preds_store` |
| 10 | `.sum()` on scalar `alpha_` | `AttributeError: 'bool' object has no attribute 'sum'` | `alpha_` is a scalar — just print it |

**The meta-lesson:** the `results`/`cv_results` and `sv`/`sv_l` bugs are the same failure mode —
**reusing a stale variable name.** After renaming a variable for a new model, grep the cell for
the *old* name before running.

---

## Repository structure

```
part 1/
├── README.md                  ← this document
├── index.html                 ← rendered notebook (GitHub Pages view)
├── pipeline.ipynb             ← runnable source (outputs stripped)
├── pipeline_executed.ipynb    ← executed copy with all outputs/plots
├── requirements.txt           ← pinned dependencies
└── dataset/
    └── AmesHousing.csv        ← source data (2,930 × 82)
```

Backups (`pipeline.ipynb.bak*`) are local working artifacts and should be `part 1/.gitignore`d.

---

## How to reproduce

```bash
# 1. environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. run the notebook top-to-bottom
jupyter nbconvert --to notebook --execute --inplace pipeline.ipynb
#    (or open it in Jupyter and "Restart & Run All")

# 3. regenerate the rendered view
jupyter nbconvert --to html pipeline.ipynb --output index.html
```

**Reproducibility notes:**
- Every randomized estimator carries `random_state=42`.
- `KFold(5, shuffle=True, random_state=42)` for all cross-validation.
- The preprocessing is fit inside the CV loop, so CV numbers are leakage-free by construction.
- **Run top-to-bottom.** Re-running a single cell out of order (e.g. the log-transform twice) will
  silently corrupt state — see bug #4.

### Environment (pinned)

| Package | Version |
|---|---|
| Python | 3.12.14 |
| scikit-learn | 1.9.1 |
| pandas | 3.0.6 |
| numpy | 2.4.6 |
| xgboost | 3.4.1 |
| shap | 0.54.1 |
| matplotlib | 3.11.2 |
| seaborn | 0.13.2 |

---
