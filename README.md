# Theories of Theory

> **A hands-on project for mastering ML fundamentals** — every model, every metric, and
> explainability (SHAP) done properly, in one disciplined, reproducible pipeline.

This repo is a working record of learning ML **by building it correctly the first time** —
not by copying a tutorial and hoping the numbers mean something. Each project takes a real
dataset and runs it through a single, leakage-free protocol so that every model is compared
**apples-to-apples**, every metric is reported with the reason it exists, and every result can
be explained — not just observed.

---

## The mission

Most ML tutorials teach you to *call a model*. This project teaches you to **trust a result**:

- **Preprocess without leaking.** Fit-on-train-only, enforced structurally by putting
  imputation and scaling *inside* the pipeline — not by remembering to.
- **Evaluate honestly.** A single train/test split is one draw, not an answer. Every model is
  scored on a held-out set **and** under 5-fold cross-validation, so fragility surfaces.
- **Report the full metric vocabulary** — not just accuracy/R². RMSE, MAE, MAPE, sMAPE,
  Adjusted R², and the residuals that explain *where* a model fails, not just *how much*.
- **Explain the model, don't just run it.** SHAP turns a black box into a ranked, per-feature
  story — and cross-checks linear models against tree ensembles to expose where they agree
  and *why* they diverge.

---

## Projects

### 📊 The Metric & SHAP Benchmark

The flagship. **One dataset, one preprocessing protocol, many models** — benchmarked,
explained, and stress-tested.

| | | Status |
|---|---|---|
| **Part 1 — Regression** | Ames Housing · 8 models · 9 metrics · SHAP | ✅ **Complete** |
| **Part 2 — Classification** | Churn / Default · classification metrics · SHAP | 🚧 Planned |

➡️ **Deep dive:** [`part 1/README.md`](./part%201/README.md) — full methodology, results, and
the bugs we caught along the way.

---

## Part 1 at a glance — Regression on Ames Housing

**2,929 houses · 79 features → 328 after encoding · 8 models · 9 metrics · SHAP cross-analysis**

### Two honest winners

| | Holdout R² | Holdout RMSE | CV R² ($) | CV RMSE |
|---|---|---|---|---|
| **LinearRegression** | **0.944** | **$21,136** | 0.853 | $29,675 |
| **XGBoost** | 0.915 | $26,063 | **0.896** | **$24,946** |
| RandomForest | 0.897 | $28,722 | 0.879 | $26,908 |
| LassoCV | 0.915 | $26,097 | 0.855 | $29,406 |

OLS tops the *single split*; XGBoost tops *cross-validation*. The gap between those two
rankings is the project's most instructive result — see Finding 2 in the Part 1 README.

### Three findings worth the whole project

1. **Lasso selects, Ridge only shrinks** — LassoCV kept **116 of 328** features (α = 0.000625);
   RidgeCV kept all **328** (α = 10). L1 amputates, L2 only nudges.
2. **OLS wins the holdout by extrapolating** — a single CV fold emitted a log-price of `15.02`
   against a training max of `13.53`. Under `expm1` that became **$3.3M for a $160K house**,
   cratering its unclipped CV R² to 0.069 while log-space stayed 0.85. Trees *can't*
   extrapolate — that's why XGBoost held steady.
3. **SHAP and Lasso agree 19/20** — and the one disagreement (`2nd Flr SF`) is explainable:
   it is linearly derivable from `Gr Liv Area`, so L1 drops it as redundant while XGBoost
   uses the 2nd story's identity directly.

### 🐛 The bugs we caught

The real curriculum wasn't the models — it was the **silent failures** that ran fine and
produced plausible-but-wrong numbers. All 10 are logged in the
[Part 1 README](./part%201/README.md#lessons-learned--the-bugs-we-caught), including data
leakage before split-scoping, scoring against the wrong target (`log` vs dollars), and stale
variable names that silently displayed one model's results as another's.

**The meta-lesson:** most were *silent*. Catching them — not avoiding them — is the skill.

---

## Repository structure

```
theories of theory/
├── README.md                     ← you are here (project overview)
├── part 1/                       ← Part 1: Regression
│   ├── README.md                 ← deep dive: methodology, results, bug log
│   ├── pipeline.ipynb            ← runnable source (outputs stripped)
│   ├── pipeline_executed.ipynb   ← executed copy with all plots/SHAP
│   ├── requirements.txt          ← pinned dependencies
│   └── dataset/
│       └── AmesHousing.csv       ← source data (2,930 × 82)
└── webpage/
    └── index.html                ← rendered notebook (GitHub Pages view)
```

---

## Viewing & running

### 🌐 Read the results (no setup needed)
Open **[`webpage/index.html`](./webpage/index.html)** — the fully rendered notebook with every
plot, residual chart, and SHAP visualisation embedded. This is the GitHub Pages landing page.

### 🔬 Reproduce Part 1
```bash
cd "part 1"
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# run top-to-bottom (important: re-running cells out of order can corrupt state)
jupyter nbconvert --to notebook --execute --inplace pipeline.ipynb
```

**Reproducibility guarantees:** every randomized estimator carries `random_state=42`;
cross-validation uses `KFold(5, shuffle=True, random_state=42)`; preprocessing is fit inside
the CV loop, so CV numbers are leakage-free by construction.

### Environment (pinned)
Python 3.12 · scikit-learn 1.9.1 · pandas 3.0.6 · numpy 2.4.6 · xgboost 3.4.1 ·
shap 0.54.1 · matplotlib 3.11.2 · seaborn 0.13.2

---

## Part 2 — next up

Part 1 was **regression**. Part 2 completes the benchmark on **classification** — the same
discipline (one protocol, leakage-free, many models) applied to a binary target, with the
metric vocabulary switched to the classification family (accuracy, precision, recall, F1,
ROC-AUC, PR-AUC, confusion matrix, calibration) and SHAP used again for cross-model agreement.

**Dataset under selection:**
- **Telco Customer Churn** (~7K rows, balanced, heavy categorical mix — the classic)
- **Credit Card Default, Taiwan** (~30K rows, numeric-heavy, ~22% default — harder, more real)

---

*A working record of learning ML by building it right. Every number here is produced by the
committed notebook and is reproducible with the pinned environment.*
