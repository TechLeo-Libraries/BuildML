# breast-cancer-classical

You have diagnostic imaging-derived features from the Wisconsin breast cancer
table and a malignancy label. You want a leakage-safe classical holdout on a
public dataset, not a synthetic blob.

## Data

**REAL_PUBLIC_DATASET** -- `sklearn.datasets.load_breast_cancer` (UCI Breast
Cancer Wisconsin Diagnostic, redistributed with sklearn). Offline; no network.

Provenance fields are written under `results/results.json` -> `data`.

## Leakage

Stratified train / validation / test before any fitting. `cv_score` with a
`PreprocessRecipe` runs on train folds only. Session-global impute and scale
learn from train. The threshold is tuned on validation only; test once.

## How to run

```bash
python proofs/breast-cancer-classical/script.py
python proofs/breast-cancer-classical/baseline_industry.py
```

## What you'll get

`results/results.json` with holdout accuracy, F1, and ROC-AUC. The script
refuses perfect scores (`>= 1.0`). `results/comparison.json` is a sklearn
`Pipeline` (`SimpleImputer` + `StandardScaler` + `LogisticRegression`) twin
on the same `SplitPlan`.

## Limitations

Small n; single seed; not clinical certification.

Related: [Classical quickstart](../../guides/quickstart-classical.md).
Paste without the harness:
[`examples/breast_cancer_classical_loop.py`](../../examples/breast_cancer_classical_loop.py).
