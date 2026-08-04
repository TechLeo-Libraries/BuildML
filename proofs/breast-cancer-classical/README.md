# breast-cancer-classical

## Business purpose

Binary malignancy classification from diagnostic imaging-derived features
(Wisconsin breast cancer). Exercises leakage-safe classical Session fitting on a
**real public dataset** (not a synthetic blob).

## Data source

**REAL_PUBLIC_DATASET** — `sklearn.datasets.load_breast_cancer` (UCI Breast
Cancer Wisconsin Diagnostic, redistributed with sklearn). Offline; no network.

Provenance fields are written under `results/results.json` → `data`.

## Leakage controls

- Stratified train / validation / test before any fitting
- `cv_score` with `PreprocessRecipe` on train folds only
- Session-global impute / scale fit on train
- Threshold tuned on validation only; test once

## BuildML API steps

1. `Session.ingest` → `set_roles` → stratified `split`
2. `cv_score(..., preprocess=PreprocessRecipe(...))`
3. Re-inject split → `impute` → `scale` → `fit(LogisticRegression)`
4. `evaluate(validation)` → `tune_threshold` → `evaluate(test)`
5. `save_pipeline`

## Metrics

Holdout accuracy, F1, ROC-AUC (see `results/results.json`). Refuses perfect
scores (`>= 1.0`) as score ceilings.

## How to run

```bash
python proofs/breast-cancer-classical/script.py
```

## Limitations

Small n; single seed; not clinical certification.
