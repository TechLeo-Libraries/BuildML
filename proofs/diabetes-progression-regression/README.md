# diabetes-progression-regression

## Business purpose

Predict quantitative diabetes disease progression from baseline clinical
covariates. Exercises leakage-safe classical regression on a **real public
dataset**.

## Data source

**REAL_PUBLIC_DATASET** — `sklearn.datasets.load_diabetes` (Efron et al. LARS
diabetes study sample redistributed with sklearn). Offline; no network.

## Leakage controls

- Random train / validation / test before fitting
- Scaler fit on train only
- Validation for model choice; test once

## BuildML API steps

1. `ingest` → roles → `split`
2. `scale` → `fit(HistGradientBoostingRegressor)` (Ridge fallback)
3. `evaluate(validation)` → `evaluate(test)`
4. `save_pipeline`

## Metrics

Holdout R², RMSE, MAE. Refuses R² ≥ 1.0 and non-positive R².

## How to run

```bash
python proofs/diabetes-progression-regression/script.py
```

## Limitations

Small research sample; single seed; not clinical certification.
