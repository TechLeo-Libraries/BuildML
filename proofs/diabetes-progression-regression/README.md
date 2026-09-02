# diabetes-progression-regression

You have baseline clinical covariates and a quantitative diabetes progression
target. You want leakage-safe classical regression on a public table, with
metrics in the target's units.

## Data

**REAL_PUBLIC_DATASET** -- `sklearn.datasets.load_diabetes` (Efron et al. LARS
diabetes study sample redistributed with sklearn). Offline; no network.

## Leakage

Random train / validation / test before fitting. The scaler learns from train
only. Validation is for model choice; test once.

## How to run

```bash
python proofs/diabetes-progression-regression/script.py
python proofs/diabetes-progression-regression/baseline_industry.py
```

## What you'll get

`results/results.json` with holdout R^2, RMSE, and MAE. The script refuses
R^2 >= 1.0 and non-positive R^2. `results/comparison.json` is a sklearn
`Pipeline` (`StandardScaler` + `HistGradientBoostingRegressor`, Ridge
fallback) twin on the same `SplitPlan`.

## Limitations

Small sample; single seed; not clinical certification.

Related: [Classical quickstart](../../guides/quickstart-classical.md).
