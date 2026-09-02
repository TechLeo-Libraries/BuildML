# claim-severity-regression

You have vehicle age, driver age, prior claims, an urban flag, and deductible.
You want a holdout severity (loss amount) in the target's units for reserving
and pricing, not a classification accuracy.

## Data

In-repo synthetic severity table (`load_claim_severity_synthetic`):
license-clear, deterministic. Not a real P&C claims extract.

## Leakage

Random train / validation / test before any fitting. The scaler learns from
train only. Model choice reads validation only. Test is evaluated once after
selection. The Ridge twin uses the same `SplitPlan` indices.

## How to run

```bash
python proofs/claim-severity-regression/script.py
python proofs/claim-severity-regression/baseline_industry.py
```

## What you'll get

`results/results.json` with holdout R^2, RMSE, and MAE on test.
`results/comparison.json` is a sklearn `StandardScaler` + `Ridge` twin on the
same split. The Session path fits `HistGradientBoostingRegressor` (Ridge
fallback).

## Limitations

Synthetic severity; no Tweedie / GLM severity stack; single seed; not
actuarial certification.

Related: [Classical quickstart](../../guides/quickstart-classical.md).
