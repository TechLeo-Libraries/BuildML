# claim-severity-regression

## Business purpose

Estimate insurance claim severity (loss amount) from vehicle age, driver age, prior claims, urban flag, and deductible for reserving and pricing workflows.

## Data source

In-repo synthetic severity table (`load_claim_severity_synthetic`): license-clear, deterministic. **Not** a real P&C claims extract.

## Leakage controls

- Random train / validation / test before any fitting
- Scaler fit on train only
- Model choice reads validation only
- Test evaluated once after selection
- Industry Ridge twin uses the same SplitPlan indices

## BuildML API steps

1. `Session.ingest` → `set_roles` → `split`
2. `scale` → `fit(HistGradientBoostingRegressor)` (Ridge fallback)
3. `evaluate(validation)` → `evaluate(test)`
4. `save_pipeline`

## Metrics

Primary holdout: R², RMSE, MAE on test (see `results/results.json`).

## Industry comparison (Tier C)

Filled: sklearn `StandardScaler` + `Ridge` twin via `baseline_industry.py` → `results/comparison.json`.

## Limitations

- Synthetic severity; no Tweedie / GLM severity stack
- Single seed; not actuarial certification

## How to run

```bash
python proofs/claim-severity-regression/script.py
python proofs/claim-severity-regression/baseline_industry.py
```
