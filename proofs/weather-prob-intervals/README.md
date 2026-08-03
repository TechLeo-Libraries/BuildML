# weather-prob-intervals

## Business purpose

Predict temperature with Bayesian Ridge plus conformal / quantile-style intervals so operations can plan with calibrated uncertainty bands.

## Data source

In-script synthetic weather regression (hour, humidity, pressure, wind → temp). **Not** a real METAR extract.

## Leakage controls

- Random train / validation / test before scale / fit
- Probabilistic model fit on train
- Interval calibration uses non-test partitions when required by the API
- Test evaluate after lock
- Industry twin uses the same SplitPlan

## BuildML API steps

1. `Session.ingest` → `set_roles` → `split` → `scale`
2. `fit_probabilistic(estimator="bayesian_ridge", conformal=True)`
3. `predict_interval(test)` → `evaluate_probabilistic(test)`
4. `save_probabilistic_bundle`

## Metrics

Primary holdout: regression metrics plus interval coverage / width (see `results/results.json`).

## Industry comparison (Tier C)

Filled — sklearn `BayesianRidge` + validation residual quantile twin via `baseline_industry.py` → `results/comparison.json`.

## Limitations

- Synthetic weather; empirical coverage ≠ guaranteed under distribution shift
- Single seed

## How to run

```bash
python proofs/weather-prob-intervals/script.py
python proofs/weather-prob-intervals/baseline_industry.py
```
