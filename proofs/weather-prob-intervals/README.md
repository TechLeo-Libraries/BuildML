# weather-prob-intervals

You have hour, humidity, pressure, and wind, and you want temperature with
calibrated uncertainty bands (Bayesian Ridge plus conformal / quantile-style
intervals).

## Data

In-script synthetic weather regression (hour, humidity, pressure, wind ->
temp). Not a real METAR extract.

## Leakage

Random train / validation / test before scale or fit. The probabilistic
model fits on train. Interval calibration uses non-test partitions when the
API requires it. Test evaluate runs after lock. The twin uses the same
`SplitPlan`.

## How to run

```bash
python proofs/weather-prob-intervals/script.py
python proofs/weather-prob-intervals/baseline_industry.py
```

## What you'll get

`results/results.json` with regression metrics plus interval coverage /
width. `results/comparison.json` is sklearn `BayesianRidge` plus a
validation residual quantile twin. The Session path is
`session.probabilistic.fit(estimator="bayesian_ridge", conformal=True)`.

## Limitations

Synthetic weather; empirical coverage is not a guarantee under distribution
shift; single seed.

Related: [Probabilistic quickstart](../../guides/quickstart-probabilistic.md),
[examples/probabilistic_bayesian_ridge.py](../../examples/probabilistic_bayesian_ridge.py).
