# energy-load-forecast

You have hourly grid load with temperature and lag features. You want an
honest chronological split for operations planning, then one test forecast.

## Data

In-repo synthetic hourly energy load (`load_energy_load_synthetic`):
license-clear, deterministic. Not a real ISO / utility extract.

## Leakage

`time_split`: chronological train -> validation -> test.
`session.timeseries.analyze` is scoped to train only. The forecast fits on
train; selection metrics read validation. Test `session.forecast.evaluate`
runs after the model is locked. The twin uses the same `time_split`.

## How to run

```bash
python proofs/energy-load-forecast/script.py
python proofs/energy-load-forecast/baseline_industry.py
```

## What you'll get

`results/results.json` with MAE, RMSE, and MAPE (rolling one-step) on test.
`results/comparison.json` is seasonal naive (period=24) or a Ridge lag twin
selected on validation. The Session path fits
`session.forecast.fit(method="lag_ridge", horizon=24)`.

## Limitations

Synthetic load; no multi-zone hierarchy. lag_ridge is a classical baseline,
not energy SOTA.

Related: [Forecasting quickstart](../../guides/quickstart-forecasting.md),
[examples/forecast_lag_loop.py](../../examples/forecast_lag_loop.py).
