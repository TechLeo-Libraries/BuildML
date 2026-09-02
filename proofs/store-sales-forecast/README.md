# store-sales-forecast

You have a daily store sales series with trend, weekly seasonality, and promo
spikes. You want a chronological holdout forecast after train-only seasonal
diagnostics.

## Data

Synthetic daily sales (`load_store_sales_synthetic`) with trend, weekly
seasonality, and promo spikes: license-clear.

## Leakage

`time_split` is chronological; latest rows are test.
`session.timeseries.analyze(scope="train")` does not peek at the future.
The forecast fits on train. Rolling metrics on validation are for disclosure.
Test `session.forecast.evaluate` runs only after the model is locked.

## How to run

```bash
python proofs/store-sales-forecast/script.py
python proofs/store-sales-forecast/baseline_industry.py
```

## What you'll get

`results/results.json` with rolling one-step forecast errors on validation
and test (MAE / RMSE / MAPE-style). `results/comparison.json` is statsmodels
SARIMAX (fallback: seasonal naive) with rolling one-step evaluation on the
same `time_split`. The Session path fits `session.forecast.fit(method="lag_ridge")`.
Analysis needs `statsmodels` when you want STL / diagnostics.

## Limitations

Single synthetic series; not hierarchical multi-store M5.

Related: [Forecasting quickstart](../../guides/quickstart-forecasting.md),
[examples/forecast_lag_loop.py](../../examples/forecast_lag_loop.py).
