# store-sales-forecast

## Business purpose

Forecast daily store sales for replenishment and promo planning, with
train-only seasonal diagnostics before locking a forecast method.

## Data source

Synthetic daily sales (`load_store_sales_synthetic`) with trend, weekly
seasonality, and promo spikes: license-clear.

## Leakage controls

- `time_split` (chronological; latest rows = test)
- `session.timeseries.analyze(scope="train")`: no peek at future
- Model fit on train; rolling metrics on validation for disclosure
- Test `session.forecast.evaluate` only after the model is locked

## BuildML API steps

1. `ingest` → roles (`time`, `feature`, `target`) → `time_split`
2. `session.timeseries.analyze` (STL / diagnostics when `statsmodels` available)
3. `session.forecast.fit(method="lag_ridge", …)`
4. `session.forecast.evaluate` on validation then test
5. `session.forecast.generate` + `session.forecast.save_bundle`

## Metrics

Rolling one-step forecast errors on validation/test (MAE/RMSE/MAPE-style :
see JSON).

## Industry comparison (Tier C)

Industry twin: `baseline_industry.py` fits statsmodels SARIMAX (fallback: seasonal naive) with rolling one-step evaluation on the same `time_split`, writing `results/comparison.json`.
## Limitations

Single synthetic series; not hierarchical multi-store M5.
