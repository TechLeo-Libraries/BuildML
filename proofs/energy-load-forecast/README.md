# energy-load-forecast

## Business purpose

Forecast hourly grid load using temperature and lag features with an honest chronological split for operations planning.

## Data source

In-repo synthetic hourly energy load (`load_energy_load_synthetic`) — license-clear, deterministic. **Not** a real ISO / utility extract.

## Leakage controls

- `time_split`: chronological train → validation → test
- `analyze_timeseries` scoped to train only
- Forecast fit on train; selection metrics on validation
- Test `evaluate_forecast` after model lock
- Industry twin uses the same time_split

## BuildML API steps

1. `Session.ingest` → `set_roles` → `time_split`
2. Optional `analyze_timeseries(scope="train")`
3. `fit_forecast(method="lag_ridge", horizon=24)`
4. `evaluate_forecast(validation)` → `evaluate_forecast(test)`
5. `generate_forecast` → `save_forecast_bundle`

## Metrics

Primary holdout: MAE, RMSE, MAPE (rolling one-step) on test.

## Industry comparison (Tier C)

Filled — seasonal naive (period=24) or Ridge lag twin selected on validation via `baseline_industry.py` → `results/comparison.json`.

## Limitations

- Synthetic load; no multi-zone hierarchy
- lag_ridge is a classical baseline, not energy SOTA

## How to run

```bash
python proofs/energy-load-forecast/script.py
python proofs/energy-load-forecast/baseline_industry.py
```
