# Forecasting quickstart

```bash
pip install buildml
# ETS / ARIMA: pip install "buildml[timeseries]"
```

You need a `time` role and `time_split` (or a chronological `inject_split`).
Random, stratified, and group splits are refused. Fit is train-only.
Default method is `auto`: ETS if statsmodels is in, otherwise `lag_ridge`.

This is not the analysis path. Stationarity and decompose live on
`session.timeseries.*` and do not fit a forecaster.

[Forecasting deep](forecasting-deep.md) ·
[store-sales-forecast](../proofs/store-sales-forecast/) ·
[Time-series analysis](quickstart-timeseries-analysis.md)

---

## First loop: time_split → auto → evaluate → bundle

```python
import numpy as np
import pandas as pd

from buildml import Session

rng = np.random.default_rng(0)
n = 120
t = pd.date_range("2024-01-01", periods=n, freq="D")
y = 10 + 0.05 * np.arange(n) + np.sin(np.arange(n) / 7) + rng.normal(0, 0.3, n)
frame = pd.DataFrame({"ts": t, "y": y})

session = (
    Session.ingest(frame)
    .set_roles({"ts": "time", "y": "target"})
    .time_split(test_size=0.2, validation_size=0.2)
)

# auto → ETS when statsmodels installed, else lag_ridge
fit = session.forecast.fit(method="auto", horizon=7, seasonal_period=7)
fit.show()

val = session.forecast.evaluate(partition="validation", strategy="rolling_one_step")
test = session.forecast.evaluate(partition="test", strategy="rolling_origin")
print(val.metrics, test.metrics)

gen = session.forecast.generate(horizon=7)
print(gen.predictions)

bundle = session.forecast.save_bundle(".buildml-artifacts/forecast_bundle")
print(bundle)
```

`session.forecast.fit` **refuses** `session.split(...)` (random/stratified): use
`time_split`.

---

## Methods

| Method | Backend | Extra |
|--------|---------|-------|
| `auto` | ETS or lag_ridge | timeseries for ETS |
| `naive`, `seasonal_naive`, `drift`, `mean` | baseline |: |
| `lag_ridge`, `lag_hgb` | sklearn |: |
| `ets`, `arima`, `auto_arima`, `sarimax` | statsmodels | timeseries |
| `prophet` | Prophet | timeseries-prophet |
| `nbeats` | neuralforecast | timeseries-ml |

---

## Baselines before claiming model value

```python
naive = (
    Session.ingest(frame)
    .set_roles({"ts": "time", "y": "target"})
    .time_split(test_size=0.2, validation_size=0.2)
)
naive.forecast.fit(method="seasonal_naive", seasonal_period=7, horizon=7)
print(naive.forecast.evaluate(partition="test").metrics)
```

---

## Honesty bounds

- Not a digital twin or full econometrics lab (no cointegration product surface).
- Prophet/N-BEATS use synthetic daily `ds` alignment: disclose for irregular clocks.
- Univariate by default; exog requires future exog at generate time.
- Bundle format: `buildml.forecast_bundle.v2` (v1 still loadable).
