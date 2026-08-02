# Forecasting quickstart

> **Install first (GitHub):** PyPI `buildml` is still legacy 1.x and does **not**
> install Session 2.x. Use
> `pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"`
> (or an editable checkout).
>
> **Industry defaults:** `pip install "buildml[timeseries]"` for statsmodels
> ETS/ARIMA/SARIMAX. Prophet: `buildml[timeseries-prophet]`. N-BEATS:
> `buildml[timeseries-ml]`. Core lag/baseline fallback without extras.

Leakage-safe forecasting on the same `Session`: `time` role + `time_split`,
train-only fit, horizon generate, holdout MAE/RMSE/MAPE, rolling-origin eval,
and forecast bundle v2.

**Go deeper:** [Forecasting deep](forecasting-deep.md) ·

**Proof:** [store-sales-forecast](../proofs/store-sales-forecast/) (+ Tier C SARIMAX/seasonal_naive). Cross-domain: [harbor-demand-desk](../proofs/harbor-demand-desk/).
[Time-series analysis](quickstart-timeseries-analysis.md) ·
[Leakage](leakage-cv-recipes.md).

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
fit = session.fit_forecast(method="auto", horizon=7, seasonal_period=7)
fit.show()

val = session.evaluate_forecast(partition="validation", strategy="rolling_one_step")
test = session.evaluate_forecast(partition="test", strategy="rolling_origin")
print(val.metrics, test.metrics)

gen = session.generate_forecast(horizon=7)
print(gen.predictions)

bundle = session.save_forecast_bundle(".buildml-artifacts/forecast_bundle")
print(bundle)
```

`fit_forecast` **refuses** `session.split(...)` (random/stratified) — use
`time_split`.

---

## Methods

| Method | Backend | Extra |
|--------|---------|-------|
| `auto` | ETS or lag_ridge | timeseries for ETS |
| `naive`, `seasonal_naive`, `drift`, `mean` | baseline | — |
| `lag_ridge`, `lag_hgb` | sklearn | — |
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
naive.fit_forecast(method="seasonal_naive", seasonal_period=7, horizon=7)
print(naive.evaluate_forecast(partition="test").metrics)
```

---

## Honesty bounds

- Not a digital twin or full econometrics lab (no cointegration product surface).
- Prophet/N-BEATS use synthetic daily `ds` alignment — disclose for irregular clocks.
- Univariate by default; exog requires future exog at generate time.
- Bundle format: `buildml.forecast_bundle.v2` (v1 still loadable).
