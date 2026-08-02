# Forecasting quickstart

> **Install first (GitHub):** PyPI `buildml` is still legacy 1.x and does **not**
> install Session 2.x. Use
> `pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"`
> (or an editable checkout). Classical forecasting uses core sklearn —
> **no optional extra**. See [installation](../docs/installation.rst).

Leakage-safe classical forecasting on the same `Session`: `time` role +
`time_split`, train-only fit, horizon generate, holdout MAE/RMSE/MAPE, and a
distinct forecast bundle.

**Go deeper:** [Forecasting deep](forecasting-deep.md) ·
[Leakage](leakage-cv-recipes.md) ·
[Artifacts](artifacts-checkpoints-bundles.md).

---

## First loop: time_split → lag_ridge → evaluate → bundle

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

fit = session.fit_forecast(
    method="lag_ridge",
    horizon=7,
    lags=[1, 2, 3, 7],
    alpha=1.0,
)
fit.show()

val = session.evaluate_forecast(partition="validation", strategy="rolling_one_step")
test = session.evaluate_forecast(partition="test", strategy="rolling_one_step")
print(val.metrics, test.metrics)

gen = session.generate_forecast(horizon=7)
print(gen.predictions)

bundle = session.save_forecast_bundle(".buildml-artifacts/forecast_bundle")
print(bundle)
```

`fit_forecast` **refuses** `session.split(...)` (random/stratified) and group
splits — use `time_split` (or a chronologically ordered `inject_split`).

---

## Baselines before claiming lag-model value

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

- Not a full econometrics suite (no ARIMA product surface, no cointegration lab).
- Univariate by default; optional numeric `exog_columns` require future exog
  for `generate_forecast`.
- MAPE is unstable near zero — prefer MAE/RMSE as primary.
- Not a digital twin and not a Torch sequence forecaster.

**Next:** [Forecasting deep](forecasting-deep.md).
