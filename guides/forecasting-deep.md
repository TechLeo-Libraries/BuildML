# Forecasting (deep)

```bash
pip install buildml
# ETS / ARIMA: pip install "buildml[timeseries]"
# Prophet: pip install "buildml[timeseries-prophet]"
# N-BEATS: pip install "buildml[timeseries-ml]"
```

One `time` role, one target, and `time_split`. Random, stratified, and
group splits are `LeakageError`. Default `method="auto"` is ETS when
statsmodels is installed, otherwise `lag_ridge`. Fit is train-only.
`generate` forecasts the requested horizon from the selected origin.
`evaluate` compares forecasts with observed holdout values under a selected protocol.

For descriptive time-series analysis without fitting a forecast model,
use `session.timeseries`.

Quickstart: [forecasting quickstart](quickstart-forecasting.md).

## Setup

```python
import numpy as np
import pandas as pd
from buildml import Session

rng = np.random.default_rng(0)
n = 120
frame = pd.DataFrame({
    "ts": pd.date_range("2024-01-01", periods=n, freq="D"),
    "y": 10 + 0.05 * np.arange(n) + np.sin(np.arange(n) / 7) + rng.normal(0, 0.3, n),
    "promo": (np.arange(n) % 7 == 0).astype(float),
})
session = (Session.ingest(frame)
    .set_roles({"ts": "time", "y": "target", "promo": "ignore"})
    .time_split(test_size=0.2, validation_size=0.15))

```

`horizon` is the default H-step stored on `ForecastPlan`.
`session.forecast.generate` may override it. Lags are positive integers;
row *t* uses only `y[t-lag]`. Early rows that lack a full lag history
are dropped from fit.

## Data requirements

- `split` / `group_split` / stratified: `LeakageError`.
- Fit when train does not end before holdout on the time column.
- Missing future exogenous values. If you fit with `exog_columns`,
  `generate` needs `future_exog`. Offline `evaluate` may use holdout
  exog at each scored timestamp and discloses that.

```python
import numpy as np
import pandas as pd
from buildml import Session

rng = np.random.default_rng(0)
n = 120
frame = pd.DataFrame({
    "ts": pd.date_range("2024-01-01", periods=n, freq="D"),
    "y": 10 + 0.05 * np.arange(n) + np.sin(np.arange(n) / 7) + rng.normal(0, 0.3, n),
    "promo": (np.arange(n) % 7 == 0).astype(float),
})
session = (Session.ingest(frame)
    .set_roles({"ts": "time", "y": "target", "promo": "ignore"})
    .time_split(test_size=0.2, validation_size=0.15))

from buildml.core.errors import LeakageError

bad = Session.ingest(frame).set_roles({"ts": "time", "y": "target", "promo": "ignore"}).split(test_size=0.2)
try:
    bad.forecast.fit(method="naive")
except LeakageError as exc:
    print("Expected rejection of a random time-series split:", exc)
```

## Methods

| Method | Role | Extra |
| --- | --- | --- |
| `auto` | ETS if statsmodels else `lag_ridge` | timeseries |
| `naive` | Last train value | core |
| `mean` | Train mean | core |
| `drift` | Line from first to last train point | core |
| `seasonal_naive` | Repeat last `seasonal_period` | core |
| `lag_ridge` | Ridge on lag (+ optional exog) | core |
| `lag_hgb` | HistGradientBoosting on lag/exog | core |
| `ets` | Holt-Winters | timeseries |
| `arima` / `auto_arima` | ARIMA (auto = AIC grid) | timeseries |
| `sarimax` | Seasonal ARIMAX with optional exog | timeseries |
| `prophet` | Prophet | timeseries-prophet |
| `nbeats` | N-BEATS via neuralforecast | timeseries-ml |

Prefer a baseline on the **same** split and eval strategy before
claiming lag-model value.

## Generate vs evaluate

```python
import numpy as np
import pandas as pd
from buildml import Session

rng = np.random.default_rng(0)
n = 120
frame = pd.DataFrame({
    "ts": pd.date_range("2024-01-01", periods=n, freq="D"),
    "y": 10 + 0.05 * np.arange(n) + np.sin(np.arange(n) / 7) + rng.normal(0, 0.3, n),
    "promo": (np.arange(n) % 7 == 0).astype(float),
})
session = (Session.ingest(frame)
    .set_roles({"ts": "time", "y": "target", "promo": "ignore"})
    .time_split(test_size=0.2, validation_size=0.15))

session.forecast.fit(method="lag_ridge", horizon=14, lags=[1, 2, 7, 14])

gen = session.forecast.generate(horizon=14, origin="train_end")

roll = session.forecast.evaluate(partition="test", strategy="rolling_one_step")
origin = session.forecast.evaluate(partition="test", strategy="origin")
rolling_origin = session.forecast.evaluate(
    partition="test", strategy="rolling_origin"
)
print(roll.metrics, origin.metrics, rolling_origin.metrics)
```

Rolling eval appends holdout **actuals** after each one-step
prediction. Origin eval is recursive multi-step from the prior
partition end. Metrics: MAE, RMSE, MAPE. MAPE may be NaN near zero
actuals; lead with MAE/RMSE.

## Exogenous

```python
import numpy as np
import pandas as pd
from buildml import Session

rng = np.random.default_rng(0)
n = 120
frame = pd.DataFrame({
    "ts": pd.date_range("2024-01-01", periods=n, freq="D"),
    "y": 10 + 0.05 * np.arange(n) + np.sin(np.arange(n) / 7) + rng.normal(0, 0.3, n),
    "promo": (np.arange(n) % 7 == 0).astype(float),
})
session = (Session.ingest(frame)
    .set_roles({"ts": "time", "y": "target", "promo": "ignore"})
    .time_split(test_size=0.2, validation_size=0.15))

session.forecast.fit(
    method="lag_ridge",
    lags=[1, 2, 3],
    exog_columns=["promo"],
)
import numpy as np
future = np.zeros((7, 1))
session.forecast.generate(horizon=7, future_exog=future)
```

## Bundle

`buildml.forecast_bundle.v2` (`meta.json` + plan). v1 remains loadable.
Not a Session checkpoint.

```python
import numpy as np
import pandas as pd
from buildml import Session

rng = np.random.default_rng(0)
n = 120
frame = pd.DataFrame({
    "ts": pd.date_range("2024-01-01", periods=n, freq="D"),
    "y": 10 + 0.05 * np.arange(n) + np.sin(np.arange(n) / 7) + rng.normal(0, 0.3, n),
    "promo": (np.arange(n) % 7 == 0).astype(float),
})
session = (Session.ingest(frame)
    .set_roles({"ts": "time", "y": "target", "promo": "ignore"})
    .time_split(test_size=0.2, validation_size=0.15))

session.forecast.fit(method="lag_ridge", lags=[1, 2, 7])
path = session.forecast.save_bundle("artifacts/forecast_bundle")
restored = Session.ingest(frame).set_roles({"ts": "time", "y": "target"})
restored.time_split(test_size=0.2, validation_size=0.15)
restored.forecast.load_bundle(path, trusted=True)
print(restored.forecast.generate(horizon=7).predictions)
```

## What usually goes wrong

- Used `split` instead of `time_split`.
- Time column not datetime-parseable.
- Series shorter than `max(lags)`.
- Plan was fit with `exog_columns` and `future_exog` is missing.

[Time-series analysis](timeseries-analysis-deep.md) ·
[Leakage](leakage-cv-recipes.md)
