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

Horizon, lags and an explicit seasonal period require positive integer values.
Boolean, fractional and string counts are refused. Target history and exogenous
values must be finite; future exogenous data must have one row per forecast step
and the columns named by the plan. DataFrame columns are selected in plan order.

## Data requirements

- `split` / `group_split` / stratified: `LeakageError`.
- Injected splits must keep train, validation and test in chronological order.
  Validation observations cannot occur after the test origin because test
  forecasting conditions on the preceding validation history.
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
| `naive` | Last observation at the selected origin | core |
| `mean` | Train mean | core |
| `drift` | Train-fitted slope projected from the selected origin | core |
| `seasonal_naive` | Repeat the last `seasonal_period` observations at the origin | core |
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

`origin="validation_end"` and `origin="test_end"` condition generation on
observations available through that partition. Naive and seasonal-naive use the
latest observation or season. Drift keeps its train-fitted slope and starts from
the latest observation; mean keeps the train mean. ARIMA/SARIMAX keep fitted
parameters and receive the matching historical exogenous rows. Saved forecast
bundles retain these rules when loaded into a Session with the matching data
and chronological split.

N-BEATS returns the first requested positions of each native forecast block.
When a request exceeds `nbeats_horizon`, subsequent blocks use the preceding
predictions as context. Rolling evaluation appends observed holdout values
after scoring each window; it keeps fitted network weights fixed. The default
one-step native horizon uses an identity stack because the native trend and
seasonal bases require more than one output position. Input size, native horizon
and training steps must be positive integers, and supplied history must contain
at least `nbeats_input_size` observations. Native predictions use a synthetic
daily clock. These checks establish alignment and replay, without establishing
forecast quality from a short training run.

Prophet advances its synthetic daily forecast clock with the selected origin.
`validation_end`, `test_end` and rolling evaluation therefore predict the dates
after the supplied history, rather than repeatedly forecasting from train end.
Its fitted trend and seasonal parameters stay fixed: later target values do
not refit Prophet. The synthetic daily alignment remains a limitation for
irregular clocks; this does not map forecasts to the original calendar.
Saved Prophet bundles use the same origin rule when replayed.
Use compatible dependency versions for replay: validation uses matching
Pandas producer/reader versions. A Prophet artifact written with Pandas 3
failed to unpickle under Pandas 2 because of its stored datetime array format.

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

Agent `generate_forecast` accepts `future_exog` as numeric rows in the fitted
`exog_columns` order, with one row for each forecast step. Supply values known
at the forecast origin or an explicitly assumed scenario; this API does not
establish that future covariates are available in deployment. Offline holdout
backtests can use contemporaneous holdout covariates, as disclosed in results.
