# Forecasting deep

> **Install (GitHub 2.x):**
> `pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"`
> Core lag/baseline always available. Industry defaults:
> `pip install "buildml[timeseries]"` (statsmodels ETS/ARIMA/SARIMAX).
> Prophet: `buildml[timeseries-prophet]`. N-BEATS: `buildml[timeseries-ml]`.
> See [installation](../docs/installation.rst).

Depth guide for BuildML's classical forecasting Session path: temporal roles,
leakage discipline, lag/baseline models, generate vs evaluate protocols,
exogenous support, bundles, and honesty bounds.

**Related:** [Forecasting quickstart](quickstart-forecasting.md) ·
[Leakage](leakage-cv-recipes.md) ·
[Artifacts](artifacts-checkpoints-bundles.md) ·
[AutoML deep](automl-deep.md).

---

## What this path is (and is not)

| Is | Is not |
| --- | --- |
| Session API with history / explain / walkthrough | A toy stub without metrics or bundles |
| Train-only fit + holdout eval | Fit-on-full-frame "evaluation" |
| Lag tabularization + strong baselines | Full econometrics lab / digital twin |
| Univariate default + optional numeric exog | Silent multivariate identification claims |
| `buildml.forecast_bundle.v2` (v1 loadable) | Session checkpoint substitute |
| statsmodels ETS/ARIMA when `[timeseries]` | Torch sequence forecaster in this package |

Related: [unsupervised](unsupervised-deep.md), [ensembles](ensemble-deep.md),
[AutoML](automl-deep.md), [anomaly](anomaly-deep.md), and semi/self-supervised
guides. Explicit non-goals (neuromorphic, swarm, digital twins, AV/robotics,
TTS, multi-agent sims, full COCO detection) stay out.

---

## Conventions: time index, target, horizon

1. Assign exactly one **`time`** role and one **`target`** role.
2. Call **`time_split`** so train ends before validation/test in clock time.
3. **`horizon`** is the default H-step length stored on `ForecastPlan`
   (`session.forecast.generate` may override).
4. **`lags`** are positive integers; row *t* uses only `y[t-lag]`.

```python
session = (
    Session.ingest(frame)
    .set_roles({"ts": "time", "y": "target"})
    .time_split(test_size=0.2, validation_size=0.15)
)
```

---

## Leakage discipline

| Rule | Behavior |
| --- | --- |
| Forbidden splits | `random`, `stratified`, `group` → hard `LeakageError` |
| Allowed splits | `time` (preferred), `injected` if chronologically ordered |
| Partition order | Fit checks train end ≤ holdout start by time column |
| Lag features | Past targets only; early rows lacking full lag history are dropped from fit |
| Rolling eval | Appends holdout **actuals** after each one-step prediction |
| Origin eval | Fixed recursive multi-step from prior partition end |

```python
# This raises LeakageError:
bad = Session.ingest(frame).set_roles({"ts": "time", "y": "target"}).split(test_size=0.2)
bad.forecast.fit(method="naive")  # refused
```

---

## Methods

| Method | Role | Extra |
| --- | --- | --- |
| `auto` | ETS if statsmodels else `lag_ridge` | timeseries |
| `naive` | Last train value |: |
| `mean` | Train mean |: |
| `drift` | Linear extrapolation from first→last train point |: |
| `seasonal_naive` | Repeat last `seasonal_period` |: |
| `lag_ridge` | Ridge on lag (+ optional exog) features |: |
| `lag_hgb` | HistGradientBoosting on lag/exog features |: |
| `ets` | Holt-Winters exponential smoothing | timeseries |
| `arima` / `auto_arima` | ARIMA (auto = lightweight AIC grid) | timeseries |
| `sarimax` | Seasonal ARIMAX with optional exog | timeseries |
| `prophet` | Facebook Prophet | timeseries-prophet |
| `nbeats` | N-BEATS via neuralforecast | timeseries-ml |

Prefer baselines before claiming lag-model value on the **same** split and
eval strategy.

---

## Generate vs evaluate

```python
session.forecast.fit(method="lag_ridge", horizon=14, lags=[1, 2, 7, 14])

# Operational path: recursive H-step from train end
gen = session.forecast.generate(horizon=14, origin="train_end")

# Holdout skill: expanding one-step (default)
roll = session.forecast.evaluate(partition="test", strategy="rolling_one_step")

# Harder multi-step protocol
origin = session.forecast.evaluate(partition="test", strategy="origin")

# Rolling-origin backtest (M4-style windows)
rolling_origin = session.forecast.evaluate(partition="test", strategy="rolling_origin")
print(roll.metrics, origin.metrics, rolling_origin.metrics)
```

Metrics: **MAE**, **RMSE**, **MAPE**. MAPE may be NaN near zero actuals :
disclosed; lead with MAE/RMSE.

---

## Univariate vs exogenous

```python
# Univariate (default)
session.forecast.fit(method="lag_ridge", lags=[1, 2, 3])

# Light exogenous: numeric columns known at prediction time
session.forecast.fit(
    method="lag_ridge",
    lags=[1, 2, 3],
    exog_columns=["promo"],
)
# session.forecast.generate requires future_exog with shape (horizon, n_exog)
import numpy as np
future = np.zeros((7, 1))
session.forecast.generate(horizon=7, future_exog=future)
```

BuildML does **not** invent future exogenous drivers. Offline
`session.forecast.evaluate` may use holdout exog at each scored timestamp with
disclosure.

---

## Bundles

```python
path = session.forecast.save_bundle("artifacts/forecast_bundle")
restored = Session.ingest(frame).set_roles({"ts": "time", "y": "target"})
restored.time_split(test_size=0.2, validation_size=0.15)
restored.forecast.load_bundle(path)
print(restored.forecast.generate(horizon=7).predictions)
```

Format: `buildml.forecast_bundle.v2` (`meta.json` + `session.forecast.plan.joblib`).
v1 bundles remain loadable.
Not interchangeable with Session checkpoints, classical pipelines, Torch, RAG,
unsupervised, ensemble, or AutoML bundles. See
[Artifacts](artifacts-checkpoints-bundles.md).

---

## Teaching surfaces

- `session.explain("session.forecast.fit", moment="before")`
- Catalog concepts: `forecast-temporal-leakage`, `forecast-lag-features`,
  `forecast-univariate-vs-exog`, `forecast-eval-protocols`,
  `forecast-metric-limits`, `forecast-bundle-boundary`
- Walkthrough includes `forecasting_status`
- AI tools allowlist: `session.forecast.fit`, `session.forecast.generate`, `session.forecast.evaluate`,
  bundle save/load

---

## Failure modes

| Symptom | Likely cause |
| --- | --- |
| `LeakageError` on fit | Used `split` / `group_split` instead of `time_split` |
| Unparseable timestamps | Time column not datetime-parseable |
| Need more than max(lags) rows | Series too short for lag matrix |
| `future_exog` required | Plan was fit with `exog_columns` |
| MAPE is NaN | Near-zero actuals: use MAE/RMSE |

---

## Intentional limits

- No statsmodels/Prophet/NeuralForecast product surface (core stays light).
- No Torch sequence forecaster in this package (prefer a complete classical
  Session over a shallow DL stub).
- No causal identification APIs; EDA remains associational.
- Anomaly/fraud lives in a separate guide: not this surface.
