# Time-series analysis quickstart

```bash
pip install buildml
# STL / ADF / changepoints: pip install "buildml[timeseries]"
```

A `time` role, a target, and `time_split` are required, same refuse as
forecast. Default scope is train. No forecast model is fitted here.
That is `session.forecast`.

[TS analysis deep](timeseries-analysis-deep.md) ·
Paste: [`examples/timeseries_analyze_loop.py`](../examples/timeseries_analyze_loop.py) ·
[Forecasting](quickstart-forecasting.md)

---

## First loop: time_split → analyze → decompose → diagnostics

```python
import numpy as np
import pandas as pd

from buildml import Session

rng = np.random.default_rng(0)
n = 150
t = pd.date_range("2024-01-01", periods=n, freq="D")
y = 12 + 0.03 * np.arange(n) + 2 * np.sin(2 * np.pi * np.arange(n) / 7)
y += rng.normal(0, 0.25, n)
frame = pd.DataFrame({"ts": t, "y": y})

session = (
    Session.ingest(frame)
    .set_roles({"ts": "time", "y": "target"})
    .time_split(test_size=0.2, validation_size=0.2)
)

report = session.timeseries.analyze(scope="train", seasonal_period=7)
report.show()

# Focused calls
session.timeseries.decompose(decompose_method="stl", seasonal_period=7)
session.timeseries.diagnostics(acf_lags=30)
```

`session.timeseries.analyze` **refuses** `session.split(...)` random/stratified splits.

---

## Honesty bounds

- Descriptive EDA only: not forecast fit (`session.forecast.fit`) and not a digital twin.
- `scope='all'` includes holdout rows; use for exploration, not silent tuning.
- ADF/KPSS require `buildml[timeseries]`; core fallback exposes ACF/PACF only.
- Changepoints: PELT/binseg via ruptures when installed; CUSUM fallback otherwise.

---

## Typical next step

After diagnostics, run forecasting with industry defaults:

```python
session.forecast.fit(method="auto", horizon=7)
session.forecast.evaluate(partition="test", strategy="rolling_origin")
```

See [quickstart-forecasting.md](quickstart-forecasting.md).
