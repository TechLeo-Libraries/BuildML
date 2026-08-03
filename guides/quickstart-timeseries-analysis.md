# Time-series analysis quickstart

> **Install:** Core analysis uses moving-average / numpy fallbacks. For industry
> defaults (STL, ADF/KPSS, ruptures changepoints):
> `pip install "buildml[timeseries]"` on your Session 2.x checkout.

Descriptive time-series analysis on the same `Session`: `time` role +
`time_split`, train-only scope by default, decomposition, diagnostics,
changepoints, and spectral features: distinct from `fit_forecast`.

**Go deeper:** [Time-series analysis deep](timeseries-analysis-deep.md) ·

**Proof:** [store-sales-forecast](../proofs/store-sales-forecast/) (train-scoped analysis) · [harbor-demand-desk](../proofs/harbor-demand-desk/).
[Forecasting quickstart](quickstart-forecasting.md).

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

report = session.analyze_timeseries(scope="train", seasonal_period=7)
report.show()

# Focused calls
session.ts_decompose(decompose_method="stl", seasonal_period=7)
session.ts_diagnostics(acf_lags=30)
```

`analyze_timeseries` **refuses** `session.split(...)` random/stratified splits.

---

## Honesty bounds

- Descriptive EDA only: not forecast fit (`fit_forecast`) and not a digital twin.
- `scope='all'` includes holdout rows; use for exploration, not silent tuning.
- ADF/KPSS require `buildml[timeseries]`; core fallback exposes ACF/PACF only.
- Changepoints: PELT/binseg via ruptures when installed; CUSUM fallback otherwise.

---

## Typical next step

After diagnostics, run forecasting with industry defaults:

```python
session.fit_forecast(method="auto", horizon=7)
session.evaluate_forecast(partition="test", strategy="rolling_origin")
```

See [quickstart-forecasting.md](quickstart-forecasting.md).
