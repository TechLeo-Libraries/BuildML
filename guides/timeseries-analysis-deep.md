# Time-series analysis (deep)

```bash
pip install buildml
# STL / ADF / changepoints: pip install "buildml[timeseries]"
```

Descriptive analysis on the same Session as forecasting. You still need a
`time` role, a target, and `time_split`. Random, stratified, and group
splits are refused. Default scope is train. Nothing here fits a forecast
model. That is `session.forecast`.

Prophet and N-BEATS extras belong to forecasting, not this surface.

Short on-ramp: [time-series analysis quickstart](quickstart-timeseries-analysis.md)
· [Forecasting](quickstart-forecasting.md).

## What you get

`session.timeseries.analyze` is the full report. Toggles:
`include_decompose`, `include_diagnostics`, `include_changepoints`,
`include_features`. `scope='train'` is the default. `scope='all'`
includes holdout rows and discloses that.

Focused calls: `session.timeseries.decompose` and
`session.timeseries.diagnostics`.

Core (no extra): moving-average decompose, numpy ACF/PACF, CUSUM
changepoints, rolling stats. With `buildml[timeseries]`: STL, classical
decompose, ADF/KPSS, Welch spectrum, PELT/binseg.

STL is the decompose default when statsmodels is installed; otherwise a
moving-average fallback.

## Results

`TSAnalysisResult` can hold:

- `TSDecomposeResult`: trend / seasonal / residual plus timestamps
- `TSDiagnosticsResult`: ACF/PACF (and confidence intervals with statsmodels)
- `TSChangepointResult`: index boundaries
- `TSFeatureResult`: rolling mean/std, dominant spectral period

Use train-only analysis to choose `ets` vs `arima` vs lag models, then
`session.forecast.fit`. Do not treat a decompose on `scope='all'` as a
fit protocol.
