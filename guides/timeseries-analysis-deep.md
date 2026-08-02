# Time-series analysis (deep)

Phase R3 analysis plugin — distinct Session surface from forecasting.

## Architecture

```
Session.analyze_timeseries / ts_decompose / ts_diagnostics
        ↓
buildml.timeseries.analyze
        ↓
decompose · diagnostics · changepoints · features
```

Temporal guards reuse `buildml.forecasting.features` (`time_split` required;
random/stratified/group splits refused).

## Dependency policy

| Extra | Packages | Enables |
|-------|----------|---------|
| *(core)* | numpy/pandas/sklearn | moving-average decompose, numpy ACF/PACF, CUSUM changepoints, rolling stats |
| `buildml[timeseries]` | statsmodels, scipy, ruptures | STL, classical decompose, ADF/KPSS, Welch spectrum, PELT/binseg |
| `buildml[timeseries-prophet]` | prophet | *(forecasting only)* |
| `buildml[timeseries-ml]` | neuralforecast, torch | *(forecasting N-BEATS)* |

## APIs

### `analyze_timeseries`

Full report with toggles:

- `include_decompose`, `include_diagnostics`, `include_changepoints`, `include_features`
- `scope='train'` (default) or `'all'` (EDA — disclosed leakage risk)

### `ts_decompose`

STL (default when statsmodels installed), classical additive, or moving-average fallback.

### `ts_diagnostics`

ACF/PACF arrays (+ confidence intervals with statsmodels), ADF and KPSS p-values.

## Results

`TSAnalysisResult` holds optional sub-results:

- `TSDecomposeResult` — trend/seasonal/residual tuples + timestamps
- `TSDiagnosticsResult` — exportable ACF/PACF for plotting
- `TSChangepointResult` — index boundaries
- `TSFeatureResult` — rolling mean/std, dominant spectral period

## Walkthrough / AI

`session.walkthrough()` includes `timeseries_status`. AI allowlist:
`analyze_timeseries`, `ts_decompose`, `ts_diagnostics`.

## Benchmark

```bash
python benchmarks/timeseries/analysis_smoke.py
```

Writes `benchmarks/timeseries/results/analysis_smoke.json`.

## Relationship to forecasting

Analysis informs method choice (`ets` vs `arima` vs lag models) but does not
fit predictors. Use `fit_forecast(method='auto')` after train-only analysis.
