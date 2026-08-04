# Time-series analysis domain floor

This proof documents the **analysis-only** floor for `buildml.timeseries`
(decompose, diagnostics, changepoints). It is intentionally thin: the domain
does not ship a fitted-plan checkpoint bundle: forecasting persistence lives
under `buildml.forecasting`.

## Floor checklist

| Artifact | Status |
| --- | --- |
| `catalog.py` + `session.timeseries.capability_matrix` | required |
| Session mixin matrix | required |
| `explain_hooks.py` | required |
| Guide (`guides/quickstart-timeseries-analysis.md`) | required |
| Unit / teaching tests | required |
| `checkpoint.py` | **not required** (`analysis_only`) |

## Smoke

```bash
python benchmarks/timeseries/analysis_smoke.py
pytest tests/unit/test_timeseries_r3_depth.py -q
```

See also `proofs/store-sales-forecast/` for the forecasting product surface.
