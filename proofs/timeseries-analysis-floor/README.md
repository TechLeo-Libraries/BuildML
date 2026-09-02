# timeseries-analysis-floor

`session.timeseries` is analysis only: decompose, diagnostics, changepoints.
There is no fitted bundle. Analysis does not persist a plan you can reload
and score. Forecasting persistence lives on `session.forecast`.

## Data

This directory is a floor note, not a fitted proof on a table. The user path
is the [time-series analysis quickstart](../../guides/quickstart-timeseries-analysis.md).
The forecast proof that follows analysis is
[store-sales-forecast](../store-sales-forecast/).

## Leakage

Analysis must be scoped to train (`session.timeseries.analyze(scope="train")`).
Running STL or changepoints on the full series lets the test regime into
discovery.

## How to run

There is no `script.py` here that fits a bundle. For analysis then forecast:

```bash
python proofs/store-sales-forecast/script.py
python examples/timeseries_analyze_loop.py
```

## What you'll get

Nothing under this slug. Analysis output, when you run it, is diagnostics
from `session.timeseries.analyze`, not a pipeline bundle. That is why there
is no checkpoint: the domain does not ship a fitted plan.

## Limitations

Analysis-only has no bundle by design. Use `session.forecast` when you need
something you can save and score.

Related: [Time-series analysis quickstart](../../guides/quickstart-timeseries-analysis.md),
[store-sales-forecast](../store-sales-forecast/).
