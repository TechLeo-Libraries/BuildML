# harbor-demand-desk

This script composes `session.timeseries`, `session.forecast`,
`session.probabilistic`, and `session.decision` on one synthetic store
series. It is not a product BuildML ships.

The script runs train-scoped TS analysis, a lag forecast, residual
intervals, and knapsack allocation over SKU-like candidates derived from
the frozen forecast.

## Data

Single synthetic store series. Not a retail extract.

## Leakage

`time_split` chronological train -> validation -> test.
`session.timeseries.analyze(scope="train")` only. Forecast fit on train;
selection metrics on validation. The probabilistic residual model uses its
own internal split. Allocation policy is selected on a disjoint validation
slice of future candidates.

## What fails if leakage is ignored

Shuffled date splits peek at future seasonality. STL/diagnostics on the
full series contaminate discovery with the test regime. Calibrating
intervals on test residuals reports perfect coverage by construction.
Choosing allocation with realized future demand is not a planning decision.

## How to run

```bash
python proofs/harbor-demand-desk/script.py
```

## What you'll get

`results/` summary plus `timeseries_analysis.json`, `forecast.json`,
`probabilistic.json`, and `allocation.json`.

## Upstream

`store-sales-forecast`, `prob-interval-risk`, `cost-sensitive-collections`.

## Limitations

Single synthetic store; knapsack allocation is not a full supply-chain MIP.
Missing `statsmodels` skips analysis with disclosed status.
