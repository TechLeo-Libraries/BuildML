# Harbor Demand Desk

**Tier B** cross-domain product proof — train-scoped TS analysis + lag forecast
+ probabilistic residual intervals + knapsack allocation over forecast SKUs
candidates.

## Product narrative

Harbor is a demand / promo desk for a single synthetic store series. Planners
need honest chronological evaluation, residual uncertainty, and a budgeted
allocation of promo/inventory spend across a short horizon of SKU-like
candidates derived from the frozen forecast.

## Status

`completed` — run `script.py`; see `results/summary.json` plus
`timeseries_analysis.json`, `forecast.json`, `probabilistic.json`,
`allocation.json`.

## How to run

```bash
.\.venv\Scripts\python.exe proofs\harbor-demand-desk\script.py
```

## Leakage controls

- `time_split` chronological train → validation → test
- `analyze_timeseries(scope="train")` only
- Forecast fit on train; selection metrics on validation
- Probabilistic residual model uses its own internal split
- Allocation policy selected on a disjoint validation slice of future candidates

## What fails if leakage is ignored

- Shuffled date splits peek at future seasonality
- STL/diagnostics on the full series contaminate discovery with the test regime
- Calibrating intervals on test residuals reports perfect coverage by construction
- Choosing allocation with realized future demand is not a desk decision

## Upstream Tier A building blocks

`store-sales-forecast`, `prob-interval-risk`, `cost-sensitive-collections`

## Limitations

Single synthetic store; knapsack allocation is not a full supply-chain MIP.
Missing `statsmodels` skips analysis with disclosed status.
