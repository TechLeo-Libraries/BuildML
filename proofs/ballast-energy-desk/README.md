# ballast-energy-desk

This script composes `session.forecast`, `session.probabilistic`, and
`session.decision` on one synthetic hourly load series. It is not a product BuildML ships.

Lag forecasts set the horizon; conformal intervals quantify residual risk;
knapsack allocation picks generation / DR blocks on validation.

## Data

Single synthetic load series. Not an ISO extract.

## Leakage

`time_split` chronological train -> validation -> test. Forecast fit on
train; selection metrics on validation. Probabilistic residual model uses
train-only history. Allocation policy selected on the validation half of
future blocks.

## What fails if leakage is ignored

Random split on hours lets the model peek at future seasonality.
Calibrating intervals on test residuals reports perfect coverage. Choosing
allocation with realized future demand is not a desk decision.

## How to run

```bash
python proofs/ballast-energy-desk/script.py
```

## What you'll get

`results/` summary and per-stage JSON.

## Upstream

`energy-load-forecast`, `store-sales-forecast`, `weather-prob-intervals`,
`prob-interval-risk`, `campaign-budget-optimize`, `harbor-demand-desk`.

## Limitations

Single synthetic load series. Knapsack is not a full unit-commitment MIP.
