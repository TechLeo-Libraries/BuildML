# Ballast Energy Desk

**Tier B** cross-domain product proof — chronological energy forecast +
probabilistic intervals + optimize allocation for demand-response capacity.

## Product narrative

Ballast is an energy trading / ops desk for a synthetic hourly load series.
Lag forecasts set the horizon; conformal intervals quantify residual risk;
knapsack allocation picks generation / DR blocks. The platform:

1. Fits lag Ridge forecast under chronological `time_split`
2. Fits Bayesian-ridge + conformal intervals on train residual proxies
3. Allocates future blocks with validation-selected knapsack policy

## Status

`completed` — run `script.py`; see `results/summary.json` and stage JSONs.

## How to run

```bash
.\.venv\Scripts\python.exe proofs\ballast-energy-desk\script.py
```

## Leakage controls

- `time_split` chronological train → validation → test
- Forecast fit on train; selection metrics on validation
- Probabilistic residual model uses train-only history
- Allocation policy selected on validation half of future blocks

## What fails if leakage is ignored

- Random split on hours lets the model peek at future seasonality
- Calibrating intervals on test residuals reports perfect coverage
- Choosing allocation with realized future demand is not a desk decision

## Upstream Tier A building blocks

`energy-load-forecast`, `store-sales-forecast`, `weather-prob-intervals`,
`prob-interval-risk`, `campaign-budget-optimize`, `harbor-demand-desk`

## Limitations

Single synthetic load series. Knapsack ≠ full unit-commitment MIP.
