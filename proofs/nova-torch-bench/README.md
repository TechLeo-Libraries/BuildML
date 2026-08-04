# Nova Torch Bench

**Tier B** cross-domain product proof: torch tabular MLP + classical
supervised baseline + probabilistic intervals / calibration.

## Product narrative

Nova benches tabular underwriting models side by side. A short torch MLP,
a classical logistic baseline, and Bayesian-ridge intervals share an honest
mortgage split. The platform:

1. Fits a CPU torch MLP (skips if torch unavailable)
2. Trains a classical logistic baseline on the same inject_split
3. Fits probabilistic intervals on a train-derived residual/rate view

## Status

Run `script.py`. Outputs land under `results/` (summary and stage JSON)..
Core classical + probabilistic stages keep the product completed when torch skips.

## How to run

```bash
python proofs\nova-torch-bench\script.py
```

## Leakage controls

- Stratified split before impute/encode/loaders
- Torch normalize stats from train loader only
- Classical baseline uses the same inject_split
- Probabilistic intervals calibrated on train-derived internal split

## What fails if leakage is ignored

- Torch normalize stats from the full table leak holdout scale
- Early-stopping on test epochs cherry-picks the MLP
- Interval calibration on outer test reports perfect coverage by construction

## Upstream Tier A building blocks

`torch-tabular-underwrite`, `mortgage-default-classical`, `loan-approval-classical`,
`weather-prob-intervals`, `prob-interval-risk`

## Limitations

Synthetic mortgage labels; 3-epoch MLP smoke. Torch optional.
