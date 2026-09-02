# nova-torch-bench

This script composes `session.dl`, classical `session.fit`, and
`session.probabilistic` on one synthetic mortgage table. It is not a
product BuildML ships.

The script fits a short CPU torch MLP (skips if torch is unavailable), a
classical logistic baseline on the same `inject_split`, and probabilistic
intervals on a train-derived residual/rate view. Classical and
probabilistic stages still run when torch skips.

## Data

Synthetic mortgage labels.

## Leakage

Stratified split before impute/encode/loaders. Torch normalize stats from
the train loader only. Classical baseline uses the same `inject_split`.
Probabilistic intervals calibrated on a train-derived internal split.

## What fails if leakage is ignored

Torch normalize stats from the full table leak holdout scale.
Early-stopping on test epochs cherry-picks the MLP. Interval calibration
on outer test reports perfect coverage by construction.

## How to run

```bash
python proofs/nova-torch-bench/script.py
```

## What you'll get

`results/` summary and per-stage JSON. Torch skip is disclosed when the
import is unhealthy.

## Upstream

`torch-tabular-underwrite`, `mortgage-default-classical`,
`loan-approval-classical`, `weather-prob-intervals`, `prob-interval-risk`.

## Limitations

Synthetic mortgage labels; 3-epoch MLP smoke. Torch optional.
