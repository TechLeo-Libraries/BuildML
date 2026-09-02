# nexus-federated-clinical

This script composes `session.federated` and `session.probabilistic` on one
synthetic multi-site clinical table. It is not a product BuildML ships.

The script runs `group_split` by hospital, FedAvg local updates on train
clients only, Bayesian Ridge plus conformal intervals on a risk-score
proxy, and a pooled centralized SGD contrast for disclosure (not used to
tune FedAvg).

## Data

Synthetic labs with site shift only. No PHI.

## Leakage

`group_split` by hospital before any federated / probabilistic fit.
Federated local updates use train-client rows only. Holdout
hospitals/rows are reserved for `session.federated.evaluate`. The
probabilistic model fits on train; intervals are evaluated on test after
lock.

## What fails if leakage is ignored

Including test sites as FL clients invents cross-silo generalization.
Fitting intervals on the full book hides miscalibration.

## How to run

```bash
python proofs/nexus-federated-clinical/script.py
```

## What you'll get

`results/` summary and per-stage JSON, including honesty fields: in-process
aggregation, no secure aggregation, no PHI. Local FedAvg simulation: raw
rows stay in-process; not a deployed FL network. Aggregation is weighted
coefficient averaging, not cryptographic secure aggregation. Probabilistic
intervals are empirical coverage tools, not clinical guarantees.

## Upstream

`federated-hospital-sim`, `prob-interval-risk`.

## Limitations

Simulation honesty: not production cross-silo FL. Not a clinical decision
support device; no regulatory claim. Site shift is synthetic and mild.
