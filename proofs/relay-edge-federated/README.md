# relay-edge-federated

This script composes `session.federated`, `session.probabilistic`, and
classical `session.fit` on one synthetic edge-device table. It is not a
product BuildML ships.

Site shifts make the problem non-IID. The script runs FedAvg with
`group_split` by site (held-out sites never train), Bayesian-ridge plus
conformal intervals on a continuous risk proxy, and discloses a pooled
classical logistic baseline on the same split.

## Data

Synthetic edge sensors.

## Leakage

`group_split` by site so held-out edges never train FedAvg clients.
Probabilistic fit uses the same `inject_split` indices. Classical pooled
baseline is a disclosure contrast on the same split. Test evaluate after
locks.

## What fails if leakage is ignored

Including test sites as FL clients invents cross-silo generalization.
Fitting probabilistic intervals on the full fleet hides miscalibration.
Pooling then splitting after feature stats overstates classical ROC.

## How to run

```bash
python proofs/relay-edge-federated/script.py
```

## What you'll get

`results/` summary and per-stage JSON.

## Upstream

`edge-fleet-federated`, `federated-hospital-sim`, `prob-interval-risk`,
`weather-prob-intervals`, `loan-approval-classical`.

## Limitations

Local FedAvg simulation: not a deployed cross-silo network. Synthetic edge
sensors only.
