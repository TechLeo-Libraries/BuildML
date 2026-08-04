# Relay Edge Federated

**Tier B** cross-domain product proof: multi-site FedAvg + probabilistic
intervals + centralized classical baseline for synthetic edge device risk.

## Product narrative

Relay aggregates fault signals across edge sites without shipping raw rows.
Site shifts make the problem non-IID. The platform:

1. Runs FedAvg with `group_split` by site (held-out sites never train)
2. Fits Bayesian-ridge + conformal intervals on a continuous risk proxy
3. Discloses a pooled classical logistic baseline on the same split

## Status

Run `script.py`. Outputs land under `results/` (summary and stage JSON)..

## How to run

```bash
python proofs\relay-edge-federated\script.py
```

## Leakage controls

- `group_split` by site so held-out edges never train FedAvg clients
- Probabilistic fit uses the same `inject_split` indices
- Classical pooled baseline is a disclosure contrast on the same split
- Test evaluate after locks

## What fails if leakage is ignored

- Including test sites as FL clients invents cross-silo generalization
- Fitting probabilistic intervals on the full fleet hides miscalibration
- Pooling then splitting after feature stats overstates classical ROC

## Upstream Tier A building blocks

`edge-fleet-federated`, `federated-hospital-sim`, `prob-interval-risk`,
`weather-prob-intervals`, `loan-approval-classical`

## Limitations

Local FedAvg simulation: not a deployed cross-silo network. Synthetic edge
sensors only.
