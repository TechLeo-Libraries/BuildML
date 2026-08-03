# Nexus Federated Clinical

**Tier B** cross-domain product proof: federated hospital simulation +
probabilistic uncertainty + honest evaluation disclosures.

## Product narrative

Nexus simulates multi-site clinical risk modeling without claiming a deployed
FL network or PHI:

1. `group_split` by hospital, then FedAvg local updates on train clients only
2. Bayesian Ridge (+ conformal intervals) for uncertainty on a risk-score proxy
3. Pooled centralized SGD contrast for disclosure (not used to tune FedAvg)
4. Explicit honesty: in-process aggregation, no secure aggregation, no PHI

## Status

`completed`: run `script.py`; see `results/summary.json` and stage JSONs.

## How to run

```bash
.\.venv\Scripts\python.exe proofs\nexus-federated-clinical\script.py
```

## Leakage controls

- `group_split` by hospital before any federated / probabilistic fit
- Federated local updates use train-client rows only
- Holdout hospitals/rows reserved for `evaluate_federated`
- Probabilistic model fit on train; intervals evaluated on test after lock

## Disclosures

- Local FedAvg simulation: raw rows stay in-process; not a deployed FL network
- Aggregation is weighted coefficient averaging: not cryptographic secure aggregation
- No PHI; synthetic labs with site shift only
- Probabilistic intervals are empirical coverage tools, not clinical guarantees

## Upstream Tier A building blocks

`federated-hospital-sim`, `prob-interval-risk`

## Limitations

Simulation honesty: not production cross-silo FL. Not a clinical decision
support device; no regulatory claim. Site shift is synthetic and mild.
