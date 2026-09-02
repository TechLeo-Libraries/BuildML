# lattice-supply-graph

This script composes `session.graph`, `session.kg`, and classical
`session.fit` on one synthetic supplier network. It is not a product BuildML ships.

The script fits classical inductive graph features on a stratified node
split, TransE link prediction on warehouse-route-hub triples, and a
logistic late-delivery risk scorer on the same node split.

## Data

Synthetic supplier communities. Not a TMS extract.

## Leakage

Stratified node split before graph / supervised fit. Classical graph
features from the train graph view. KG triple split before TransE. Test
evaluate after each stage locks.

## What fails if leakage is ignored

Graph features conditioned on test labels overstate community risk.
Training TransE on all triples makes link metrics meaningless. Supervised
late-risk trained with test rows overstates TMS readiness.

## How to run

```bash
python proofs/lattice-supply-graph/script.py
```

## What you'll get

`results/` summary and per-stage JSON.

## Upstream

`peer-lending-graph`, `graph-fraud-rings`, `logistics-kg-linkpred`,
`kg-biomed-linkpred`, `loan-approval-classical`.

## Limitations

Synthetic supplier communities. Classical graph path is primary.
