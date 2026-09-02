# beacon-label-factory

This script composes `session.ssl`, `session.semisupervised`, and
`session.active_learning` on one synthetic inspection-feature table. It is
not a product BuildML ships.

Most train labels are masked; holdouts keep full labels for evaluation. The
script fits masked-tabular SSL pretext, runs label propagation, and queries
a train-only margin-sampling loop with a simulated oracle.

## Data

Tabular inspection proxies. Not a plant annotation UI.

## Leakage

Stratified split before masking / pretext / AL. Label masking applied to
train indices only. Holdouts retain full labels solely for evaluation. AL
queries drawn from the train unlabeled pool only.

## What fails if leakage is ignored

Masking validation/test then recovering labels via the graph overstates SSL
gains. Allowing AL to query the test pool turns the budget curve into a
cheat sheet. Fitting SSL pretext on the full table leaks holdout geometry
into embeddings.

## How to run

```bash
python proofs/beacon-label-factory/script.py
```

## What you'll get

`results/` summary and per-stage JSON.

## Upstream

`radiology-semi-labels`, `semi-label-efficiency`, `active-labeling-budget`,
`defect-active-budget`, `tabular-ssl-probe`, `ssl-representation-probe`,
`atlas-label-studio`.

## Limitations

Simulated oracle; tabular inspection proxies. Missing extras skip with JSON
disclosures.
