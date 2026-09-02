# atlas-label-studio

This script composes `session.ssl`, `session.semisupervised`, and
`session.active_learning` on one synthetic table with scarce labels. It is
not a product BuildML ships.

Most train labels are masked; holdouts keep full labels for evaluation
only. The script fits a masked-tabular pretext, runs label propagation, and
queries a train-only active-learning loop with a simulated oracle
(ground-truth for queried train indices). That oracle is not a workforce
UI.

## Data

Synthetic blobs. Not a labeling product.

## Leakage

Stratified split before masking / pretext / AL. Label masking applied to
train indices only. Holdouts retain full labels solely for evaluation. AL
queries are drawn from the train unlabeled pool only.

## What fails if leakage is ignored

Masking validation/test then "recovering" labels via the graph overstates
SSL gains. Allowing AL to query the test pool turns the budget curve into a
cheat sheet. Fitting SSL pretext on the full table leaks holdout geometry
into embeddings.

## How to run

```bash
python proofs/atlas-label-studio/script.py
```

## What you'll get

`results/` summary plus `semisupervised.json` and `active_learning.json`.

## Upstream

`ssl-representation-probe`, `semi-label-efficiency`,
`active-labeling-budget`.

## Limitations

Simulated oracle; synthetic blobs; production label noise is not modeled.
Missing extras are skipped with JSON disclosures.
