# compass-catalog-recs

This script composes `session.recommender`, `session.graph`, and classical
`session.fit` on one synthetic catalog interaction table. It is not a
product BuildML ships.

The script fits ALS / item-kNN recommenders on a train/validation/test
interaction split, builds an item co-purchase graph with classical
inductive features, and trains a logistic repurchase scorer on the same
node split.

## Data

Synthetic catalog interactions. Not a real retail extract. The co-purchase
graph is derived from the same interactions table.

## Leakage

Recommender split before fit; train-only ALS / item_knn. Graph node split
before classical graph features. Classical repurchase scorer uses the same
node `inject_split`. Test recommend / `session.graph.evaluate` / evaluate
after locks.

## What fails if leakage is ignored

Fitting recommenders on test interactions invents recall@k. Graph features
conditioned on test labels overstate ring repurchase. Fitting classical
scores on the full catalog invents holdout ROC.

## How to run

```bash
python proofs/compass-catalog-recs/script.py
```

## What you'll get

`results/` summary and per-stage JSON.

## Upstream

`catalog-recs-implicit`, `movie-recs-collaborative`, `graph-fraud-rings`,
`peer-lending-graph`, `loan-approval-classical`.

## Limitations

Synthetic catalog interactions. Co-purchase graph is derived from the same
table.
