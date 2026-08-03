# Compass Catalog Recs

**Tier B** cross-domain product proof: collaborative recommenders + item
co-purchase graph features + classical repurchase scoring.

## Product narrative

Compass personalizes a synthetic catalog, mines co-purchase structure, and
scores item repurchase propensity:

1. Fits ALS / item-kNN recommenders on a train/validation/test interaction split
2. Builds an item co-purchase graph and fits classical inductive graph features
3. Trains a classical logistic repurchase scorer on the same node split

## Status

`completed`: run `script.py`; see `results/summary.json` and stage JSONs.

## How to run

```bash
.\.venv\Scripts\python.exe proofs\compass-catalog-recs\script.py
```

## Leakage controls

- Recommender split before fit; train-only ALS / item_knn
- Graph node split before classical graph features
- Classical repurchase scorer uses the same node `inject_split`
- Test recommend / `evaluate_graph` / evaluate after locks

## What fails if leakage is ignored

- Fitting recommenders on test interactions invents recall@k
- Graph features conditioned on test labels overstate ring repurchase
- Fitting classical scores on the full catalog invents holdout ROC

## Upstream Tier A building blocks

`catalog-recs-implicit`, `movie-recs-collaborative`, `graph-fraud-rings`,
`peer-lending-graph`, `loan-approval-classical`

## Limitations

Synthetic catalog interactions: not a real retail extract. Co-purchase graph
is derived from the same interactions table.
