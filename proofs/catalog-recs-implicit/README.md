# catalog-recs-implicit

You have user-item catalog interactions. You want SKU recommendations from
ALS (when `implicit` is installed) or item-kNN collaborative filtering, with
holdout hit-rate / nDCG.

## Data

In-repo synthetic catalog interactions (`load_catalog_interactions_synthetic`):
license-clear, deterministic. Not a real retail extract.

## Leakage

Split before recommender fit. Train-only recommender fit. Test metrics after
lock. The item-cosine twin uses the same `SplitPlan`.

## How to run

```bash
python proofs/catalog-recs-implicit/script.py
python proofs/catalog-recs-implicit/baseline_industry.py
```

## What you'll get

`results/results.json` with hit-rate@k / nDCG@k on test.
`results/comparison.json` is item-cosine plus a popularity cold-start twin.

## Limitations

Synthetic interactions. ALS requires the `implicit` extra; otherwise
item_knn fallback.

Related: [Recommenders quickstart](../../guides/quickstart-recommenders.md),
[examples/recommender_item_knn_loop.py](../../examples/recommender_item_knn_loop.py).
