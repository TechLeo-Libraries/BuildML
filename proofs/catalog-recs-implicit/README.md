# catalog-recs-implicit

## Business purpose

Recommend catalog SKUs from user–item interactions using ALS (when `implicit` is installed) or item-kNN collaborative filtering.

## Data source

In-repo synthetic catalog interactions (`load_catalog_interactions_synthetic`): license-clear, deterministic. **Not** a real retail extract.

## Leakage controls

- Split before recommender fit
- Train-only recommender fit
- Test metrics after lock
- Industry item-cosine twin uses the same SplitPlan

## BuildML API steps

1. `Session.ingest` → `set_roles` → `split`
2. `session.recommender.fit(method="als"|"item_knn")`
3. `session.recommender.recommend(test)` → `session.recommender.evaluate(test, k=5)`
4. `session.recommender.save_bundle`

## Metrics

Primary holdout: hit-rate@k / nDCG@k (see `results/results.json`).

## Industry comparison (Tier C)

Industry twin: item-cosine + popularity cold-start twin via `baseline_industry.py` → `results/comparison.json`.

## Limitations

- Synthetic interactions
- ALS requires the `implicit` extra; otherwise item_knn fallback

## How to run

```bash
python proofs/catalog-recs-implicit/script.py
python proofs/catalog-recs-implicit/baseline_industry.py
```
