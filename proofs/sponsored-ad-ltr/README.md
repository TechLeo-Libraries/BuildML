# sponsored-ad-ltr

## Business purpose

Rank sponsored ads within a query using graded relevance judgments (LambdaRank when LightGBM is available, else pointwise Ridge).

## Data source

In-repo synthetic ad LTR judgments (`load_ad_ltr_judgments_synthetic`): license-clear, deterministic. **Not** a real auction log.

## Leakage controls

- `group_split` on `query_id` (no query leakage across partitions)
- Train-only ranker fit
- Test nDCG after lock
- Industry Ridge twin uses the same group split

## BuildML API steps

1. `Session.ingest` → `set_roles` → `group_split`
2. `fit_ranker(method="lambdarank"|"pointwise")`
3. `rank(test)` → `evaluate_ranker(test, k=5)`
4. `save_ranker_bundle`

## Metrics

Primary holdout: nDCG@k on test queries (see `results/results.json`).

## Industry comparison (Tier C)

Filled: sklearn pointwise Ridge LTR twin via `baseline_industry.py` → `results/comparison.json`.

## Limitations

- Synthetic graded judgments
- LambdaRank requires LightGBM; otherwise pointwise Ridge fallback

## How to run

```bash
python proofs/sponsored-ad-ltr/script.py
python proofs/sponsored-ad-ltr/baseline_industry.py
```
