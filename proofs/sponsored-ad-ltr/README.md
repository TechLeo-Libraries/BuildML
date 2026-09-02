# sponsored-ad-ltr

You have graded relevance judgments for ads within a query. You want a
ranker (LambdaRank when LightGBM is available, else pointwise Ridge) that
does not leak queries into the holdout.

## Data

In-repo synthetic ad LTR judgments (`load_ad_ltr_judgments_synthetic`):
license-clear, deterministic. Not a real auction log.

## Leakage

`group_split` on `query_id` (no query leakage across partitions). Train-only
ranker fit. Test nDCG after lock. The Ridge twin uses the same group split.

## How to run

```bash
python proofs/sponsored-ad-ltr/script.py
python proofs/sponsored-ad-ltr/baseline_industry.py
```

## What you'll get

`results/results.json` with nDCG@k on test queries.
`results/comparison.json` is a sklearn pointwise Ridge LTR twin on the same
split.

## Limitations

Synthetic graded judgments. LambdaRank requires LightGBM; otherwise
pointwise Ridge fallback.

Related: [Ranking quickstart](../../guides/quickstart-ranking.md),
[examples/ranking_pointwise_loop.py](../../examples/ranking_pointwise_loop.py).
