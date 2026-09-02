# aurora-ad-ranker

This script composes `session.ranking`, classical `session.fit`, and
`session.decision` on one synthetic sponsored-ad table. It is not a product BuildML ships.

The script fits LambdaRank / pointwise LTR with `group_split` by
`query_id`, trains a logistic CTR scorer on query x ad judgment pairs, and
selects threshold / knapsack impression capacity on validation only.

## Data

Synthetic graded ad judgments. Not a real auction log. CTR proxy is derived
from query x ad judgment pairs.

## Leakage

LTR `group_split` by `query_id` before ranker fit. Classical CTR split is
stratified and disjoint from test. Impression capacity / knapsack tuned on
validation only. Test nDCG and decision eval after each stage locks.

## What fails if leakage is ignored

Fitting the ranker on test queries overstates NDCG. Allocating impressions
on test invents CTR lift. Tuning serve thresholds on test understates
opportunity cost.

## How to run

```bash
python proofs/aurora-ad-ranker/script.py
```

## What you'll get

`results/` summary and per-stage JSON.

## Upstream

`sponsored-ad-ltr`, `search-relevance-ltr`, `loan-approval-classical`,
`campaign-budget-optimize`, `cost-sensitive-collections`.

## Limitations

Synthetic graded ad judgments. CTR proxy is derived from the same
judgments.
