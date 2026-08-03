# Aurora Ad Ranker

**Tier B** cross-domain product proof: learning-to-rank + classical CTR proxy +
validation-tuned impression allocation for synthetic sponsored ads.

## Product narrative

Aurora ranks ads per query, scores a CTR proxy, and allocates scarce impressions:

1. Fits LambdaRank / pointwise LTR with `group_split` by `query_id`
2. Trains a classical logistic CTR scorer on query×ad judgment pairs
3. Selects threshold / knapsack capacity on validation only

## Status

`completed`: run `script.py`; see `results/summary.json` and stage JSONs.

## How to run

```bash
.\.venv\Scripts\python.exe proofs\aurora-ad-ranker\script.py
```

## Leakage controls

- LTR `group_split` by `query_id` before ranker fit
- Classical CTR split is stratified and disjoint from test
- Impression capacity / knapsack tuned on validation only
- Test nDCG and decision eval after each stage locks

## What fails if leakage is ignored

- Fitting the ranker on test queries overstates NDCG
- Allocating impressions on test invents CTR lift
- Tuning serve thresholds on test understates opportunity cost

## Upstream Tier A building blocks

`sponsored-ad-ltr`, `search-relevance-ltr`, `loan-approval-classical`,
`campaign-budget-optimize`, `cost-sensitive-collections`

## Limitations

Synthetic graded ad judgments: not a real auction log. CTR proxy is derived
from query×ad judgment pairs.
