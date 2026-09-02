# meridian-recs-commerce

This script composes `session.recommender`, `session.ranking`, classical
`session.fit`, and optional `session.decision` on one synthetic retail
catalog. It is not a product BuildML ships.

The script fits collaborative recommenders (ALS when `implicit` is present,
else item-kNN), a query-group LTR ranker (LambdaRank or pointwise Ridge),
a purchase-propensity logistic model, and optional cost-sensitive promo
thresholds / knapsack on validation.

## Data

Synthetic catalog, judgments, and propensity. Not a retail extract.

## Leakage

Interaction / group / stratified splits before any fit. Recommenders and
rankers fit on train only. Decision policies tuned on validation only.
Test is evaluated once per stage after that stage locks.

## What fails if leakage is ignored

Fitting ALS on full interactions leaks test preferences into embeddings.
Query-group leakage in LTR inflates nDCG on held-out queries. Tuning promo
thresholds on test understates campaign cost.

## How to run

```bash
python proofs/meridian-recs-commerce/script.py
```

## What you'll get

`results/` summary and per-stage JSON.

## Upstream

`catalog-recs-implicit`, `sponsored-ad-ltr`, `movie-recs-collaborative`,
`search-relevance-ltr`, `loan-approval-classical`,
`campaign-budget-optimize`.

## Limitations

Synthetic catalog / judgments / propensity. Missing extras skip with JSON
disclosures (`MissingExtraError`).
