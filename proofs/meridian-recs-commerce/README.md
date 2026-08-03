# Meridian Recs Commerce

**Tier B** cross-domain product proof: collaborative recommenders +
learning-to-rank browse relevance + classical purchase propensity + optional
promo decision thresholds.

## Product narrative

Meridian is a commerce personalization desk for a synthetic retail catalog.
Shoppers interact with SKUs; browse queries need ranked ads/items; purchase
propensity drives promo spend. The platform:

1. Fits collaborative recommenders (ALS when `implicit` is present, else item-kNN)
2. Trains a query-group LTR ranker (LambdaRank or pointwise Ridge)
3. Scores purchase propensity with a stratified classical logistic model
4. Optionally selects cost-sensitive promo thresholds / knapsack on validation

## Status

`completed`: run `script.py`; see `results/summary.json` and stage JSONs.

## How to run

```bash
.\.venv\Scripts\python.exe proofs\meridian-recs-commerce\script.py
```

## Leakage controls

- Interaction / group / stratified splits before any fit
- Recommenders and rankers fit on train only
- Decision policies tuned on validation only
- Test evaluated once per stage after that stage locks

## What fails if leakage is ignored

- Fitting ALS on full interactions leaks test preferences into embeddings
- Query-group leakage in LTR inflates nDCG on held-out queries
- Tuning promo thresholds on test understates campaign cost

## Upstream Tier A building blocks

`catalog-recs-implicit`, `sponsored-ad-ltr`, `movie-recs-collaborative`,
`search-relevance-ltr`, `loan-approval-classical`, `campaign-budget-optimize`

## Limitations

Synthetic catalog / judgments / propensity. Missing extras skip with JSON
disclosures (`MissingExtraError`).
