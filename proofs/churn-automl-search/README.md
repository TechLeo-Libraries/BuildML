# churn-automl-search

You have telco-style customer features and a churn label. You want a family
and recipe search under a disclosed trial budget, then one test number after
the winner is refit.

## Data

Synthetic telco churn (`load_telco_churn_synthetic`): license-clear stand-in
for IBM Telco-style schemas.

## Leakage

Stratified train / validation / test before search.
`session.automl.run(..., selection="cv")` ranks on train folds only. Session
test never enters family or recipe ranking.
`session.automl.evaluate(partition="test")` runs once after refit.

## How to run

```bash
python proofs/churn-automl-search/script.py
python proofs/churn-automl-search/baseline_industry.py
```

## What you'll get

`results/results.json` with classification metrics on validation and test,
plus a search summary. `results/comparison.json` is sklearn
`RandomizedSearchCV` over logistic / RF / GBM on the same stratified split.
The Session path uses FLAML or AutoGluon when installed, else native plus
LightGBM / XGBoost families.

## Limitations

Finite budget; synthetic labels; not a full CRM feature store.

Related: [AutoML quickstart](../../guides/quickstart-automl.md),
[examples/automl_search_loop.py](../../examples/automl_search_loop.py).
