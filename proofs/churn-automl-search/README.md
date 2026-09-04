# churn-automl-search

You want a family and recipe search under a disclosed trial budget, then
one test number after the winner is refit. The slug is historical; the
table is not a telco CRM extract.

## Data

**REAL_PUBLIC_DATASET** -- `sklearn.datasets.load_breast_cancer` (UCI Breast
Cancer Wisconsin Diagnostic, redistributed with sklearn). Offline; no
network. Same public table as
[breast-cancer-classical](../breast-cancer-classical/), different Session
surface (`session.automl`).

Provenance fields are written under `results/results.json` -> `data`.

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

Finite budget; not a CRM feature store. The script refuses a perfect
holdout score (`>= 1.0`).

Related: [AutoML quickstart](../../guides/quickstart-automl.md),
[examples/automl_search_loop.py](../../examples/automl_search_loop.py).
