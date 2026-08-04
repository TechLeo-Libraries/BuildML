# churn-automl-search

## Business purpose

Predict telco customer churn to prioritize retention offers. AutoML searches
model families and preprocess recipes under a disclosed trial budget.

## Data source

Synthetic telco churn (`load_telco_churn_synthetic`): license-clear stand-in
for IBM Telco-style schemas.

## Leakage controls

- Stratified train / validation / test before search
- `session.automl.run(..., selection="cv")` ranks on **train folds only**
- Session test never enters family/recipe ranking
- `session.automl.evaluate(partition="test")` once after refit

## BuildML API steps

1. `ingest` → `set_roles` → stratified `split`
2. `session.automl.run` (FLAML/AutoGluon when installed; else native + LightGBM/XGBoost families)
3. `session.automl.evaluate` on validation then test
4. `session.automl.save_bundle`

## Metrics

Classification metrics on validation/test; search summary in JSON.

## Industry comparison (Tier C)

Industry twin: `baseline_industry.py` runs sklearn `RandomizedSearchCV` over logistic / RF / GBM on the same stratified split (`results/comparison.json`).
## Limitations

Finite budget; synthetic labels; not a full CRM feature store.
