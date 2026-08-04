# loan-approval-classical

## Business purpose

Approve or decline consumer loan applications from applicant age, income,
debt ratio, employment tenure, region, and product type. The business needs a
leakage-safe classical baseline before AutoML or cost-sensitive policies.

## Data source

In-repo synthetic credit table (`load_credit_approval_synthetic`): license-clear,
deterministic, with MCAR-style missingness. **Not** a real FCRA / bureau extract.

## Leakage controls

- Stratified train / validation / test before any fitting
- `cv_score` with `PreprocessRecipe` on **train folds only** (unpoisoned Session)
- Session-global impute / encode / scale / outlier fences fit on train
- Decision threshold tuned on **validation** only
- Test evaluated **once** after selection
- Industry sklearn twin uses the **same** `SplitPlan` indices

## BuildML API steps

1. `Session.ingest` → `set_roles` → `split` (stratified)
2. `cv_score(..., preprocess=PreprocessRecipe(...))`
3. Re-inject same split → `impute` → `encode` → `scale` → `handle_outliers` → `fit`
4. `evaluate(validation)` → `tune_threshold(validation)` → `evaluate(test)`
5. `save_pipeline`

## Metrics

Primary holdout: accuracy, F1, ROC-AUC on test (see `results/results.json`).
CV mean±std reported for model selection disclosure.

## Industry comparison (Tier C)

Industry twin: sklearn `ColumnTransformer` + `LogisticRegression` twin on the same SplitPlan indices, written to `results/comparison.json` from `script.py`.

## Limitations

Synthetic labels; no fairness audit; single seed; not a deployment certification.
