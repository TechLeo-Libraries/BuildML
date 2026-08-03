# mortgage-default-classical

## Business purpose

Score mortgage applications for default risk from LTV, DTI, credit score, note rate, term, and property type before underwriting policies harden.

## Data source

In-repo synthetic mortgage table (`load_mortgage_default_synthetic`): license-clear, deterministic, with MCAR-style missingness on credit score. **Not** a real servicing / HMDA extract.

## Leakage controls

- Stratified train / validation / test before any fitting
- Impute / encode / scale fit on train only
- Decision threshold tuned on validation only
- Test evaluated once after selection
- Industry sklearn twin uses the same SplitPlan indices

## BuildML API steps

1. `Session.ingest` → `set_roles` → `split` (stratified)
2. `impute` → `encode` → `scale` → `fit(LogisticRegression)`
3. `evaluate(validation)` → `tune_threshold(validation)` → `evaluate(test)`
4. `save_pipeline`

## Metrics

Primary holdout: accuracy, F1, ROC-AUC on test (see `results/results.json`).

## Industry comparison (Tier C)

Filled: sklearn `ColumnTransformer` + `LogisticRegression` twin via `baseline_industry.py` → `results/comparison.json`.

## Limitations

- Synthetic labels; no fairness / disparate-impact audit
- Single seed; not a deployment certification

## How to run

```bash
python proofs/mortgage-default-classical/script.py
python proofs/mortgage-default-classical/baseline_industry.py
```
