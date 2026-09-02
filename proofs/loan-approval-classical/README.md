# loan-approval-classical

You have applicant age, income, debt ratio, employment tenure, region, and
product type. You want a holdout approve/decline number that did not help fit
the scaler, the encoder, or the threshold.

## Data

In-repo synthetic credit table (`load_credit_approval_synthetic`): license-clear,
deterministic, with MCAR-style missingness. Not a real FCRA / bureau extract.

## Leakage

Stratified train / validation / test before any fitting. `cv_score` with a
`PreprocessRecipe` runs on train folds only, on an unpoisoned Session.
Session-global impute, encode, scale, and outlier fences learn from train.
The decision threshold is tuned on validation only. Test is evaluated once
after selection. The sklearn twin uses the same `SplitPlan` indices.

## How to run

```bash
python proofs/loan-approval-classical/script.py
```

## What you'll get

`results/results.json` with holdout accuracy, F1, and ROC-AUC on test, plus
CV mean+/-std for selection disclosure. `results/comparison.json` is written
from `script.py` against a sklearn `ColumnTransformer` + `LogisticRegression`
twin on the same split.

## Limitations

Synthetic labels; no fairness audit; single seed; not a deployment
certification.

Related: [Classical quickstart](../../guides/quickstart-classical.md),
[examples/classical_loan_loop.py](../../examples/classical_loan_loop.py).
