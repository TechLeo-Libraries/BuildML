# mortgage-default-classical

You have LTV, DTI, credit score, note rate, term, and property type. You want
a holdout default score before you harden an underwriting policy.

## Data

In-repo synthetic mortgage table (`load_mortgage_default_synthetic`):
license-clear, deterministic, with MCAR-style missingness on credit score.
Not a real servicing / HMDA extract.

## Leakage

Stratified train / validation / test before any fitting. Impute, encode, and
scale learn from train only. The decision threshold is tuned on validation
only. Test is evaluated once after selection. The sklearn twin uses the same
`SplitPlan` indices.

## How to run

```bash
python proofs/mortgage-default-classical/script.py
python proofs/mortgage-default-classical/baseline_industry.py
```

## What you'll get

`results/results.json` with holdout accuracy, F1, and ROC-AUC on test.
`results/comparison.json` is a sklearn `ColumnTransformer` +
`LogisticRegression` twin on the same split.

## Limitations

Synthetic labels; no fairness / disparate-impact audit; single seed; not a
deployment certification.

Related: [Classical quickstart](../../guides/quickstart-classical.md).
