# loan-approval-classical

You have a credit table and an approve / decline label. You want a holdout
number that did not help fit the scaler, the encoder, or the threshold.

## Data

Prefers OpenML German Credit (`credit-g`) via
`load_classical_credit_table`. If OpenML is unavailable, the in-repo
credit draw is used and `data.loader_selected` records the fallback.
`sex_standin` is ignored (not a predictor). Not a regulated bureau extract.

Provenance is written under `results/results.json` -> `data`.

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

OpenML cache or network on first fetch; otherwise a disclosed synthetic
fallback. No fairness audit; single seed; not a deployment certification.

Related: [Classical quickstart](../../guides/quickstart-classical.md),
[examples/classical_loan_loop.py](../../examples/classical_loan_loop.py).
