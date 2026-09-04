# mortgage-default-classical

You want a shorter classical holdout on a public credit-risk table, then a
separate sklearn twin file. The slug is historical; the table is not a
mortgage servicing extract.

## Data

Same loader as [loan-approval-classical](../loan-approval-classical/):
OpenML German Credit (`credit-g`) when cached, otherwise the in-repo
credit draw. `data.loader_selected` says which one ran.

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

Same public credit table as loan-approval-classical, different Session
spine (no fold-local CV recipe). No fairness audit; single seed; not HMDA.

Related: [Classical quickstart](../../guides/quickstart-classical.md).
