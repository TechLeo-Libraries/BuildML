# stacking-credit-risk

You have mortgage application features and a default label. You want a stack
of logistic and random-forest bases with a logistic meta-learner, using
out-of-fold train predictions only.

## Data

In-repo synthetic mortgage table (`load_mortgage_default_synthetic`):
license-clear, deterministic. Not a real credit bureau extract.

## Leakage

Stratified train / validation / test before any fit. Impute, encode, and
scale run on train only. Stacking OOF meta features come from train CV folds
only (`cv=3`). Test `session.ensemble.evaluate` runs after lock. The sklearn
`StackingClassifier` twin uses the same `SplitPlan`.

## How to run

```bash
python proofs/stacking-credit-risk/script.py
python proofs/stacking-credit-risk/baseline_industry.py
```

## What you'll get

`results/results.json` with holdout accuracy, F1, and ROC-AUC on test.
`results/comparison.json` is a sklearn `StackingClassifier(cv=3)` twin on the
same split.

## Limitations

Synthetic default labels; two-base stack only; single seed; not a regulated
underwriting certification.

Related: [Ensemble quickstart](../../guides/quickstart-ensemble.md),
[examples/ensemble_vote_stack_loop.py](../../examples/ensemble_vote_stack_loop.py).
