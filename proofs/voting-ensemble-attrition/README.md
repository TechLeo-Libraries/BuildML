# voting-ensemble-attrition

You have employee features and an attrition label. You want a soft-voting
ensemble (logistic + random forest) whose holdout number did not leak into
model selection.

## Data

In-repo synthetic attrition table (`load_attrition_tabular_synthetic`):
license-clear, deterministic. Not a real employee extract.

## Leakage

Stratified train / validation / test before encode, scale, or ensemble fit.
One-hot encode and scale learn from train only. Voting bases fit on train
only. Test `session.ensemble.evaluate` runs after lock. The sklearn
`VotingClassifier` twin uses the same `SplitPlan`.

## How to run

```bash
python proofs/voting-ensemble-attrition/script.py
python proofs/voting-ensemble-attrition/baseline_industry.py
```

## What you'll get

`results/results.json` with holdout accuracy, F1, and ROC-AUC on test.
`results/comparison.json` is a sklearn `VotingClassifier(soft)` twin on the
same split.

## Limitations

Synthetic HR labels; two-base vote only; single seed; no nested outer CV.

Related: [Ensemble quickstart](../../guides/quickstart-ensemble.md),
[examples/ensemble_vote_stack_loop.py](../../examples/ensemble_vote_stack_loop.py).
