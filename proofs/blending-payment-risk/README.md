# blending-payment-risk

You have payment-rail authorization features and an attack label. You want a
holdout-blend ensemble (logistic + random forest bases, logistic meta-learner)
that never lets Session validation or test into the blend.

## Data

In-repo synthetic payment authorizations
(`load_payment_rail_anomaly_synthetic`): license-clear, deterministic. Not a
card-network extract.

## Leakage

Stratified outer train / validation / test before scale or blend. The blend
holdout is carved from train only (`holdout_fraction=0.2`). Session
validation and test never fit the meta-learner. Test
`session.ensemble.evaluate` runs after lock. The holdout-blend twin uses the
same `SplitPlan`.

## How to run

```bash
python proofs/blending-payment-risk/script.py
python proofs/blending-payment-risk/baseline_industry.py
```

## What you'll get

`results/results.json` with holdout accuracy, F1, and ROC-AUC on test.
`results/comparison.json` is a sklearn holdout-blend twin on the same split.

## Limitations

Synthetic payment labels; the supervised blend assumes labeled attacks;
single seed; not a fraud certification.

Related: [Ensemble quickstart](../../guides/quickstart-ensemble.md),
[examples/ensemble_vote_stack_loop.py](../../examples/ensemble_vote_stack_loop.py).
