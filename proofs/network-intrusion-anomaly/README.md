# network-intrusion-anomaly

You have traffic features and a rare attack class. You want unsupervised
anomaly scores with a validation-tuned alert threshold, then one labeled
holdout check.

## Data

Synthetic KDD-inspired flow table (`load_intrusion_anomaly_synthetic`):
license-clear. Not the full KDD Cup 1999 corpus.

## Leakage

Stratified train / validation / test (rare attack class preserved). The
unsupervised detector fits on train features only.
`session.anomaly.tune_threshold` uses validation labels only
(`allow_test_tuning=False`). Test is scored and evaluated after the
threshold is locked.

## How to run

```bash
python proofs/network-intrusion-anomaly/script.py
python proofs/network-intrusion-anomaly/baseline_industry.py
```

## What you'll get

`results/results.json` with labeled precision / recall / F1 on test, alert
rate, and the tuned threshold. `results/comparison.json` is sklearn
`IsolationForest` on the same `SplitPlan`, with the decision threshold tuned
on validation F1. The Session path typically uses PyOD HBOS when installed.
Deltas are descriptive on one synthetic draw.

## Limitations

Synthetic attacks; labeled eval overstates production unlabeled deployment.

Related: [Anomaly quickstart](../../guides/quickstart-anomaly.md),
[examples/anomaly_iforest_loop.py](../../examples/anomaly_iforest_loop.py).
