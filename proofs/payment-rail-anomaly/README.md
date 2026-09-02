# payment-rail-anomaly

You have ACH / card-style authorization features. You want unsupervised
anomaly scores and a validation-tuned alert threshold, then one labeled
holdout check.

## Data

In-repo synthetic payment table (`load_payment_rail_anomaly_synthetic`):
license-clear, deterministic. Not a card-network extract.

## Leakage

Stratified train / validation / test before scale or fit. Unsupervised
anomaly fits on train features only. The threshold is tuned on validation
labels only. Test is scored and evaluated after the threshold locks. The
IsolationForest twin uses the same `SplitPlan`.

## How to run

```bash
python proofs/payment-rail-anomaly/script.py
python proofs/payment-rail-anomaly/baseline_industry.py
```

## What you'll get

`results/results.json` with labeled holdout ROC-AUC, average precision, F1,
precision, and recall. `results/comparison.json` is a sklearn
`IsolationForest` twin on the same split. The Session path uses PyOD HBOS
when available, else IsolationForest.

## Limitations

Synthetic attacks; production is often unlabeled; single seed; not a PCI /
fraud certification.

Related: [Anomaly quickstart](../../guides/quickstart-anomaly.md),
[examples/anomaly_iforest_loop.py](../../examples/anomaly_iforest_loop.py).
