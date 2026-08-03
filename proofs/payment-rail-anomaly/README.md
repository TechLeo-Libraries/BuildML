# payment-rail-anomaly

## Business purpose

Detect anomalous payment-rail authorizations (ACH / card style) with unsupervised scoring and a validation-tuned alert threshold.

## Data source

In-repo synthetic payment table (`load_payment_rail_anomaly_synthetic`): license-clear, deterministic. **Not** a card-network extract.

## Leakage controls

- Stratified train / validation / test before scale / fit
- Unsupervised anomaly fit on train features only
- Threshold tuned on validation labels only
- Test scored / evaluated after threshold lock
- Industry IsolationForest twin uses the same SplitPlan

## BuildML API steps

1. `Session.ingest` → `set_roles` → `split` → `scale`
2. `fit_anomaly` (PyOD HBOS when available, else IsolationForest)
3. `tune_anomaly_threshold(validation)` → `score_anomalies(test)` → `evaluate_anomaly(test)`
4. `save_anomaly_bundle`

## Metrics

Primary labeled holdout: ROC-AUC, average precision, F1, precision, recall.

## Industry comparison (Tier C)

Filled: sklearn `IsolationForest` twin via `baseline_industry.py` → `results/comparison.json`.

## Limitations

- Synthetic attacks; production often unlabeled
- Single seed; not a PCI / fraud certification

## How to run

```bash
python proofs/payment-rail-anomaly/script.py
python proofs/payment-rail-anomaly/baseline_industry.py
```
