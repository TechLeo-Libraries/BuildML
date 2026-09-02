# volt-sensor-fusion

This script composes `session.anomaly`, optional `session.tda`, and
classical `session.fit` on one synthetic industrial-sensor table. It is not
a product BuildML ships.

The script runs unsupervised anomaly detection with validation-only
threshold tuning, fits TDA persistence-image heads when `ripser`/`persim`
are present (else skips), and trains a classical logistic fault scorer on
the same stratified split.

## Data

Synthetic industrial sensors. Not a real SCADA extract.

## Leakage

Stratified split before anomaly / TDA / classical. Anomaly threshold tuned
on validation only. TDA + scale fit on train only when extras present.
Classical scorer uses `inject_split`: test after lock.

## What fails if leakage is ignored

Tuning anomaly thresholds on test inflates F1. Fitting TDA descriptors on
the full fleet invents shape separability. Fitting classical scores on the
full table invents holdout ROC.

## How to run

```bash
python proofs/volt-sensor-fusion/script.py
```

## What you'll get

`results/` summary and per-stage JSON.

## Upstream

`iot-sensor-anomaly`, `network-intrusion-anomaly`, `process-tda-shape`,
`credit-tda-shape`, `loan-approval-classical`.

## Limitations

Synthetic industrial sensors. TDA skipped without `ripser`/`persim`.
