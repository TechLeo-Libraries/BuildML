# prism-shape-monitor

This script composes `session.tda`, `session.anomaly`, and classical
`session.fit` on one synthetic process-cloud table. It is not a product BuildML ships.

The script optionally fits a TDA persistence-image head (skips if `ripser`
/ `persim` are missing), runs unsupervised anomaly with validation-only
threshold tuning, and fits a classical logistic pass/fail scorer on the
same split. Anomaly and supervised stages still run when TDA skips.

## Data

Synthetic process clouds.

## Leakage

Stratified split before TDA / anomaly / supervised fit. Scale + TDA fit on
train only. Anomaly threshold tuned on validation only. Test used once per
stage after lock.

## What fails if leakage is ignored

Fitting persistence images on the full cloud leaks holdout geometry. Tuning
anomaly thresholds on test inflates F1 for drift alerts. Supervised
pass/fail trained with test rows overstates SPC readiness.

## How to run

```bash
python proofs/prism-shape-monitor/script.py
```

## What you'll get

`results/` summary and per-stage JSON. TDA skip is disclosed when extras
are missing.

## Upstream

`process-tda-shape`, `credit-tda-shape`, `iot-sensor-anomaly`,
`network-intrusion-anomaly`, `loan-approval-classical`.

## Limitations

Synthetic process clouds. TDA extras optional; skips disclosed in JSON.
