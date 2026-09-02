# iot-sensor-anomaly

You have factory sensor channels (temperature, vibration, current, pressure,
RPM). You want unsupervised anomaly scores and a validation-tuned threshold
for fault flags.

## Data

In-repo synthetic industrial sensors (`load_iot_sensor_anomaly_synthetic`):
license-clear, deterministic. Not a real SCADA extract.

## Leakage

Stratified train / validation / test before scale or fit. Unsupervised
anomaly fits on train features only. The threshold is tuned on validation
labels only. Test is scored and evaluated after the threshold locks. The
IsolationForest twin uses the same `SplitPlan`.

## How to run

```bash
python proofs/iot-sensor-anomaly/script.py
python proofs/iot-sensor-anomaly/baseline_industry.py
```

## What you'll get

`results/results.json` with labeled holdout ROC-AUC, average precision, F1,
precision, and recall. `results/comparison.json` is a sklearn
`IsolationForest` twin on the same split. The Session path uses PyOD HBOS
when available, else IsolationForest.

## Limitations

Synthetic faults; production is often unlabeled; single seed; not an OT
safety certification.

Related: [Anomaly quickstart](../../guides/quickstart-anomaly.md),
[examples/anomaly_iforest_loop.py](../../examples/anomaly_iforest_loop.py).
