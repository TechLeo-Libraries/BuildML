# iot-sensor-anomaly

## Business purpose

Flag factory IoT sensor faults (temperature, vibration, current, pressure, RPM) with unsupervised anomaly scoring and a validation-tuned threshold.

## Data source

In-repo synthetic industrial sensors (`load_iot_sensor_anomaly_synthetic`): license-clear, deterministic. **Not** a real SCADA extract.

## Leakage controls

- Stratified train / validation / test before scale / fit
- Unsupervised anomaly fit on train features only
- Threshold tuned on validation labels only
- Test scored / evaluated after threshold lock
- Industry IsolationForest twin uses the same SplitPlan

## BuildML API steps

1. `Session.ingest` → `set_roles` → `split` → `scale`
2. `session.anomaly.fit` (PyOD HBOS when available, else IsolationForest)
3. `session.anomaly.tune_threshold(validation)` → `session.anomaly.evaluate(test)`
4. `session.anomaly.save_bundle`

## Metrics

Primary labeled holdout: ROC-AUC, average precision, F1, precision, recall.

## Industry comparison (Tier C)

Industry twin: sklearn `IsolationForest` twin via `baseline_industry.py` → `results/comparison.json`.

## Limitations

- Synthetic faults; production often unlabeled
- Single seed; not an OT safety certification

## How to run

```bash
python proofs/iot-sensor-anomaly/script.py
python proofs/iot-sensor-anomaly/baseline_industry.py
```
