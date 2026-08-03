# Volt Sensor Fusion

**Tier B** cross-domain product proof: unsupervised anomaly + optional TDA +
classical fault scoring for synthetic industrial sensors.

## Product narrative

Volt fuses factory sensor channels to flag faults. Density shifts and shape
changes both matter. The stack:

1. Runs unsupervised anomaly detection with **validation-only** threshold tuning
2. Fits TDA persistence-image heads when `ripser`/`persim` are present (else skips)
3. Trains a classical logistic fault scorer on the same stratified split

## Status

`completed` / `partial`: run `script.py`; see `results/summary.json` and stage JSONs.

## How to run

```bash
.\.venv\Scripts\python.exe proofs\volt-sensor-fusion\script.py
```

## Leakage controls

- Stratified split before anomaly / TDA / classical
- Anomaly threshold tuned on validation only
- TDA + scale fit on train only when extras present
- Classical scorer uses `inject_split`: test after lock

## What fails if leakage is ignored

- Tuning anomaly thresholds on test inflates F1
- Fitting TDA descriptors on the full fleet invents shape separability
- Fitting classical scores on the full table invents holdout ROC

## Upstream Tier A building blocks

`iot-sensor-anomaly`, `network-intrusion-anomaly`, `process-tda-shape`,
`credit-tda-shape`, `loan-approval-classical`

## Limitations

Synthetic industrial sensors: not a real SCADA extract. TDA skipped without
`ripser`/`persim`.
