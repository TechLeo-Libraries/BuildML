# Sentinel IoT Watch

**Tier B** cross-domain product proof — unsupervised anomaly + online
`partial_fit` streaming + lag forecast for factory IoT telemetry.

## Product narrative

Sentinel watches a synthetic plant. Sensor faults are flagged with anomaly
detection; an online classifier streams train-cursor updates; a separate
plant-load series is forecast chronologically. The platform:

1. Fits unsupervised anomaly with validation-only threshold tuning
2. Streams SGD `partial_fit` updates from the train cursor only
3. Forecasts plant load with `time_split` + lag Ridge

## Status

`completed` — run `script.py`; see `results/summary.json` and stage JSONs.

## How to run

```bash
.\.venv\Scripts\python.exe proofs\sentinel-iot-watch\script.py
```

## Leakage controls

- Stratified sensor split before anomaly / online fit
- Anomaly threshold tuned on validation only
- Online `partial_fit` consumes train cursor only
- Forecast uses chronological `time_split`

## What fails if leakage is ignored

- Tuning anomaly thresholds on test inflates fault F1
- Streaming updates that include test rows make online metrics meaningless
- Random split on plant load lets the forecaster peek at future seasonality

## Upstream Tier A building blocks

`iot-sensor-anomaly`, `network-intrusion-anomaly`, `clickstream-online`,
`stream-fraud-online`, `energy-load-forecast`, `store-sales-forecast`

## Limitations

Synthetic sensors / load. Batch online chunks, not Kafka/Flink.
