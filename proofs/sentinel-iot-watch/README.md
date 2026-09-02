# sentinel-iot-watch

This script composes `session.anomaly`, `session.online`, and
`session.forecast` on synthetic factory telemetry plus a plant-load series.
It is not a product BuildML ships.

The script flags sensor faults with validation-only anomaly thresholds,
streams SGD `partial_fit` from the train cursor only, and forecasts plant
load with `time_split` plus lag Ridge.

## Data

Synthetic sensors and load. Not SCADA.

## Leakage

Stratified sensor split before anomaly / online fit. Anomaly threshold
tuned on validation only. Online `partial_fit` consumes the train cursor
only. Forecast uses chronological `time_split`.

## What fails if leakage is ignored

Tuning anomaly thresholds on test inflates fault F1. Streaming updates that
include test rows make online metrics meaningless. Random split on plant
load lets the forecaster peek at future seasonality.

## How to run

```bash
python proofs/sentinel-iot-watch/script.py
```

## What you'll get

`results/` summary and per-stage JSON.

## Upstream

`iot-sensor-anomaly`, `network-intrusion-anomaly`, `clickstream-online`,
`stream-fraud-online`, `energy-load-forecast`, `store-sales-forecast`.

## Limitations

Synthetic sensors / load. Batch online chunks, not Kafka/Flink.
