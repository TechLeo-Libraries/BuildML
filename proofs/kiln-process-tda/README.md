# kiln-process-tda

This script composes `session.tda`, `session.unsupervised`, and
`session.anomaly` on one synthetic kiln process-cloud table. It is not a
product BuildML ships.

In-spec vs drifted regimes differ in topology and density. TDA
persistence-image heads run when `ripser`/`persim` extras are present;
otherwise that stage skips with JSON disclosure.

## Data

Synthetic kiln clouds. Not plant SPC charts.

## Leakage

Stratified split before TDA / clusters / anomaly. TDA + scale fit on train
only; test `session.tda.evaluate` after lock. Cluster fit on train;
external labels only for holdout eval. Anomaly threshold tuned on
validation only.

## What fails if leakage is ignored

Fitting TDA descriptors on the full cloud invents shape separability.
Choosing k / thresholds on test invents cluster purity and F1. Including
test rows in anomaly fit understates drift rates.

## How to run

```bash
python proofs/kiln-process-tda/script.py
```

## What you'll get

`results/` summary and per-stage JSON. TDA skip disclosed when extras are
missing.

## Upstream

`process-tda-shape`, `credit-tda-shape`, `sku-embedding-clusters`,
`cluster-customer-segments`, `iot-sensor-anomaly`,
`network-intrusion-anomaly`.

## Limitations

Synthetic kiln clouds. TDA stage skipped when `ripser`/`persim` extras are
missing.
