# Kiln Process TDA

**Tier B** cross-domain product proof — TDA shape descriptors + unsupervised
clustering + anomaly detection for synthetic kiln process clouds.

## Product narrative

Kiln monitors manufacturing process clouds. In-spec vs drifted regimes differ in
topology and density. The desk:

1. Fits TDA persistence-image heads when `ripser`/`persim` extras are present
   (otherwise skips with JSON disclosure)
2. Clusters scaled process embeddings (k-means on PCA) with holdout external labels
3. Runs unsupervised anomaly detection with **validation-only** threshold tuning

## Status

`completed` / `partial` — run `script.py`; see `results/summary.json` and stage JSONs.

## How to run

```bash
.\.venv\Scripts\python.exe proofs\kiln-process-tda\script.py
```

## Leakage controls

- Stratified split before TDA / clusters / anomaly
- TDA + scale fit on train only; test `evaluate_tda` after lock
- Cluster fit on train; external labels only for holdout eval
- Anomaly threshold tuned on validation only

## What fails if leakage is ignored

- Fitting TDA descriptors on the full cloud invents shape separability
- Choosing k / thresholds on test invents cluster purity and F1
- Including test rows in anomaly fit understates drift rates

## Upstream Tier A building blocks

`process-tda-shape`, `credit-tda-shape`, `sku-embedding-clusters`,
`cluster-customer-segments`, `iot-sensor-anomaly`, `network-intrusion-anomaly`

## Limitations

Synthetic kiln clouds — not plant SPC charts. TDA stage skipped when
`ripser`/`persim` extras are missing.
