# citadel-ensemble-desk

This script composes `session.ensemble`, `session.anomaly`, and
`session.decision` on one synthetic attrition table. It is not a product BuildML ships.

The script fits soft voting and stacking ensembles on a stratified split,
runs unsupervised anomaly with validation-only threshold tuning, and
selects review threshold / knapsack policies on validation. Two-base
ensembles keep smoke latency down.

## Data

Synthetic attrition table.

## Leakage

Stratified split before encode/scale/ensemble fit. Stacking OOF meta
features from train CV folds only. Anomaly threshold + decisions tuned on
validation only. Test evaluate after each stage locks.

## What fails if leakage is ignored

Picking the voting/stacking winner with test scores is not a fair ensemble.
Anomaly thresholds on test inflate review F1. Review knapsack tuned on
test understates HR cost.

## How to run

```bash
python proofs/citadel-ensemble-desk/script.py
```

## What you'll get

`results/` summary and per-stage JSON.

## Upstream

`voting-ensemble-attrition`, `stacking-credit-risk`,
`blending-payment-risk`, `network-intrusion-anomaly`,
`cost-sensitive-collections`.

## Limitations

Synthetic attrition table. Two-base ensembles for smoke latency.
