# Citadel Ensemble Desk

**Tier B** cross-domain product proof — voting/stacking ensembles + unsupervised
anomaly + decision thresholds for attrition review.

## Product narrative

Citadel is an HR risk review desk. Soft voting and stacking ensembles score
attrition; anomaly detection flags unusual employee profiles; cost-sensitive
policies allocate review capacity. The platform:

1. Fits soft voting and stacking ensembles on a stratified split
2. Runs unsupervised anomaly with validation-only threshold tuning
3. Selects review threshold / knapsack policies on validation

## Status

`completed` — run `script.py`; see `results/summary.json` and stage JSONs.

## How to run

```bash
.\.venv\Scripts\python.exe proofs\citadel-ensemble-desk\script.py
```

## Leakage controls

- Stratified split before encode/scale/ensemble fit
- Stacking OOF meta features from train CV folds only
- Anomaly threshold + decisions tuned on validation only
- Test evaluate after each stage locks

## What fails if leakage is ignored

- Picking the voting/stacking winner with test scores is not a fair ensemble
- Anomaly thresholds on test inflate review F1
- Review knapsack tuned on test understates HR cost

## Upstream Tier A building blocks

`voting-ensemble-attrition`, `stacking-credit-risk`, `blending-payment-risk`,
`network-intrusion-anomaly`, `cost-sensitive-collections`

## Limitations

Synthetic attrition table. Two-base ensembles for smoke latency.
