# canyon-segment-studio

This script composes `session.unsupervised`, classical `session.fit`, and
`session.decision` on one synthetic CRM table. It is not a product BuildML
ships.

The script fits k-means on train-scaled PCA features (external labels
eval-only), trains a logistic respond-propensity scorer on the same split,
and selects threshold / knapsack outreach policies on validation only.

## Data

Synthetic CRM features. External labels exist only for evaluation.

## Leakage

Scale + PCA + clusters fit on train only. External segment labels used only
for cluster evaluation. Propensity + decision policies selected on
validation only. Test after each stage locks.

## What fails if leakage is ignored

Clustering with test-conditioned PCA overstates segment purity. Using
external labels as features collapses unsupervised into supervised.
Outreach thresholds tuned on test understate CRM cost.

## How to run

```bash
python proofs/canyon-segment-studio/script.py
```

## What you'll get

`results/` summary and per-stage JSON.

## Upstream

`sku-embedding-clusters`, `cluster-customer-segments`,
`loan-approval-classical`, `campaign-budget-optimize`,
`cost-sensitive-collections`.

## Limitations

Synthetic CRM features. External labels exist only for evaluation.
