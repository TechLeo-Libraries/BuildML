# Canyon Segment Studio

**Tier B** cross-domain product proof — unsupervised clustering + classical
segment propensity + decision thresholds for CRM targeting.

## Product narrative

Canyon segments a synthetic CRM portfolio, scores outreach propensity, and
allocates a contact budget. The platform:

1. Fits k-means on train-scaled PCA features (external labels eval-only)
2. Trains classical logistic respond propensity on the same split
3. Selects threshold / knapsack outreach policies on validation only

## Status

`completed` — run `script.py`; see `results/summary.json` and stage JSONs.

## How to run

```bash
.\.venv\Scripts\python.exe proofs\canyon-segment-studio\script.py
```

## Leakage controls

- Scale + PCA + clusters fit on train only
- External segment labels used only for cluster evaluation
- Propensity + decision policies selected on validation only
- Test after each stage locks

## What fails if leakage is ignored

- Clustering with test-conditioned PCA overstates segment purity
- Using external labels as features collapses unsupervised into supervised
- Outreach thresholds tuned on test understate CRM cost

## Upstream Tier A building blocks

`sku-embedding-clusters`, `cluster-customer-segments`, `loan-approval-classical`,
`campaign-budget-optimize`, `cost-sensitive-collections`

## Limitations

Synthetic CRM features. External labels exist only for evaluation.
