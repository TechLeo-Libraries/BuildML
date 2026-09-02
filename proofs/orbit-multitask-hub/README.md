# orbit-multitask-hub

This script composes `session.multitask`, `session.automl` (or classical
logistic fallback), and `session.decision` on one synthetic SKU table. It
is not a product BuildML ships.

Each SKU has joint buy / high-margin targets. A primary buy scorer feeds
promo allocation. AutoML trial budget is small for smoke latency.

## Data

Synthetic SKU outcomes.

## Leakage

Split before multitask / AutoML / decision fit. AutoML CV uses train folds
only. Decision policies selected on validation only. Test evaluated after
each stage locks.

## What fails if leakage is ignored

Multitask heads trained on test labels overstate joint skill. An AutoML
winner picked with test scores is not a fair search. Promo thresholds
tuned on test understate campaign cost.

## How to run

```bash
python proofs/orbit-multitask-hub/script.py
```

## What you'll get

`results/` summary and per-stage JSON.

## Upstream

`sku-multitask-retail`, `multi-target-underwriting`, `churn-automl-search`,
`campaign-budget-optimize`, `loan-approval-classical`.

## Limitations

Synthetic SKU outcomes. Small AutoML trial budget for smoke latency.
