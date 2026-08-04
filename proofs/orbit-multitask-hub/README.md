# Orbit Multitask Hub

**Tier B** cross-domain product proof: multi-output multitask learning +
AutoML/classical search + validation-tuned decision thresholds.

## Product narrative

Orbit is a retail SKU outcome hub. Each SKU has joint buy / high-margin
targets; a primary buy scorer feeds promo allocation. The platform:

1. Fits multi-output multitask models on a shared feature set
2. Runs native AutoML (or classical logistic fallback) on the buy target
3. Selects cost-sensitive thresholds / knapsack on validation only

## Status

Run `script.py`. Outputs land under `results/` (summary and stage JSON)..

## How to run

```bash
python proofs\orbit-multitask-hub\script.py
```

## Leakage controls

- Split before multitask / AutoML / decision fit
- AutoML CV uses train folds only
- Decision policies selected on validation only
- Test evaluated after each stage locks

## What fails if leakage is ignored

- Multitask heads trained on test labels overstate joint skill
- AutoML winner picked with test scores is not a fair search
- Promo thresholds tuned on test understate campaign cost

## Upstream Tier A building blocks

`sku-multitask-retail`, `multi-target-underwriting`, `churn-automl-search`,
`campaign-budget-optimize`, `loan-approval-classical`

## Limitations

Synthetic SKU outcomes. Small AutoML trial budget for smoke latency.
