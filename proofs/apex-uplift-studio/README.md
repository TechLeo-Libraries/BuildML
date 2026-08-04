# Apex Uplift Studio

**Tier B** cross-domain product proof: causal marketing uplift + classical
conversion scoring + validation-tuned promo allocation.

## Product narrative

Apex plans promo spend for a synthetic CRM book. Treatment assignment is
confounded by RFM features; true spend ATE ≈ 1.2. The studio:

1. Declares causal assumptions and estimates promo ATE (AIPW) with a placebo refute
2. Trains a classical conversion scorer (+ optional spend ridge disclosure)
3. Selects threshold / knapsack promo allocation on validation only

## Status

Run `script.py`. Outputs land under `results/` (summary and stage JSON)..

## How to run

```bash
python proofs\apex-uplift-studio\script.py
```

## Leakage controls

- Shared stratified split before causal / classical / decisions
- Causal assumptions declared before `session.causal.fit`
- Promo budget knapsack / threshold tuned on validation only
- Test evaluated after each stage locks

## What fails if leakage is ignored

- Allocating promo budget on test invents ROI
- Skipping assumption declaration hides confounding in uplift ATE
- Fitting conversion scores on the full book invents holdout ROC

## Upstream Tier A building blocks

`uplift-marketing-causal`, `causal-treatment-effect`,
`loan-approval-classical`, `campaign-budget-optimize`,
`cost-sensitive-collections`

## Limitations

Synthetic uplift DGP: not a real CRM extract. ATE assumes declared
unconfoundedness (not proven).
