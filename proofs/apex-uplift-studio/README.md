# apex-uplift-studio

This script composes `session.causal`, classical `session.fit`, and
`session.decision` on one synthetic CRM table. It is not a product BuildML
ships.

Treatment assignment is confounded by RFM features; true spend ATE is about
1.2. The script declares causal assumptions, estimates promo ATE (AIPW)
with a placebo refute, trains a conversion scorer (optional spend ridge
disclosure), and selects threshold / knapsack promo allocation on
validation only.

## Data

Synthetic uplift DGP. Not a real CRM extract.

## Leakage

Shared stratified split before causal / classical / decisions. Causal
assumptions declared before `session.causal.fit`. Promo budget knapsack /
threshold tuned on validation only. Test evaluated after each stage locks.

## What fails if leakage is ignored

Allocating promo budget on test invents ROI. Skipping assumption
declaration hides confounding in uplift ATE. Fitting conversion scores on
the full book invents holdout ROC.

## How to run

```bash
python proofs/apex-uplift-studio/script.py
```

## What you'll get

`results/` summary and per-stage JSON.

## Upstream

`uplift-marketing-causal`, `causal-treatment-effect`,
`loan-approval-classical`, `campaign-budget-optimize`,
`cost-sensitive-collections`.

## Limitations

Synthetic uplift DGP. ATE assumes declared unconfoundedness (not proven).
