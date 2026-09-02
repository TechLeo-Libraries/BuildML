# cornerstone-mortgage-suite

This script composes classical `session.fit`, `session.causal`, and
`session.decision` on one synthetic mortgage book. It is not a product BuildML ships.

High-LTV / high-DTI loans are riskier; counseling is offered more often to
those same loans (confounded). The script fits a logistic default scorer,
estimates the counseling ATE under declared unconfoundedness / positivity,
and selects review threshold / knapsack on validation only.

## Data

Synthetic mortgage. Not FCRA / bureau data.

## Leakage

Stratified split before classical / causal / decisions. Causal assumptions
declared before `session.causal.fit`. Decision threshold + knapsack
selected on validation only. Test evaluate after each stage locks.

## What fails if leakage is ignored

Tuning the review threshold on test understates expected loss. Skipping
causal assumption declaration hides confounding risk. Fitting classical
scores on the full book invents holdout ROC.

## How to run

```bash
python proofs/cornerstone-mortgage-suite/script.py
```

## What you'll get

`results/` summary and per-stage JSON.

## Upstream

`mortgage-default-classical`, `loan-approval-classical`,
`causal-treatment-effect`, `uplift-marketing-causal`,
`cost-sensitive-collections`.

## Limitations

Synthetic mortgage: not FCRA / bureau data. Causal ATE assumes declared
unconfoundedness (not proven).
