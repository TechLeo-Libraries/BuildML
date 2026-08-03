# Cornerstone Mortgage Suite

**Tier B** cross-domain product proof: classical mortgage default scoring +
declared-assumption causal counseling effect + cost-sensitive decisions.

## Product narrative

Cornerstone underwrites a synthetic mortgage book. High-LTV / high-DTI loans
are riskier; counseling is offered more often to those same loans (confounded).
The suite:

1. Fits a classical logistic default scorer on a stratified split
2. Estimates the counseling ATE under declared unconfoundedness / positivity
3. Selects review threshold / knapsack on validation only

## Status

`completed`: run `script.py`; see `results/summary.json` and stage JSONs.

## How to run

```bash
.\.venv\Scripts\python.exe proofs\cornerstone-mortgage-suite\script.py
```

## Leakage controls

- Stratified split before classical / causal / decisions
- Causal assumptions declared before `fit_causal`
- Decision threshold + knapsack selected on validation ONLY
- Test evaluate after each stage locks

## What fails if leakage is ignored

- Tuning the review threshold on test understates expected loss
- Skipping causal assumption declaration hides confounding risk
- Fitting classical scores on the full book invents holdout ROC

## Upstream Tier A building blocks

`mortgage-default-classical`, `loan-approval-classical`,
`causal-treatment-effect`, `uplift-marketing-causal`,
`cost-sensitive-collections`

## Limitations

Synthetic mortgage: not FCRA / bureau data. Causal ATE assumes declared
unconfoundedness (not proven).
