# Keystone Underwrite ML

**Tier B** cross-domain product proof: stacking ensemble + AutoML search +
declared-assumption causal outreach effect for synthetic mortgage underwriting.

## Product narrative

Keystone stacks base learners, searches model families under a budget, and
estimates the effect of confounded borrower outreach:

1. Fits a two-base stacking ensemble with OOF meta features (cv=3)
2. Runs AutoML (native / FLAML / AutoGluon) with CV selection: never test
3. Declares causal assumptions and estimates outreach ATE (AIPW)

## Status

Run `script.py`. Outputs land under `results/` (summary and stage JSON)..

## How to run

```bash
python proofs\keystone-underwrite-ml\script.py
```

## Leakage controls

- Stratified split before stacking / AutoML / causal
- OOF meta features from train CV folds only (cv=3)
- AutoML search/selection never uses the test partition
- Causal assumptions declared before `session.causal.fit`

## What fails if leakage is ignored

- Stacking with test in OOF folds invents ensemble ROC
- Fitting AutoML with test in the search loop invents leaderboard wins
- Skipping causal assumption declaration hides confounding risk

## Upstream Tier A building blocks

`stacking-credit-risk`, `blending-payment-risk`, `voting-ensemble-attrition`,
`churn-automl-search`, `causal-treatment-effect`, `mortgage-default-classical`

## Limitations

Synthetic mortgage: not FCRA / bureau data. Causal ATE assumes declared
unconfoundedness (not proven).
