# keystone-underwrite-ml

This script composes `session.ensemble`, `session.automl`, and
`session.causal` on one synthetic mortgage table. It is not a product BuildML ships.

The script fits a two-base stacking ensemble with OOF meta features
(`cv=3`), runs AutoML (native / FLAML / AutoGluon) with CV selection that
never uses test, and estimates a confounded borrower-outreach ATE (AIPW)
after declaring causal assumptions.

## Data

Synthetic mortgage. Not FCRA / bureau data.

## Leakage

Stratified split before stacking / AutoML / causal. OOF meta features from
train CV folds only (`cv=3`). AutoML search/selection never uses the test
partition. Causal assumptions declared before `session.causal.fit`.

## What fails if leakage is ignored

Stacking with test in OOF folds invents ensemble ROC. Fitting AutoML with
test in the search loop invents leaderboard wins. Skipping causal
assumption declaration hides confounding risk.

## How to run

```bash
python proofs/keystone-underwrite-ml/script.py
```

## What you'll get

`results/` summary and per-stage JSON.

## Upstream

`stacking-credit-risk`, `blending-payment-risk`,
`voting-ensemble-attrition`, `churn-automl-search`,
`causal-treatment-effect`, `mortgage-default-classical`.

## Limitations

Synthetic mortgage: not FCRA / bureau data. Causal ATE assumes declared
unconfoundedness (not proven).
