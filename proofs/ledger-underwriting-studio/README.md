# ledger-underwriting-studio

This script composes classical `session.fit`, `session.automl`,
`session.causal`, `session.decision`, and calibration on one synthetic
credit book. It is not a product BuildML ships.

The script scores applications, searches estimators under a time budget,
estimates an outreach treatment under declared causal assumptions, then
selects a cost-sensitive approve threshold on validation only, with
calibration diagnostics before the holdout confirm.

Default `impute` / `encode` / `scale` skip `ignore` / `id` roles so
`review_cost` / `app_id` stay usable for knapsack.

## Data

Synthetic underwriting table. Not FCRA / bureau data.

## Leakage

Stratified split before classical / AutoML / causal / decisions. Causal
assumptions are declared before `session.causal.fit` (required API gate).
Decision threshold is not tuned on test: validation selection, test
confirm. AutoML selection never uses the test partition. Calibration is
reported on validation then confirmed on test.

## What fails if leakage is ignored

Tuning the approve threshold on test understates expected review cost.
Skipping causal assumption declaration hides confounding risk. Fitting
AutoML with test in the search loop invents leaderboard wins. Reporting
calibration only on train hides probability miscalibration.

## How to run

```bash
python proofs/ledger-underwriting-studio/script.py
```

## What you'll get

`results/` summary and per-stage JSON.

## Upstream

`loan-approval-classical`, `churn-automl-search`, `causal-treatment-effect`,
`cost-sensitive-collections`.

## Limitations

Synthetic underwriting: not FCRA / bureau data. Causal ATE assumes declared
unconfoundedness (not proven). Not a production LOS certification.
