# Ledger Underwriting Studio

**Tier B** cross-domain product proof: classical + AutoML + causal (declared
assumptions) + cost-sensitive decisions + calibration for a synthetic credit book.

## Product narrative

Ledger scores loan applications, searches estimators under a time budget,
estimates the effect of an outreach treatment under **declared** causal
assumptions, then selects a cost-sensitive approve threshold on validation
only: with calibration diagnostics before the holdout confirm.

1. Classical logistic scorer on stratified split
2. AutoML search (FLAML / AutoGluon when present; else native): test never in search
3. AIPW causal fit after `session.causal.declare_assumptions` (unconfoundedness + positivity)
4. Threshold / knapsack policies selected on **validation only**
   (default `impute`/`encode`/`scale` skip `ignore`/`id` roles so
   `review_cost` / `app_id` stay usable for knapsack)
5. Calibration report on validation, confirmed on untouched test

## Status

Run `script.py`. Outputs land under `results/` (summary and stage JSON)..

## How to run

```bash
python proofs\ledger-underwriting-studio\script.py
```

## Leakage controls (critical)

- Stratified split before classical / AutoML / causal / decisions
- Causal assumptions declared before `session.causal.fit` (required API gate)
- Decision threshold **not** tuned on test: validation selection, test confirm
- AutoML selection never uses the test partition
- Calibration reported on validation then confirmed on test

## What fails if leakage is ignored

- Tuning the approve threshold on test understates expected review cost
- Skipping causal assumption declaration hides confounding risk
- Fitting AutoML with test in the search loop invents leaderboard wins
- Reporting calibration only on train hides probability miscalibration

## Upstream Tier A building blocks

`loan-approval-classical`, `churn-automl-search`, `causal-treatment-effect`,
`cost-sensitive-collections`

## Limitations

Synthetic underwriting: not FCRA / bureau data. Causal ATE assumes declared
unconfoundedness (not proven). Product proof, not a production LOS certification.
