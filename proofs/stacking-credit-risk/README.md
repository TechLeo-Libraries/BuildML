# stacking-credit-risk

## Business purpose

Stack logistic and random-forest base learners with a logistic meta-learner for mortgage default risk, using out-of-fold train predictions only.

## Data source

In-repo synthetic mortgage table (`load_mortgage_default_synthetic`): license-clear, deterministic. **Not** a real credit bureau extract.

## Leakage controls

- Stratified train / validation / test before any fit
- Impute / encode / scale on train only
- Stacking OOF meta features from train CV folds only (`cv=3`)
- Test `evaluate_ensemble` after lock
- Industry StackingClassifier twin uses the same SplitPlan

## BuildML API steps

1. `Session.ingest` → `set_roles` → `split` (stratified)
2. `impute` → `encode` → `scale`
3. `fit_stacking(LR+RF, cv=3)`
4. `evaluate_ensemble(validation)` → `evaluate_ensemble(test)`
5. `save_ensemble_bundle`

## Metrics

Primary holdout: accuracy, F1, ROC-AUC on test (see `results/results.json`).

## Industry comparison (Tier C)

Filled: sklearn `StackingClassifier(cv=3)` twin via `baseline_industry.py` → `results/comparison.json`.

## Limitations

- Synthetic default labels; two-base stack only
- Single seed; not a regulated underwriting certification

## How to run

```bash
python proofs/stacking-credit-risk/script.py
python proofs/stacking-credit-risk/baseline_industry.py
```
