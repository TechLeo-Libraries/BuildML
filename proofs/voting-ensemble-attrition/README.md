# voting-ensemble-attrition

## Business purpose

Predict employee attrition with a soft-voting ensemble (logistic + random forest) so HR can prioritize retention outreach without leaking holdout labels into model selection.

## Data source

In-repo synthetic attrition table (`load_attrition_tabular_synthetic`) — license-clear, deterministic. **Not** a real employee extract.

## Leakage controls

- Stratified train / validation / test before encode / scale / ensemble fit
- One-hot encode and scale fit on train only
- Voting bases fit on train only
- Test `evaluate_ensemble` after lock
- Industry VotingClassifier twin uses the same SplitPlan

## BuildML API steps

1. `Session.ingest` → `set_roles` → `split` (stratified)
2. `encode` → `scale`
3. `fit_voting(LR+RF, voting="soft")`
4. `evaluate_ensemble(validation)` → `evaluate_ensemble(test)`
5. `save_ensemble_bundle`

## Metrics

Primary holdout: accuracy, F1, ROC-AUC on test (see `results/results.json`).

## Industry comparison (Tier C)

Filled — sklearn `VotingClassifier(soft)` twin via `baseline_industry.py` → `results/comparison.json`.

## Limitations

- Synthetic HR labels; two-base vote only
- Single seed; no nested outer CV

## How to run

```bash
python proofs/voting-ensemble-attrition/script.py
python proofs/voting-ensemble-attrition/baseline_industry.py
```
