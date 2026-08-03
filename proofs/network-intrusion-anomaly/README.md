# network-intrusion-anomaly

## Business purpose

Flag rare network intrusion / fraud-like flows from traffic features so SOC
analysts can investigate high-score alerts under a controlled alert rate.

## Data source

Synthetic KDD-inspired flow table (`load_intrusion_anomaly_synthetic`) :
license-clear. Not the full KDD Cup 1999 corpus.

## Leakage controls

- Stratified train / validation / test (rare attack class preserved)
- Unsupervised detector fitted on **train** features only
- `tune_anomaly_threshold` on **validation** labels only (`allow_test_tuning=False`)
- Test scored and evaluated after the threshold is locked

## BuildML API steps

1. `ingest` → `set_roles` → `split` → `scale`
2. `fit_anomaly` (PyOD when available, else sklearn IsolationForest)
3. `tune_anomaly_threshold(partition="validation")`
4. `score_anomalies` / `evaluate_anomaly` on test
5. `save_anomaly_bundle`

## Metrics

Labeled precision/recall/F1 (and related) on test; alert rate; tuned threshold
details in `results/results.json`.

## Industry comparison (Tier C)

Filled: `baseline_industry.py` runs sklearn `IsolationForest` on the **same SplitPlan**, tunes the decision threshold on validation F1, and writes `results/comparison.json`. BuildML path typically uses PyOD HBOS when installed. Deltas are descriptive on one synthetic draw (competitive qualitative bar 5-B).
## Limitations

Synthetic attacks; labeled eval overstates production unlabeled deployment.
