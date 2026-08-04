# blending-payment-risk

## Business purpose

Score payment-rail authorizations for attack risk with a holdout-blend ensemble (logistic + random forest bases, logistic meta-learner) without leaking Session validation/test into the blend.

## Data source

In-repo synthetic payment authorizations (`load_payment_rail_anomaly_synthetic`): license-clear, deterministic. **Not** a card-network extract.

## Leakage controls

- Stratified outer train / validation / test before scale / blend
- Blend holdout carved from train only (`holdout_fraction=0.2`)
- Session validation / test never used for meta-learner fit
- Test `session.ensemble.evaluate` after lock
- Industry holdout-blend twin uses the same SplitPlan

## BuildML API steps

1. `Session.ingest` → `set_roles` → `split` (stratified)
2. `scale`
3. `session.ensemble.fit_blending(LR+RF, holdout_fraction=0.2)`
4. `session.ensemble.evaluate(validation)` → `session.ensemble.evaluate(test)`
5. `session.ensemble.save_bundle`

## Metrics

Primary holdout: accuracy, F1, ROC-AUC on test (see `results/results.json`).

## Industry comparison (Tier C)

Industry twin: sklearn holdout-blend twin via `baseline_industry.py` → `results/comparison.json`.

## Limitations

- Synthetic payment labels; supervised blend assumes labeled attacks
- Single seed; not a fraud certification

## How to run

```bash
python proofs/blending-payment-risk/script.py
python proofs/blending-payment-risk/baseline_industry.py
```
