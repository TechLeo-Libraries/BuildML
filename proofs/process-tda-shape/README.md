# process-tda-shape

## Business purpose

Classify manufacturing process health from sensor feature clouds using
topological (TDA) descriptors: distinct from credit-risk TDA proofs.

## Data source

Synthetic / in-script license-clear in-spec vs drifted process clouds generated
by `script.py` (temperature, pressure, vibration, flow, torque).

## Leakage controls

- Stratified train / validation / test before any fit
- Scale + TDA vectorizer / head fit on train only
- Holdout `session.tda.evaluate` after model lock
- Test never used for selection

## BuildML API steps

1. `ingest` → `set_roles` → `split` → `scale`
2. `session.tda.fit` (persistence_image + logistic head)
3. `session.tda.evaluate` on validation then test
4. `session.tda.save_bundle`

## Metrics

See `results/results.json` after a successful run (accuracy / macro_f1).

## Industry comparison (Tier C)

Industry twin: `baseline_industry.py` fits classical logistic regression on scaled
raw process features (no TDA) on the same stratified split
(`results/comparison.json`).

## Limitations

Synthetic sensor clouds; TDA features are shape descriptors, not plant SPC.
Requires `ripser` + `persim`. Skips honestly with `skipped_missing_extra` when
extras are unavailable.
