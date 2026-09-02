# Ensemble learning (deep)

```bash
pip install buildml
```

More than one estimator, combined on the same Session. You need at least
two named bases. Voting, stacking, and blending all fit on train.
Stacking's meta-learner sees out-of-fold predictions **inside train**.
Blending carves a holdout **from train**. Session validation and test
never enter the combiner.

A single RandomForest passed to `session.fit` is not this path.

Short on-ramp: [ensemble quickstart](quickstart-ensemble.md).

## Which combiner

| Strategy | API | Combiner | When |
| --- | --- | --- | --- |
| Voting | `session.ensemble.fit_voting` | Hard or soft aggregate | Diverse bases, no learned meta |
| Stacking | `session.ensemble.fit_stacking` | CV out-of-fold inside train | Learned combiner; usual default |
| Blending | `session.ensemble.fit_blending` | One holdout carved from train | Explicit blend; smaller meta set |

Soft voting needs `predict_proba` on every classification base.

## Voting

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from buildml import Session

bases = {
    "lr": LogisticRegression(max_iter=500),
    "rf": RandomForestClassifier(n_estimators=80, random_state=0),
}

session = (
    Session.ingest(frame)
    .set_roles(...)
    .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0)
    .impute(strategy="median")
    .encode()
    .scale(method="standard")
)
session.ensemble.fit_voting(bases, voting="soft")
session.ensemble.evaluate(partition="validation")
session.ensemble.evaluate(partition="test")
```

Regression voting averages predictions. Unspecified stacking / blending
meta-learners default to Ridge.

## Stacking

```python
session.ensemble.fit_stacking(
    bases, cv=5, final_estimator=LogisticRegression(max_iter=500)
)
print(session.ensemble.plan.disclosures)
```

sklearn `StackingClassifier` / `StackingRegressor` build meta features
with cross-validation on the train matrix only. Session test stays out.

## Blending

```python
session.ensemble.fit_blending(
    bases,
    holdout_fraction=0.2,
    blend_method="predict_proba",
    refit_bases_on_full_train=True,
    random_state=0,
)
```

The carve is from train (stratified for classification). Bases fit on
blend-train. The meta-learner fits on blend-holdout predictions.
`refit_bases_on_full_train=True` (default) refits bases on full train
for deploy and discloses it. Prefer stacking when you want CV OOF
instead of one holdout.

## Evaluate

`fit_*` sets both `session.ensemble.plan` and classical `fit_result`.
`evaluate` scores the combiner and, on the same partition, base
contributions and diversity. That scoring is predict-only. Bases are
not refit. Test never re-enters fitting.

```python
ev = session.ensemble.evaluate(partition="test")
print(ev.metrics)
for row in ev.diagnostics["base_contributions"]:
    print(row["name"], row["metrics"], row["agree_with_ensemble"])
print(ev.diagnostics["diversity"]["mean_pairwise_disagreement"])

session.ensemble.save_bundle("artifacts/ensemble_bundle")
session.save_pipeline("artifacts/ensemble_pipeline", evaluate_partition="test")
```

`buildml.ensemble.build_ensemble_eval_report(...)` builds the same
report without a Session.

## Bundle

`buildml.ensemble_bundle.v1` stores the EnsemblePlan and the FitResult
contract. It does not store the dataset, the split, or preprocess
plans. If you need impute/scale on reload, use `save_pipeline`.
Session checkpoints do not embed fitted ensemble weights.

## What usually goes wrong

- No split: `LeakageError`.
- Soft voting without `predict_proba`: `ValidationError`.
- One estimator: `ValidationError` (need ≥2 named bases).
- Tiny blend holdout: warning on the fit result; prefer stacking.
- Expecting preprocess plans inside the ensemble bundle.
- Fold-local recipes inside stacking CV follow the same Session-global
  refuse as classical CV. See [leakage](leakage-cv-recipes.md).

Runnable mirror: [`examples/ensemble_vote_stack_loop.py`](../examples/ensemble_vote_stack_loop.py).
