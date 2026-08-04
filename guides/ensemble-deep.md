# Ensemble learning (deep)

> **Install:**
> `pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"`
> Ensembles are core (sklearn). See [installation](../docs/installation.rst).

This guide covers native Session ensembles: voting, stacking, and holdout
blending with leakage-safe meta-learner fitting, classical evaluation, and
`buildml.ensemble_bundle.v1`. It matches the depth bar of classical / Torch /
RAG / unsupervised guides.

**Related:** [Quickstart](quickstart-ensemble.md) ·
[Classical E2E](classical-end-to-end.md) ·
[Leakage](leakage-cv-recipes.md) ·
[Artifacts](artifacts-checkpoints-bundles.md) ·
[Diagnostics & search](classical-diagnostics-search.md)

---

## What this path is (and is not)

| Is | Is not |
| --- | --- |
| Native multi-estimator voting / stacking / blending | Passing one RandomForest to `Session.fit` |
| Train-only stacking CV and train-inner blend holdout | Meta features built from Session test |
| Classical supervised metrics via `session.ensemble.evaluate` | Unsupervised cluster validity |
| Ensemble bundle + classical pipeline compatibility | Session checkpoint substitute |
| sklearn Voting*/Stacking* façade (+ honest blend) | AutoML search product |

Related: [unsupervised](unsupervised-deep.md), [AutoML](automl-deep.md),
[forecasting](forecasting-deep.md), [anomaly](anomaly-deep.md). Explicit
non-goals (neuromorphic, swarm zoo, digital twins, AV/robotics, TTS, full COCO
suite) stay out.

---

## Strategy choice

| Strategy | API | Meta-learner | When to prefer |
| --- | --- | --- | --- |
| Voting | `session.ensemble.fit_voting` | None (hard/soft aggregate) | Diverse bases, simple combiner |
| Stacking | `session.ensemble.fit_stacking` | CV out-of-fold inside train | Learned combiner; usually best default |
| Blending | `session.ensemble.fit_blending` | Single holdout carved from train | Explicit holdout blend; smaller meta set |

Soft voting needs `predict_proba` on every classification base.

---

## Core loops

### Voting

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from buildml import Session

bases = {
    "lr": LogisticRegression(max_iter=500),
    "rf": RandomForestClassifier(n_estimators=80, random_state=0),
}

session = (
    Session.ingest(frame)  # frame with features + target
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

### Stacking

```python
session.ensemble.fit_stacking(bases, cv=5, final_estimator=LogisticRegression(max_iter=500))
# disclosures record cv and train-only OOF contract
print(session.ensemble.plan.disclosures)
```

sklearn `StackingClassifier` / `StackingRegressor` build meta features with
cross-validation **on the train matrix only**: because Session fit materializes
train rows exclusively. Session test stays out.

### Blending

```python
session.ensemble.fit_blending(
    bases,
    holdout_fraction=0.2,
    blend_method="predict_proba",
    refit_bases_on_full_train=True,
    random_state=0,
)
```

Leakage contract:

1. Carve `holdout_fraction` from **train** (stratified for classification).
2. Fit bases on blend-train; fit meta-learner on blend-holdout predictions.
3. Optionally refit bases on full train for deploy (default; disclosed).
4. Never use Session validation/test for meta fit.

Prefer stacking when you want CV OOF meta features instead of one holdout.

---

## Evaluation and classical interop

`fit_*` sets both `session.ensemble.plan` and classical `fit_result`:

- `session.ensemble.evaluate`: supervised metrics + ensemble disclosures
- **Base-learner contributions** and **diversity** on the same partition
  (`diagnostics["base_contributions"]`, `diagnostics["diversity"]`,
  `diagnostics["ensemble_report"]`) — predict-only scoring of train-fitted
  bases (no refit during evaluate; Session test never re-enters fitting)
- `evaluate` / `predict`: same estimator path
- `save_pipeline` / `save_model`: classical artifacts
- `session.ensemble.save_bundle`: strategy disclosures + EnsemblePlan

```python
ev = session.ensemble.evaluate(partition="test")
print(ev.metrics)
for row in ev.diagnostics["base_contributions"]:
    print(row["name"], row["metrics"], row["agree_with_ensemble"])
print(ev.diagnostics["diversity"]["mean_pairwise_disagreement"])

session.ensemble.save_bundle("artifacts/ensemble_bundle")
session.save_pipeline("artifacts/ensemble_pipeline", evaluate_partition="test")
```

Library helper: `buildml.ensemble.build_ensemble_eval_report(...)` builds the
same contribution / diversity report without a Session.

---

## Artifact boundary

| Artifact | Contains | Does not |
| --- | --- | --- |
| `buildml.ensemble_bundle.v1` | EnsemblePlan, FitResult contract | Dataset, splits, preprocess plans |
| Pipeline bundle | plans + estimator + card | EnsemblePlan prose (estimator may still be the ensemble) |
| Session checkpoint | data, roles, splits, history | Fitted ensemble weights |

See [Artifacts](artifacts-checkpoints-bundles.md).

---

## Teaching surface

```python
session.explain("fit_stacking", moment="before")
session.explain("fit_blending")
session.walkthrough().ensemble_status
```

Concept keys: `ensemble-voting-vs-single-tree`, `ensemble-stacking-oof`,
`ensemble-blending-holdout`, `ensemble-bundle-boundary`.

AI tools (allowlist): `session.ensemble.fit_voting`, `session.ensemble.fit_stacking`, `session.ensemble.fit_blending`,
`session.ensemble.evaluate`, `session.ensemble.save_bundle`, `session.ensemble.load_bundle`.

---

## Failure modes

| Failure | Symptom | Fix |
| --- | --- | --- |
| No split | `LeakageError` / validation error on fit | `split` first |
| Soft voting without proba | `ValidationError` | Use hard voting or proba-capable bases |
| One estimator | `ValidationError` | Pass ≥2 named bases |
| Tiny blend holdout | Warning in fit result | Larger train or smaller `holdout_fraction`; prefer stacking |
| Expecting plans in ensemble bundle | Missing impute/scale at reload | Use `save_pipeline` |
| Calling RF via `fit` “an ensemble product” | Single-estimator history | Use `session.ensemble.fit_voting` / `session.ensemble.fit_stacking` / `session.ensemble.fit_blending` |

---

## Regression

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

session.ensemble.fit_voting(
    {"ridge": Ridge(), "rf": RandomForestRegressor(n_estimators=40, random_state=0)},
    task="regression",
)
session.ensemble.evaluate(partition="test")
```

Voting for regression averages predictions (VotingRegressor). Stacking/blending
use Ridge as the default meta-learner when unspecified.

---

## Known limits

- No dedicated ensemble dashboard charts
- No fold-local preprocess recipe inside stacking CV (same Session-global
  preprocess caveats as classical CV: see [leakage guide](leakage-cv-recipes.md))
- AutoML and forecasting are separate Session paths (see their deep guides)

Runnable mirror: [`examples/ensemble_vote_stack_loop.py`](../examples/ensemble_vote_stack_loop.py).
