# Ensemble learning (deep)

> **Install (GitHub 2.x):**
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
| Classical supervised metrics via `evaluate_ensemble` | Unsupervised cluster validity |
| Ensemble bundle + classical pipeline compatibility | Session checkpoint substitute |
| sklearn Voting*/Stacking* façade (+ honest blend) | AutoML search product |

---

## Strategy choice

| Strategy | API | Meta-learner | When to prefer |
| --- | --- | --- | --- |
| Voting | `fit_voting` | None (hard/soft aggregate) | Diverse bases, simple combiner |
| Stacking | `fit_stacking` | CV out-of-fold inside train | Learned combiner; usually best default |
| Blending | `fit_blending` | Single holdout carved from train | Explicit holdout blend; smaller meta set |

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
session.fit_voting(bases, voting="soft")
session.evaluate_ensemble(partition="validation")
session.evaluate_ensemble(partition="test")
```

### Stacking

```python
session.fit_stacking(bases, cv=5, final_estimator=LogisticRegression(max_iter=500))
# disclosures record cv and train-only OOF contract
print(session.ensemble_plan.disclosures)
```

sklearn `StackingClassifier` / `StackingRegressor` build meta features with
cross-validation **on the train matrix only**: because Session fit materializes
train rows exclusively. Session test stays out.

### Blending

```python
session.fit_blending(
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

`fit_*` sets both `ensemble_plan` and classical `fit_result`:

- `evaluate_ensemble`: supervised metrics + ensemble disclosures
- `evaluate` / `predict`: same estimator path
- `save_pipeline` / `save_model`: classical artifacts
- `save_ensemble_bundle`: strategy disclosures + EnsemblePlan

```python
session.save_ensemble_bundle("artifacts/ensemble_bundle")
session.save_pipeline("artifacts/ensemble_pipeline", evaluate_partition="test")
```

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

AI tools (allowlist): `fit_voting`, `fit_stacking`, `fit_blending`,
`evaluate_ensemble`, `save_ensemble_bundle`, `load_ensemble_bundle`.

---

## Failure modes

| Failure | Symptom | Fix |
| --- | --- | --- |
| No split | `LeakageError` / validation error on fit | `split` first |
| Soft voting without proba | `ValidationError` | Use hard voting or proba-capable bases |
| One estimator | `ValidationError` | Pass ≥2 named bases |
| Tiny blend holdout | Warning in fit result | Larger train or smaller `holdout_fraction`; prefer stacking |
| Expecting plans in ensemble bundle | Missing impute/scale at reload | Use `save_pipeline` |
| Calling RF via `fit` “an ensemble product” | Single-estimator history | Use `fit_voting` / `fit_stacking` / `fit_blending` |

---

## Regression

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

session.fit_voting(
    {"ridge": Ridge(), "rf": RandomForestRegressor(n_estimators=40, random_state=0)},
    task="regression",
)
session.evaluate_ensemble(partition="test")
```

Voting for regression averages predictions (VotingRegressor). Stacking/blending
use Ridge as the default meta-learner when unspecified.

---

## Intentional residuals (non-blocking)

- No dedicated ensemble dashboard charts
- No fold-local preprocess recipe inside stacking CV (same Session-global
  preprocess caveats as classical CV: see [leakage guide](leakage-cv-recipes.md))
- AutoML and forecasting are separate Session paths (see their deep guides)

Runnable mirror: [`examples/ensemble_vote_stack_loop.py`](../examples/ensemble_vote_stack_loop.py).
