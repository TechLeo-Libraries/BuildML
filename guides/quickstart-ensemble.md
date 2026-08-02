# Ensemble quickstart

> **Install first (GitHub):** PyPI `buildml` is still legacy 1.x and does **not**
> install Session 2.x. Use
> `pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"`
> (or an editable checkout). Ensembles use core sklearn — no optional extra.
> See [installation](../docs/installation.rst).

Native voting, stacking, and holdout blending on the Session — not merely
passing a `RandomForest` to `Session.fit`.

**Go deeper:** [Ensemble deep](ensemble-deep.md) ·
[Classical E2E](classical-end-to-end.md) ·
[Leakage](leakage-cv-recipes.md) ·
[Artifacts](artifacts-checkpoints-bundles.md).

---

## First loop: soft voting

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from buildml import Session

frame = pd.DataFrame(
    {
        "age": [21, 35, 40, 29, 33, 52, 47, 25, 38, 44, 31, 50],
        "income": [40, 60, 80, 50, 70, 90, 65, 45, 55, 75, 62, 88],
        "approved": [0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
    }
)

bases = {
    "lr": LogisticRegression(max_iter=500),
    "rf": RandomForestClassifier(n_estimators=50, random_state=0),
}

session = (
    Session.ingest(frame)
    .set_roles({"age": "feature", "income": "feature", "approved": "target"})
    .split(test_size=0.25, validation_size=0.25, stratify=True, random_state=0)
    .impute(strategy="median")
    .scale(method="standard")
)

session.fit_voting(bases, voting="soft", task="classification")
validation = session.evaluate_ensemble(partition="validation")
test = session.evaluate_ensemble(partition="test")
print(validation.metrics, test.metrics)
```

`fit_voting` sets classical `fit_result`, so `evaluate` / `predict` /
`save_pipeline` also work.

---

## Stacking (CV meta-learner inside train)

```python
session.fit_stacking(bases, cv=3, task="classification")
print(session.evaluate_ensemble(partition="test").metrics)
```

Stacking folds stay inside **train**. Session test never enters meta-feature
construction. Prefer this over blending when you want out-of-fold meta features.

---

## Blending (holdout carved from train)

```python
session.fit_blending(bases, holdout_fraction=0.2, random_state=0)
print(session.ensemble_plan.disclosures[:3])
print(session.evaluate_ensemble(partition="test").metrics)
```

The blend holdout is an **inner train carve**, not Session validation/test.
Bases are refit on full train after meta fit by default (disclosed).

---

## Persist

```python
session.save_ensemble_bundle("artifacts/ensemble_bundle")
session.save_pipeline("artifacts/ensemble_pipeline", evaluate_partition="test")

fresh = (
    Session.ingest(session.to_pandas())
    .set_roles({"age": "feature", "income": "feature", "approved": "target"})
    .split(test_size=0.25, validation_size=0.25, stratify=True, random_state=0)
)
fresh.load_ensemble_bundle("artifacts/ensemble_bundle")
print(fresh.evaluate_ensemble(partition="test").metrics)
```

`buildml.ensemble_bundle.v1` stores `EnsemblePlan` + fit contract.
Prefer `save_pipeline` when impute/encode/scale plans must travel with the
estimator. Session checkpoints do **not** embed the ensemble.

---

## Boundaries (honest)

| Use | Do not confuse with |
| --- | --- |
| `fit_voting` / `fit_stacking` / `fit_blending` | `Session.fit(RandomForest…)` (single estimator) |
| Train-only stacking CV / blend holdout | Fitting meta-learners on Session test |
| `evaluate_ensemble` supervised metrics | Unsupervised cluster validity |
| Ensemble bundle | Session checkpoint |

Teaching: `session.explain("fit_stacking")`. Runnable mirror:
[`examples/ensemble_vote_stack_loop.py`](../examples/ensemble_vote_stack_loop.py).
