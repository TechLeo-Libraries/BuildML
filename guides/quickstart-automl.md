# AutoML quickstart

> **Install:** Install Session 2.x with `pip install buildml` (2.5.x). Legacy 1.x remains available as `pip install "buildml==1.0.9"`. Use
> `pip install buildml`
> Randomized/grid/evolutionary AutoML is core sklearn: no optional extra.
> Optuna backend: `buildml[automl]`. Industry adapters (FLAML / AutoGluon) and
> GBDT families: `buildml[automl-industry]`.
> See [installation](../docs/installation.rst).

Joint model-family + fold-local preprocess-strategy search on the Session :
beyond tuning one fixed estimator with `grid_search` / `optuna_search`.

**Go deeper:** [AutoML deep](automl-deep.md) ·

**Proof:** [churn-automl-search](../proofs/churn-automl-search/) (+ Tier C RandomizedSearchCV twin).
[Leakage](leakage-cv-recipes.md) ·
[Diagnostics & search](classical-diagnostics-search.md) ·
[Artifacts](artifacts-checkpoints-bundles.md).

---

## First loop: randomized family + recipe search

Prefer an **unpoisoned** frame (roles + split, no Session-global impute/scale
before search). AutoML uses fold-local `PreprocessRecipe` strategies.

```python
import pandas as pd
import numpy as np

from buildml import Session

rng = np.random.default_rng(0)
n = 200
x1 = rng.normal(size=n)
x2 = rng.normal(size=n)
cat = rng.choice(["a", "b", "c"], size=n)
y = (0.8 * x1 - 0.4 * x2 + rng.normal(scale=0.35, size=n) > 0).astype(int)
frame = pd.DataFrame({"x1": x1, "x2": x2, "cat": cat, "y": y})

session = (
    Session.ingest(frame)
    .set_roles({"x1": "feature", "x2": "feature", "cat": "feature", "y": "target"})
    .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0)
)

result = session.automl.run(
    method="randomized",
    selection="cv",  # default: train-fold ranking (fast; optimistic vs outer)
    n_trials=12,
    cv=3,
    include_recipe_search=True,
    families=("logistic", "random_forest", "gradient_boosting"),
    random_state=0,
)
result.show()
print(result.leaderboard().head())

validation = session.automl.evaluate(partition="validation")
test = session.automl.evaluate(partition="test")
print(validation.metrics, test.metrics)
```

`session.automl.run` sets classical `fit_result`, so `evaluate` / `predict` /
`save_pipeline` also work.

---

## Nested selection (prominent honesty path)

Default `selection='cv'` ranks by train-fold CV. For post-selection claims use
**`selection='nested'`** (outer mean±std), then confirm on Session test:

```python
nested = session.automl.run(
    method="randomized",
    selection="nested",
    n_trials=8,
    cv=3,
    outer_cv=3,
    include_recipe_search=True,
    random_state=1,
)
print(nested.outer_score_mean, nested.outer_score_std)
print(nested.leaderboard()[["rank", "family", "mean_score", "outer_score_mean", "nested_cv_disclosed"]])
print(session.automl.evaluate(partition="test").metrics)
```

Outer folds stay inside **train**. Session test never enters selection.

---

## Industry backends (optional)

Install `buildml[automl-industry]` for FLAML / AutoGluon adapters and
LightGBM / XGBoost / CatBoost native families:

```python
from buildml.automl import automl_capability_matrix

print(automl_capability_matrix()["backends"])

# FLAML on train only (validation ranking; nested not supported)
session.automl.run(backend="flaml", selection="validation", time_budget=120)

# Deepened Optuna (pruning, study persistence via AutoMLBudget)
from buildml.automl import AutoMLBudget

session.automl.run(
    backend="optuna",
    n_trials=20,
    budget=AutoMLBudget(
        max_trials=20,
        enable_pruning=True,
        study_storage="sqlite:///automl_study.db",
    ),
)

# Export trial comparison metrics
from buildml.automl import export_comparison_metrics

export_comparison_metrics(session.automl.result, "artifacts/automl_trials.json")
```

Native `backend='native'` remains the leakage-first path with fold-local
recipe search. Industry adapters bypass recipe strategy search: see
`limitations` on `AutoMLResult`.

---

## Optional voting of top families

```python
session.automl.run(
    method="randomized",
    n_trials=10,
    include_recipe_search=True,
    include_ensembles=True,
    max_ensemble_bases=3,
    random_state=2,
)
print(session.automl.plan.best_kind, session.automl.plan.ensemble_bases)
```

Stacking/blending remain separate Session APIs (`session.ensemble.fit_stacking` /
`session.ensemble.fit_blending`) when you want those strategies explicitly.

---

## Bundle

```python
session.automl.save_bundle(".buildml-artifacts/automl_bundle")
```

Distinct from Session checkpoints and classical pipelines. See
[artifacts](artifacts-checkpoints-bundles.md).

---

## Honest scope

| Is | Is not |
| --- | --- |
| Finite catalog of sklearn families + recipe strategies | Neural architecture search (NAS) |
| Fold-local leakage-safe selection | Causal discovery |
| Trial-budgeted search with disclosures | Fully automated AI scientist |
| Optional voting of top families | Unbounded Autosklearn zoo |

Session-global preprocess before AutoML is **refused** by default (same
contract as `cv_score` / `grid_search`). Re-ingest unpoisoned data or set
`allow_session_global_preprocess=True` explicitly (scores remain leakage-biased).
