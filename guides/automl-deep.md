# AutoML (deep)

```bash
pip install buildml
# Optuna: pip install "buildml[automl]"
# FLAML / AutoGluon: pip install "buildml[automl-industry]"
```

Family and preprocess strategy are part of the decision, not a
single-estimator grid you already chose. Default selection is `cv`
(train-fold ranking). Session test never enters selection. Confirm once
with `session.automl.evaluate(partition="test")`.

Session-global `impute` / `encode` / `scale` then AutoML is the same
`LeakageError` as classical CV, even with a recipe. Nested selection is
native-only. FLAML / AutoGluon fit on train only and disclose that
fold-local recipes are bypassed.

Short on-ramp: [AutoML quickstart](quickstart-automl.md) ·
[Leakage](leakage-cv-recipes.md).

## When this is the right call

Use `grid_search` / `optuna_search` / `evolutionary_search` when the
model family is already decided. Use `session.automl.run` when family
and preprocess strategy are still open.

| Concern | Single-estimator search | `session.automl.run` |
| --- | --- | --- |
| Estimator | One model you chose | Catalog of families (+ industry GBDT when installed) |
| Preprocess | Optional knobs on one recipe | Discrete strategy search |
| Backends | Optuna for `optuna_search`; GA for evolutionary | `native`, `optuna`, `flaml`, `autogluon` |
| Ensembles | Bring your own | Optional voting/stacking of top families |

## Selection

| Mode | Ranking evidence | When |
| --- | --- | --- |
| `cv` (default) | Train-fold CV means | Fast exploration (optimistic vs outer) |
| `nested` | Outer train folds after inner selection | Stronger post-selection estimate |
| `validation` | Session validation partition | Needs `validation_size` |

```python
result = session.automl.run(n_trials=12, cv=3, selection="cv", random_state=0)
board = result.leaderboard()
print(board.head())
```

## Leakage

```python
from buildml.core.errors import LeakageError

session = (
    Session.ingest(frame)
    .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
    .split(test_size=0.25, stratify=True, random_state=0)
    .scale(method="standard")
)

try:
    session.automl.run(n_trials=6, cv=3)
except LeakageError as exc:
    print(exc)
```

Prefer ingest → roles → split → `session.automl.run(include_recipe_search=True)`
on unpoisoned data. `allow_session_global_preprocess=True` is a known-biased
override.

Fold-local recipe strategies refit on fold-train during CV ranking. The
final refit fits the winning recipe on full train and stores a sklearn
`Pipeline` when preprocess is non-empty.

## Catalogs and budgets

Default classification families: `logistic`, `random_forest`,
`gradient_boosting`, `knn`, `decision_tree`.

Default regression families: `ridge`, `lasso`, `random_forest`,
`gradient_boosting`, `knn`, `decision_tree`.

Recipe strategies include passthrough, impute-only, impute+scale,
one-hot/ordinal encode, select, and combinations. Cap with `n_trials`,
`time_budget`, `families=...`, and `AutoMLBudget(...)`.

When `buildml[automl-industry]` is installed, native search can include
LightGBM, XGBoost, and CatBoost (`include_industry_families=True`).

`include_ensembles=True` scores voting and/or stacking of diverse top
families under a shared recipe. That is not a substitute for
`session.ensemble.fit_stacking` when you want CV OOF meta features.

## Bundle

```python
session.automl.run(n_trials=10, cv=3, random_state=0)
session.automl.save_bundle("artifacts/automl_bundle")
session.save_pipeline("artifacts/automl_pipeline", evaluate_partition="test")
```

`buildml.automl_bundle.v1` stores the AutoMLPlan and FitResult contract,
not the dataset. `checkpoint_load` does not restore the AutoMLPlan.

## What usually goes wrong

- Session-global prep without the allow flag: `LeakageError`.
- `selection='validation'` without a validation partition: `ValidationError`.
- Treating train-CV ranks as final generalization.
- Claiming NAS or causality from this API.
- Industry adapters do not support nested CV or fold-local recipes.
