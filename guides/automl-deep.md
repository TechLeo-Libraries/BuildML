# AutoML (deep)

> **Install (GitHub 2.x):**
> `pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"`
> Randomized/grid/evolutionary AutoML is core. Optuna backend needs
> `buildml[automl]`. Industry adapters need `buildml[automl-industry]`.
> See [installation](../docs/installation.rst).

This guide covers Session AutoML: joint model-family and fold-local preprocess
strategy search beyond single-estimator HPO, with leakage discipline, nested /
validation selection, optional voting ensembles, classical evaluation, and
`buildml.automl_bundle.v1`. It matches the depth bar of classical / Torch /
RAG / unsupervised / ensemble guides.

**Related:** [Quickstart](quickstart-automl.md) ·
[Leakage](leakage-cv-recipes.md) ·
[Diagnostics & search](classical-diagnostics-search.md) ·
[Ensembles](ensemble-deep.md) ·
[Artifacts](artifacts-checkpoints-bundles.md)

---

## What this path is (and is not)

| Is | Is not |
| --- | --- |
| Finite disclosed catalogs of families + recipe strategies | Neural architecture search (NAS) |
| Fold-local `PreprocessRecipe` strategy search | Session-global prep as “safe CV” |
| `cv` / `nested` / `validation` selection (test held out) | Using Session test to pick winners |
| Optional voting of diverse top families | Full stacking/blending AutoML zoo |
| AutoML bundle + classical pipeline compatibility | Session checkpoint substitute |
| Predictive ranking under a trial budget | Causal discovery / automated science |

Phase 1 depth-first order (**complete**): unsupervised → ensembles →
**AutoML (this guide)** → forecasting → anomaly (see
[Anomaly deep](anomaly-deep.md)). Explicit non-goals (neuromorphic, swarm zoo,
digital twins, AV/robotics, TTS, full COCO suite) stay out.

---

## AutoML vs `grid_search` / `optuna_search` / `evolutionary_search`

| Concern | Single-estimator search | `run_automl` |
| --- | --- | --- |
| Estimator | One fixed model you chose | Catalog of families (+ industry GBDT when installed) |
| Preprocess | Optional knobs on one recipe | Discrete strategy search (impute/scale/encode/select) |
| Backends | Optuna for `optuna_search`; GA for `evolutionary_search` | `native`, `optuna`, `flaml`, `autogluon` |
| Ensembles | Bring your own | Optional voting/stacking of top families |
| Honesty | Train-only CV | Same + nested / validation modes |

Use single-estimator search when the model family is already decided. Use
AutoML when family and preprocess strategy are part of the decision.

### Capability matrix

```python
from buildml.automl import automl_capability_matrix

matrix = automl_capability_matrix()
# matrix["backends"]["flaml"]["available"]  # True when buildml[automl-industry] installed
```

Industry backends (`flaml`, `autogluon`) fit on **train only** and disclose
that fold-local recipe search is bypassed. Nested selection is **native-only**.

---

## Selection modes

| Mode | Ranking evidence | When to prefer |
| --- | --- | --- |
| `cv` | Train-fold CV means | Fast exploration |
| `nested` | Outer train folds after inner selection | Stronger post-selection estimate |
| `validation` | Session validation partition | Explicit holdout ranking (needs `validation_size`) |

In every mode, **Session test stays out of selection**. Confirm once with
`evaluate_automl(partition='test')` after freezing the winner.

---

## Leakage contract

```python
from buildml import Session
from buildml.core.errors import LeakageError

session = (
    Session.ingest(frame)
    .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
    .split(test_size=0.25, stratify=True, random_state=0)
    .scale(method="standard")  # Session-global — poisons fold honesty
)

try:
    session.run_automl(n_trials=6, cv=3)
except LeakageError as exc:
    print(exc)
```

Same refusal as classical `cv_score` / `grid_search`. Prefer:

1. ingest → roles → split → `run_automl(include_recipe_search=True)` on
   unpoisoned data, **or**
2. re-ingest / checkpoint-load an unpoisoned frame, **or**
3. `allow_session_global_preprocess=True` with eyes open (biased scores).

Fold-local recipe strategies refit on fold-train only during CV ranking.
Final refit fits the winning recipe on full train and stores a sklearn
`Pipeline` when preprocess is non-empty.

---

## Catalogs and budgets

Default classification families: `logistic`, `random_forest`,
`gradient_boosting`, `knn`, `decision_tree`.

Default regression families: `ridge`, `lasso`, `random_forest`,
`gradient_boosting`, `knn`, `decision_tree`.

Default recipe strategies include passthrough, impute-only, impute+scale,
one-hot/ordinal encode variants, select (univariate / variance), and
expanded combinations (13 strategies).

Cap exploration with `n_trials`, `time_budget`, `families=...`, and
`AutoMLBudget(max_trials=..., max_families=..., max_recipe_strategies=...,
max_time_seconds=..., study_storage=..., enable_pruning=...)`.

When `buildml[automl-industry]` is installed, native search also includes
LightGBM, XGBoost, and CatBoost families (`include_industry_families=True`).

---

## Optional ensembles inside AutoML

When `include_ensembles=True`, AutoML scores **voting** and/or **stacking**
ensembles (`ensemble_mode='voting'|'stacking'|'both'`) built from diverse
top single-model families under a shared recipe.
This is not a substitute for native `fit_stacking` / `fit_blending` when you
want CV OOF meta features or an explicit train-inner blend holdout.

---

## Bundles and pipelines

```python
session.run_automl(n_trials=10, cv=3, random_state=0)
session.save_automl_bundle("artifacts/automl_bundle")
session.save_pipeline("artifacts/automl_pipeline", evaluate_partition="test")
```

| Artifact | Contains | Does not |
| --- | --- | --- |
| `buildml.automl_bundle.v1` | AutoMLPlan + FitResult contract | Dataset, Session-global plans |
| Session checkpoint | data/roles/splits/history | AutoMLPlan |
| Classical pipeline | Session-global plans + estimator | AutoML search disclosures |

---

## Failure modes

- Session-global prep before AutoML without the allow flag → `LeakageError`
- `selection='validation'` without a validation partition → `ValidationError`
- Treating train-CV AutoML ranks as final generalization
- Claiming NAS / causality / automated science from this API
- Expecting `checkpoint_load` to restore `AutoMLPlan`

---

## Non-blocking residuals

- No dedicated AutoML dashboard charts (use classical plot boards / evaluate)
- Industry adapters (FLAML/AutoGluon) do not support nested CV or fold-local recipes
- Full AutoGluon multi-modal / multimodel export not wrapped — tabular TabularPredictor only
- Stacking inside AutoML uses sklearn Stacking* with fixed meta-estimators (not full native `fit_stacking` OOF path)
- Catalog deliberately omits deep nets and arbitrary Pipeline DAGs
- Benchmark: `python benchmarks/automl/tabular_search.py` (skips unavailable backends)
