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

Related: [unsupervised](unsupervised-deep.md), [ensembles](ensemble-deep.md),
[forecasting](forecasting-deep.md), [anomaly](anomaly-deep.md). Explicit
non-goals (neuromorphic, swarm zoo, digital twins, AV/robotics, TTS, full COCO
suite) stay out.

---

## AutoML vs `grid_search` / `optuna_search` / `evolutionary_search`

| Concern | Single-estimator search | `session.automl.run` |
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

**Default is `selection='cv'`** (train-fold ranking). That is intentional for
fast exploration and is disclosed on the result / leaderboard. For
post-selection claims, prefer the prominent **`selection='nested'`** path
(outer mean±std) or `selection='validation'`, then confirm once on Session test.

| Mode | Ranking evidence | When to prefer |
| --- | --- | --- |
| `cv` (**default**) | Train-fold CV means | Fast exploration (optimistic vs outer) |
| `nested` (**prominent honesty path**) | Outer train folds after inner selection | Stronger post-selection estimate |
| `validation` | Session validation partition | Explicit holdout ranking (needs `validation_size`) |

In every mode, **Session test stays out of selection**. Confirm once with
`session.automl.evaluate(partition='test')` after freezing the winner.

### Leaderboard reporting

```python
result = session.automl.run(n_trials=12, cv=3, selection="cv", random_state=0)
board = result.leaderboard()  # also result.to_frame()
# columns include rank, family, recipe_strategy, mean_score, gap_to_best,
# selection, outer_score_mean, nested_cv_disclosed, ranking_metric, param_*
print(board.head())

from buildml.automl import export_comparison_metrics
export_comparison_metrics(result, "artifacts/automl_trials.json")
```

Catalog honesty: `automl_capability_matrix()["default_selection"] == "cv"`,
`selection_modes["nested"]["prominent"] is True`, and industry
`available` flags use subprocess import probes (not find_spec alone).

---

## Leakage contract

```python
from buildml import Session
from buildml.core.errors import LeakageError

session = (
    Session.ingest(frame)
    .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
    .split(test_size=0.25, stratify=True, random_state=0)
    .scale(method="standard")  # Session-global: poisons fold honesty
)

try:
    session.automl.run(n_trials=6, cv=3)
except LeakageError as exc:
    print(exc)
```

Same refusal as classical `cv_score` / `grid_search`. Prefer:

1. ingest → roles → split → `session.automl.run(include_recipe_search=True)` on
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
This is not a substitute for native `session.ensemble.fit_stacking` / `session.ensemble.fit_blending` when you
want CV OOF meta features or an explicit train-inner blend holdout.

---

## Bundles and pipelines

```python
session.automl.run(n_trials=10, cv=3, random_state=0)
session.automl.save_bundle("artifacts/automl_bundle")
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

## Known limits

- No dedicated AutoML dashboard charts (use classical plot boards / evaluate)
- Industry adapters (FLAML/AutoGluon) do not support nested CV or fold-local recipes
- Full AutoGluon multi-modal / multimodel export not wrapped: tabular TabularPredictor only
- Stacking inside AutoML uses sklearn Stacking* with fixed meta-estimators (not full native `session.ensemble.fit_stacking` OOF path)
- Catalog deliberately omits deep nets and arbitrary Pipeline DAGs
- Benchmark: `python benchmarks/automl/tabular_search.py` (skips unavailable backends)
