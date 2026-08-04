# Classical diagnostics and model search

> **Install:**
> `pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"`
> Optional: `pip install "buildml[viz]"` for plot boards,
> `"buildml[optuna]"` for `optuna_search`.
> See [installation](../docs/installation.rst).

After a honest split and fit, BuildML helps you **inspect** models (calibration,
thresholds, importance, slices) and **select** among estimators/hyperparameters
without scoring Session test inside inner loops.

Related: [leakage-cv-recipes](leakage-cv-recipes.md),
[classical-end-to-end](classical-end-to-end.md).

---

## Why validation vs test

- **Validation:** iterate thresholds, features, model families, early stops.
- **Test:** score the frozen decision policy once.
- **CV / search:** folds stay inside **train**; Session test is reserved.

Violating that protocol is the most common way “great offline metrics” fail in
production: and BuildML cannot stop you from peeking at test in your own code.
It *can* refuse Session-global prep poisoning of CV
([leakage guide](leakage-cv-recipes.md)).

---

## Setup shared by examples

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from buildml import Session
from buildml.preprocess import PreprocessRecipe

frame = pd.DataFrame(
    {
        "a": [0.1, 0.4, 0.2, 0.8, 0.3, 0.7, 0.5, 0.9, 0.15, 0.65, 0.55, 0.35],
        "b": [1.0, 0.2, 0.9, 0.1, 0.8, 0.3, 0.6, 0.4, 0.75, 0.25, 0.55, 0.45],
        "seg": list("AABBAABBAABB"),
        "y": [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0],
    }
)

session = (
    Session.ingest(frame)
    .set_roles({"a": "feature", "b": "feature", "seg": "feature", "y": "target"})
    .split(test_size=0.25, validation_size=0.25, stratify=True, random_state=0)
)

recipe = PreprocessRecipe(encode="onehot", scale="standard")
```

---

## Use case: compare_models on validation

```python
comparison = session.compare_models(
    {
        "logreg": LogisticRegression(max_iter=500),
        "tree": DecisionTreeClassifier(max_depth=3, random_state=0),
        "rf": RandomForestClassifier(n_estimators=50, random_state=0),
    },
    partition="validation",  # override default "test" during selection
    ranking_metric="f1",
)
print(comparison)
# Winner becomes session.fit_result
```

Default `partition="test"` is convenient for a final card: dangerous during
iterative selection. Prefer validation until the recipe is frozen.

---

## Use case: grid, randomized, Optuna, and evolutionary search

```python
# Fold-local prep: do not Session-impute first
grid = session.grid_search(
    DecisionTreeClassifier(random_state=0),
    param_grid={"max_depth": [2, 4, 6], "min_samples_leaf": [1, 3, 5]},
    cv=4,
    preprocess=recipe,
    ranking_metric="f1",
)
print(grid.best_params, grid.best_score)

rand = session.randomized_search(
    DecisionTreeClassifier(random_state=0),
    param_distributions={"max_depth": [2, 3, 4, 5, 6], "min_samples_leaf": [1, 2, 3, 5]},
    n_iter=6,
    cv=3,
    preprocess=recipe,
)
print(rand.best_params)

# In-tree NumPy GA (no extra). HPO backend: not neuroevolution / NAS.
evo = session.evolutionary_search(
    DecisionTreeClassifier(random_state=0),
    param_space={
        "max_depth": {"type": "int", "low": 2, "high": 8},
        "min_samples_leaf": [1, 2, 3, 5],
    },
    population_size=8,
    n_generations=4,
    cv=3,
    preprocess=recipe,
    random_state=0,
)
print(evo.best_params, evo.best_score)
# Generation history: evo.study["generation_best"]

# Optional: pip install "buildml[optuna]"
# opt = session.optuna_search(
#     DecisionTreeClassifier(random_state=0),
#     param_space={"max_depth": {"type": "int", "low": 2, "high": 8}},
#     n_trials=12,
#     cv=3,
#     preprocess=recipe,
# )
```

---

## Use case: nested CV for post-selection estimate

```python
nested = session.nested_cv_score(
    DecisionTreeClassifier(random_state=0),
    param_grid={"max_depth": [2, 4], "min_samples_leaf": [1, 5]},
    outer_cv=3,
    inner_cv=3,
    preprocess=recipe,
)
print(
    nested.mean_metrics[nested.scoring_metric],
    "±",
    nested.std_metrics[nested.scoring_metric],
)
```

---

## Use case: calibration, thresholds, importance, slices

```python
# Final fit after selection (Session-global prep OK here)
session.encode(method="onehot").scale(method="standard")
session.fit(LogisticRegression(max_iter=500), task="classification")

session.calibration(partition="validation")
session.tune_threshold(partition="validation", fp_cost=1.0, fn_cost=5.0)
# Persist the same operating point as a DecisionPlan (see guides/quickstart-optimize.md):
# session.decision.fit(method="threshold", partition="validation", fp_cost=1.0, fn_cost=5.0)
session.feature_importance(partition="validation", n_repeats=8)
session.learning_curve(
    LogisticRegression(max_iter=500),
    cv=3,
)
# If a segment column exists on the frame:
# session.error_slices(partition="validation", by="seg")

# Plot boards (buildml[viz]):
# session.eval_plots(partition="validation", export_html="artifacts/plots.html")
```

Permutation importance measures **model reliance**, not causal effect. Select
thresholds on validation; confirm the fixed policy on test.

---

## evaluate vs eval_plots

| API | Role |
| --- | --- |
| `evaluate(...)` | Metrics + diagnostics + optional plot hooks / HTML |
| `eval_plots(...)` | Adaptive PlotBoard panels (needs `[viz]`) |

Use both when you want a metric card **and** a teaching-oriented board.

---

## Failure modes

| Issue | Fix |
| --- | --- |
| Search after Session prep | Re-ingest or `allow_session_global_preprocess=True` (biased) |
| Ranking on test during iteration | Use `partition="validation"` |
| Optuna missing | Install `buildml[optuna]` |
| Importance as causality | Do not: report as reliance only |

---

## Related

- [Leakage & recipes](leakage-cv-recipes.md)
- [Artifacts](artifacts-checkpoints-bundles.md)
- [EDA / Teaching Studio](eda-teaching-studio.md)
