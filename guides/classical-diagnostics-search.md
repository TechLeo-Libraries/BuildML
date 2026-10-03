# Classical diagnostics and model search

```bash
pip install buildml
# plots: pip install "buildml[viz]"
# optuna_search: pip install "buildml[optuna]"
```

After a split and a fit, these calls inspect the model and choose among
estimators without putting Session test inside an inner loop.

`compare_models` ranks on **test** unless you pass `partition="validation"`.
The highest-ranked candidate becomes the Session's fitted model. `cv_score` and search cut
folds from train only. Session-global prep before those calls is refused
([leakage and recipes](leakage-cv-recipes.md)).

Use validation to select thresholds, features, and model families.
Use test to assess the final choice. BuildML cannot stop you from using test results to select a model in your
own notebook. It can refuse CV after Session-global fitted preprocessing.

[Classical end-to-end](classical-end-to-end.md) ·
Runnable script: [`examples/evolutionary_search_loop.py`](../examples/evolutionary_search_loop.py)

---

## Baseline model example

```python
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from buildml import Session
from buildml.preprocess import PreprocessRecipe

X, y = make_classification(n_samples=120, n_features=4, n_informative=3,
                           n_redundant=0, random_state=0)
frame = pd.DataFrame(X, columns=["a", "b", "c", "d"])
frame["seg"] = ["A" if i % 2 else "B" for i in range(len(frame))]
frame["y"] = y
session = (Session.ingest(frame)
    .set_roles({**{c: "feature" for c in frame if c != "y"}, "y": "target"})
    .split(test_size=0.25, validation_size=0.25, stratify=True, random_state=0))
recipe = PreprocessRecipe(encode="onehot", scale="standard")

```

---

## Use case: compare_models on validation

```python
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from buildml import Session
from buildml.preprocess import PreprocessRecipe

X, y = make_classification(n_samples=120, n_features=4, n_informative=3,
                           n_redundant=0, random_state=0)
frame = pd.DataFrame(X, columns=["a", "b", "c", "d"])
frame["seg"] = ["A" if i % 2 else "B" for i in range(len(frame))]
frame["y"] = y
session = (Session.ingest(frame)
    .set_roles({**{c: "feature" for c in frame if c != "y"}, "y": "target"})
    .split(test_size=0.25, validation_size=0.25, stratify=True, random_state=0))
recipe = PreprocessRecipe(encode="onehot", scale="standard")

session.encode(method="onehot").scale(method="standard")
comparison = session.compare_models(
    {
        "logreg": LogisticRegression(max_iter=500),
        "tree": DecisionTreeClassifier(max_depth=3, random_state=0),
        "rf": RandomForestClassifier(n_estimators=50, random_state=0),
    },
    partition="validation",  # override default "test" during selection
    ranking_metric="f1_macro",
)
print(comparison)
# Winner becomes session.fit_result
```

Set `partition="validation"` explicitly while comparing candidates.
Evaluate the chosen model on test after model and preprocessing choices
are fixed.

---

## Use case: grid, randomized, Optuna, and evolutionary search

Folds stay inside train. Pass the recipe so impute, encode, and scale
refit per fold. Keep the Session data unprocessed so each fold fits its own imputer.

```python
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from buildml import Session
from buildml.preprocess import PreprocessRecipe

X, y = make_classification(n_samples=120, n_features=4, n_informative=3,
                           n_redundant=0, random_state=0)
frame = pd.DataFrame(X, columns=["a", "b", "c", "d"])
frame["seg"] = ["A" if i % 2 else "B" for i in range(len(frame))]
frame["y"] = y
session = (Session.ingest(frame)
    .set_roles({**{c: "feature" for c in frame if c != "y"}, "y": "target"})
    .split(test_size=0.25, validation_size=0.25, stratify=True, random_state=0))
recipe = PreprocessRecipe(encode="onehot", scale="standard")

# Fold-local prep: do not Session-impute first
grid = session.grid_search(
    DecisionTreeClassifier(random_state=0),
    param_grid={"max_depth": [2, 4, 6], "min_samples_leaf": [1, 3, 5]},
    cv=4,
    preprocess=recipe,
    ranking_metric="f1_macro",
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
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from buildml import Session
from buildml.preprocess import PreprocessRecipe

X, y = make_classification(n_samples=120, n_features=4, n_informative=3,
                           n_redundant=0, random_state=0)
frame = pd.DataFrame(X, columns=["a", "b", "c", "d"])
frame["seg"] = ["A" if i % 2 else "B" for i in range(len(frame))]
frame["y"] = y
session = (Session.ingest(frame)
    .set_roles({**{c: "feature" for c in frame if c != "y"}, "y": "target"})
    .split(test_size=0.25, validation_size=0.25, stratify=True, random_state=0))
recipe = PreprocessRecipe(encode="onehot", scale="standard")

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
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from buildml import Session
from buildml.preprocess import PreprocessRecipe

X, y = make_classification(n_samples=120, n_features=4, n_informative=3,
                           n_redundant=0, random_state=0)
frame = pd.DataFrame(X, columns=["a", "b", "c", "d"])
frame["seg"] = ["A" if i % 2 else "B" for i in range(len(frame))]
frame["y"] = y
session = (Session.ingest(frame)
    .set_roles({**{c: "feature" for c in frame if c != "y"}, "y": "target"})
    .split(test_size=0.25, validation_size=0.25, stratify=True, random_state=0))
recipe = PreprocessRecipe(encode="onehot", scale="standard")

# Final fit after selection (Session-global prep OK here)
session.encode(method="onehot").scale(method="standard")
session.fit(LogisticRegression(max_iter=500), task="classification")

session.calibration()  # default partition is validation
session.tune_threshold(fp_cost=1.0, fn_cost=5.0)
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

Permutation importance measures model reliance, not causal effect. Select
thresholds on validation. Confirm the fixed policy on test.

`evaluate(...)` returns metrics and diagnostics. `eval_plots(...)`
requires `buildml[viz]` and produces task-specific plots.

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
