# Leakage, fold-local recipes, weights, and hard-refuse CV

> **Install:** PyPI `buildml` is legacy 1.x. Use
> `pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"`.
> See [installation](../docs/installation.rst).

Cross-validation is honest only when **every** statistic that should be
“training-only” is re-learned inside each fold. BuildML hard-refuses common
poisoning patterns so a nice mean F1 does not hide a broken protocol.

Related: [concepts](../docs/concepts.rst),
[classical end-to-end](classical-end-to-end.md),
[diagnostics & search](classical-diagnostics-search.md),
[glossary](glossary.md).

---

## Conceptual why (in depth)

### Partition leakage vs fold leakage

- **Partition leakage:** validation/test rows influence medians, encodings, or
  scales used by the model. BuildML blocks fit-capable Session ops without a
  split and without train scope (`LeakageError` / `assert_can_fit`).
- **Fold leakage:** you call `session.impute()` on the full train partition,
  then run `cv_score`. Every fold’s “held-out” rows already carry statistics
  computed with their peers. Scores are optimistically biased.

### Why recipes exist

`PreprocessRecipe` describes **unfitted** steps. Inside `cv_score` /
`grid_search` / `randomized_search` / `optuna_search` /
`evolutionary_search` / `nested_cv_score`,
BuildML refits those steps on each fold’s training rows and applies the frozen
fold plans to the fold’s eval rows. That is fold-local honesty.

### What can never be fold-local

From `SESSION_GLOBAL_ONLY_STEPS` / library policy:

- `resample` (rewrites train rows)
- `apply_custom_transform` (registered callables stay Session-global)
- Any Session-global plan already fitted on the full train partition before CV

### Hard refuse even with a recipe

If you already ran Session-global `impute` / `encode` / `scale` / … on the
frame, CV/search **refuse by default even when you pass a `PreprocessRecipe`**.
Recipes do not undo poisoned cells. Options:

1. Re-ingest clean data and use only the recipe inside CV (preferred).
2. Explicit override: `allow_session_global_preprocess=True` (scores remain
   biased: use only when you understand the contamination).

---

## Good example: fold-local CV on clean data

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.preprocess import PreprocessRecipe

frame = pd.DataFrame(
    {
        "age": [21, None, 35, 40, 29, 33, 52, 47, 31, 44, 38, 27],
        "income": [40, 55, 60, 80, 50, 70, 90, 65, 48, 88, 61, 72],
        "city": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "C", "B", "A"],
        "approved": [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0],
    }
)

session = (
    Session.ingest(frame)
    .set_roles(
        {
            "age": "feature",
            "income": "feature",
            "city": "feature",
            "approved": "target",
        }
    )
    .split(test_size=0.25, stratify=True, random_state=42)
)

# Do NOT call session.impute()/encode()/scale() before CV.
recipe = PreprocessRecipe(impute="median", encode="onehot", scale="standard")
cv = session.cv_score(
    LogisticRegression(max_iter=500),
    cv=4,
    preprocess=recipe,
)
print(cv.mean_metrics[cv.scoring_metric], "±", cv.std_metrics[cv.scoring_metric])
```

Session **test** is never scored inside CV folds. After selection, fit once on
full train (with Session prep or a final recipe path) and evaluate test once.

---

## Bad example: Session-global prep then CV (refused)

```python
session.impute(strategy="median")
session.scale(method="standard")

# Raises LeakageError by default: frame already poisoned for fold-local CV.
try:
    session.cv_score(
        LogisticRegression(max_iter=500),
        cv=4,
        preprocess=PreprocessRecipe(impute="median", scale="standard"),
    )
except Exception as exc:  # LeakageError
    print(type(exc).__name__, exc)

# Explicit override: biased scores; do not treat as honest CV.
biased = session.cv_score(
    LogisticRegression(max_iter=500),
    cv=4,
    preprocess=PreprocessRecipe(impute="median", scale="standard"),
    allow_session_global_preprocess=True,
)
print("biased override:", biased.mean_metrics)
```

---

## Good example: nested CV with recipe knobs

```python
from sklearn.tree import DecisionTreeClassifier

from buildml.preprocess import PreprocessRecipe, SAFE_RECIPE_KNOBS

print(sorted(SAFE_RECIPE_KNOBS)[:8], "...")

nested = session.nested_cv_score(
    DecisionTreeClassifier(random_state=0),
    param_grid={"max_depth": [2, 4], "min_samples_leaf": [1, 5]},
    recipe_grid={"select_k": [2, 3]},  # only SAFE_RECIPE_KNOBS
    preprocess=PreprocessRecipe(
        impute="median",
        encode="onehot",
        scale="standard",
        select="univariate",
        select_k=3,
    ),
    outer_cv=3,
    inner_cv=3,
)
print(nested.mean_metrics[nested.scoring_metric])
```

Inner search picks hyperparameters **and** safe recipe knobs; outer folds
estimate generalization of that selection process. Do not use Session test
inside nested loops.

---

## Bad example: target encoding without fold locality

Target (mean) encoding must never see fold-eval labels. Inside a recipe,
`encode="target"` fits smoothed means on **fold-train labels only**:

```python
# Good: fold-local target encoding
cv = session.cv_score(
    LogisticRegression(max_iter=500),
    cv=4,
    preprocess=PreprocessRecipe(impute="median", encode="target", scale="standard"),
)

# Risky Session-global path: session.encode(method="target") fits on full train.
# Fine for a final model after split; poison for subsequent cv_score without override.
```

---

## Weight role (`ColumnRole.WEIGHT`)

Assign at most one `weight` column. Weights are **not** features: they are
excluded from the design matrix and passed as `sample_weight` when the
estimator supports it.

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression

from buildml import Session

frame = pd.DataFrame(
    {
        "x": [0.1, 0.4, 0.2, 0.8, 0.3, 0.7, 0.5, 0.9],
        "w": [1.0, 1.0, 2.0, 1.0, 1.5, 1.0, 2.0, 1.0],
        "y": [0, 1, 0, 1, 0, 1, 1, 0],
    }
)

session = (
    Session.ingest(frame)
    .set_roles({"x": "feature", "w": "weight", "y": "target"})
    .split(test_size=0.25, stratify=True, random_state=0)
    .fit(LogisticRegression(max_iter=500), task="classification")
)

result = session.evaluate(partition="test")
print(result.diagnostics.get("sample_weight_column"))
```

**Failure modes:**

- Estimator without `sample_weight` → `ValidationError` (not silent ignore).
- Non-positive / all-NaN weights → `ValidationError`.
- Weight also marked `feature` → validation error.

Weighted metrics apply where sklearn accepts `sample_weight`.

---

## Group / time CV strategies

```python
# After group_split / time role assignment:
cv_g = session.cv_score(
    LogisticRegression(max_iter=500),
    cv=3,
    cv_strategy="group",  # or stratified_group / time / auto
    preprocess=PreprocessRecipe(scale="standard"),
)
```

`cv_strategy="auto"` picks a sensible default from roles; override when your
protocol requires it. Group CV without a `group` role fails clearly.

---

## Outliers inside recipes

Fold-local outliers support `detect` and `cap` only. **Dropping** rows would
rewrite fold membership and is refused inside CV:

```python
recipe = PreprocessRecipe(
    outliers="iqr",
    outlier_action="cap",  # not "drop" inside CV
    impute="median",
    scale="standard",
)
```

Session-global `handle_outliers(..., action="drop")` rebuilds splits after
dropping train rows: use carefully, then avoid honest CV on that poisoned
frame without re-ingest.

---

## Checklist: honest classical selection

1. Ingest → roles → split (or group/time/inject).
2. Run `cv_score` / search / nested **before** Session-global prep, with
   `PreprocessRecipe`.
3. Optionally compare models on **validation** (`compare_models(..., partition="validation")`).
4. Fit final estimator with Session prep on full train.
5. Tune thresholds / calibration on validation.
6. Evaluate **test once**.

---

## Related

- [Preprocess depth](preprocess-depth.md)
- [Diagnostics & search](classical-diagnostics-search.md)
- [Classical end-to-end](classical-end-to-end.md)
- [Artifacts](artifacts-checkpoints-bundles.md)
