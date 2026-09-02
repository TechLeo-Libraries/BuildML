# Leakage, recipes, and honest CV

```bash
pip install buildml
```

Two different leaks show up in a Session.

**Partition leakage** is when validation or test rows help compute a
median, a vocabulary, or a scale. Fit-capable Session steps refuse
without a split and without train scope.

**Fold leakage** is quieter. You call `session.impute()` on the whole
training partition, then `cv_score`. Every fold's "held-out" rows
already carry statistics that saw their peers. The mean F1 looks fine.
The protocol is broken.

`PreprocessRecipe` exists for the second leak. It is an *unfitted*
description. Inside `cv_score`, `grid_search`, `randomized_search`,
`optuna_search`, `evolutionary_search`, and `nested_cv_score`, BuildML
refits those steps on each fold's training rows and applies the frozen
fold plans to that fold's eval rows.

If Session-global `impute` / `encode` / `scale` / `handle_outliers` /
`select_features` (or Session-global dates, text, or reduce) already ran
on the full train partition, CV and search refuse **even when you pass a
recipe**. The recipe sees the already-transformed frame. It cannot
rebuild from raw cells.

Related: [concepts](../docs/concepts.rst),
[classical quickstart](quickstart-classical.md),
[diagnostics and search](classical-diagnostics-search.md).
Paste: [`examples/leakage_cv_recipe.py`](../examples/leakage_cv_recipe.py).

## What a recipe can and cannot do

Fold-local order, when those steps are set, is:

dates → text → outliers → impute → encode → binning → scale → reduce → select

Outliers inside a recipe may `detect` or `cap`. **Drop** is refused:
dropping rows would rewrite fold membership.

These stay Session-global and are never fold-local:

- `resample` (it rewrites train rows)
- `apply_custom_transform`
- Session `text_features` / `reduce_dimensions` / `extract_dates` /
  `bin` unless the same work is expressed on the recipe

`allow_session_global_preprocess=True` is an override for a known-biased
baseline. The score stays leakage-biased. Re-ingest (or
`checkpoint_load` an unpoisoned frame) if you want an honest number.

## Good: recipe on clean data

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

# Do not call session.impute() / encode() / scale() before this.
recipe = PreprocessRecipe(impute="median", encode="onehot", scale="standard")
cv = session.cv_score(
    LogisticRegression(max_iter=500),
    cv=4,
    preprocess=recipe,
)
print(cv.mean_metrics[cv.scoring_metric], "±", cv.std_metrics[cv.scoring_metric])
```

Folds are cut from **train only**. Session test is not scored here. After
you pick a setup, prepare and fit once on full train, then evaluate test
once.

`cv_strategy="auto"` reads roles and picks a splitter. Use `"group"` or
`"stratified_group"` after a `group` role, `"time"` after a `time` role,
`"stratified"` when class mix must hold in every fold. The wrong
strategy recreates the leak the split was meant to stop. Group CV
without a `group` role fails clearly.

## Bad: Session-global prep, then CV

```python
session.impute(strategy="median")
session.scale(method="standard")

# LeakageError: the frame is already poisoned for fold-local CV.
try:
    session.cv_score(
        LogisticRegression(max_iter=500),
        cv=4,
        preprocess=PreprocessRecipe(impute="median", scale="standard"),
    )
except Exception as exc:
    print(type(exc).__name__, exc)

# Override: biased scores. Do not treat this as honest CV.
biased = session.cv_score(
    LogisticRegression(max_iter=500),
    cv=4,
    preprocess=PreprocessRecipe(impute="median", scale="standard"),
    allow_session_global_preprocess=True,
)
print("biased override:", biased.mean_metrics)
```

## Nested CV with recipe knobs

`grid_search` reports the winner's inner score. That number is optimistic:
you picked the luckiest configuration and then quoted its luck. Nested
CV gives the search its own private data. Outer folds score a winner the
inner search never saw.

Only knobs in `SAFE_RECIPE_KNOBS` may be swept (`select_k`, `n_bins`,
`min_frequency`, `iqr_multiplier`, and the rest of that set). Strategy
enums (`impute`, `scale`, `encode`) stay on the base recipe.

```python
from sklearn.tree import DecisionTreeClassifier

from buildml.preprocess import PreprocessRecipe, SAFE_RECIPE_KNOBS

nested = session.nested_cv_score(
    DecisionTreeClassifier(random_state=0),
    param_grid={"max_depth": [2, 4], "min_samples_leaf": [1, 5]},
    recipe_grid={"select_k": [2, 3]},
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

## Target encoding

`encode="target"` inside a recipe fits smoothed means on **fold-train
labels only**. Eval rows never contribute. Session-global
`session.encode(method="target")` fits on full train: fine for a final
model after the split, poison for a later `cv_score`.

```python
cv = session.cv_score(
    LogisticRegression(max_iter=500),
    cv=4,
    preprocess=PreprocessRecipe(impute="median", encode="target", scale="standard"),
)
```

## Weights

Assign at most one `weight` column. Weights are not features. They are
left out of the design matrix and passed as `sample_weight` when the
estimator accepts it. An estimator that cannot take weights raises
`ValidationError` instead of silently ignoring the column. Non-positive
or all-NaN weights also raise. A column cannot be both `weight` and
`feature`.

```python
session = (
    Session.ingest(frame)
    .set_roles({"x": "feature", "w": "weight", "y": "target"})
    .split(test_size=0.25, stratify=True, random_state=0)
    .fit(LogisticRegression(max_iter=500), task="classification")
)
print(session.evaluate(partition="test").diagnostics.get("sample_weight_column"))
```

## Outliers inside recipes

```python
recipe = PreprocessRecipe(
    outliers="iqr",
    outlier_action="cap",
    impute="median",
    scale="standard",
)
```

Session-global `handle_outliers(..., action="drop")` rebuilds splits
after dropping train rows. Do not then claim honest CV on that frame
without re-ingest.

## A selection order that stays honest

1. Ingest → roles → split (or group / time / inject).
2. `cv_score` / search / nested **before** Session-global prep, with a
   `PreprocessRecipe`.
3. `compare_models(..., partition="validation")` if you are shortlisting.
4. Prepare and fit the final estimator on full train.
5. Thresholds and calibration on validation.
6. Test once.

[Preprocess depth](preprocess-depth.md) ·
[Diagnostics and search](classical-diagnostics-search.md) ·
[Classical end-to-end](classical-end-to-end.md)
