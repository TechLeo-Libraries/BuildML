# Preprocess depth

```bash
pip install buildml
# resample: pip install "buildml[imbalanced]"
```

These operations fit preprocessing parameters on the training partition
and apply the resulting plans to the other partitions. `impute`, `encode`, and `scale` with `columns=None` touch
`feature`-role columns of the matching dtype. `id`, `target`, `group`,
`time`, `weight`, and `ignore` stay as they are unless you name them.

If you need those same steps inside CV, do not call them here first. Use a
`PreprocessRecipe` on data without Session-global fitted preprocessing
([leakage and recipes](leakage-cv-recipes.md)). Session-global prep, then
CV, is refused. `resample` and `apply_custom_transform` are never
fold-local. Resample plans are lineage-only at score time: they do not
synthesize rows for inference.

---

## What a plan is for

Each step stores a serializable plan (`impute_plan`, `encode_plan`, and
the rest). `save_pipeline` ships those plans with the estimator so
score-time rows see the same frozen transforms. `apply_preprocess_plans`
replays them on a new frame.

That is why you split first. Median age, one-hot levels, and TF-IDF
vocabulary come from train. Validation and test only receive the frozen
mapping. If you skip the split, the call fails.

The default methods are `impute(strategy="median")`,
`encode(method="onehot")`. Target encoding uses training labels. Review the training-fold settings
and category support when fitting a final model. For CV, put `encode="target"` inside
`PreprocessRecipe` so means refit per fold.

---

## Use case: mixed numeric + categorical + dates + text

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression

from buildml import Session

frame = pd.DataFrame(
    {
        "signup": pd.to_datetime(
            [
                "2023-01-01",
                "2023-02-15",
                "2023-03-10",
                "2023-04-01",
                "2023-05-20",
                "2023-06-11",
                "2023-07-04",
                "2023-08-19",
            ]
        ),
        "age": [21, None, 35, 40, 29, 33, 52, 47],
        "segment": ["gold", "silver", "gold", "bronze", "silver", "gold", "bronze", "silver"],
        "note": [
            "late payment risk",
            "loyal customer",
            "new account",
            "chargeback history",
            "payroll deposit",
            "travel spend",
            "student",
            "payroll deposit",
        ],
        "approved": [0, 1, 0, 1, 0, 1, 1, 0],
    }
)

session = (
    Session.ingest(frame)
    .set_roles(
        {
            "signup": "feature",
            "age": "feature",
            "segment": "feature",
            "note": "feature",
            "approved": "target",
        }
    )
    .split(test_size=0.25, stratify=True, random_state=0)
)

session.extract_dates(include_time=False, drop_original=True)
session.impute(strategy="median")
session.encode(method="onehot", columns=["segment"])
session.text_features(method="tfidf", columns=["note"], max_features=32, ngram_range=(1, 2))
session.scale(method="standard")
session.fit(LogisticRegression(max_iter=800), task="classification")
print(session.evaluate(partition="test").metrics)
```

---

## Encoding methods

| Method | Behavior | When |
| --- | --- | --- |
| `onehot` | Dense/sparse one-hot from train levels | Low-cardinality categories |
| `ordinal` | Ordered integer codes | Categories with a meaningful order; numeric codes impose ordering |
| `infrequent` | Pool rare train levels then one-hot | Long-tail categoricals |
| `target` | Smoothed target means on train (OOF-style on train) | High-cardinality with care |

```python
import pandas as pd
from buildml import Session

frame = pd.DataFrame({"segment": ["gold", "silver", "bronze", "gold"] * 20,
                      "approved": [0, 1, 0, 1] * 20})
for method in ("infrequent", "target"):
    session = (Session.ingest(frame)
        .set_roles({"segment": "feature", "approved": "target"})
        .split(test_size=0.25, stratify=True, random_state=0))
    if method == "infrequent":
        session.encode(method=method, min_frequency=0.2)
    else:
        session.encode(method=method, smoothing=10.0, n_folds=5, random_state=0)
    print(method, session.to_pandas().head())
```

---

## Outliers, binning, selection, PCA

```python
import pandas as pd
from sklearn.datasets import make_classification
from buildml import Session

X, y = make_classification(n_samples=160, n_features=6, n_informative=4,
                           weights=[0.8, 0.2], random_state=0)
frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(6)])
frame["target"] = y
session = (Session.ingest(frame)
    .set_roles({**{c: "feature" for c in frame if c != "target"}, "target": "target"})
    .split(test_size=0.25, stratify=True, random_state=0))

session.handle_outliers(method="iqr", action="cap")
session.bin(strategy="quantile", n_bins=4, encode_as="ordinal")
session.select_features(strategy="univariate", k=4)
session.reduce_dimensions(method="pca", n_components=3, prefix="pc")
```

`action="drop"` on outliers rebuilds splits after removing train rows.
Feature selection and PCA fit on train only. Prefer expressing bin,
select, and reduce inside `PreprocessRecipe` when those steps participate
in CV or search parameters (`SAFE_RECIPE_KNOBS`).

---

## Custom transforms (Session-global only)

```python
import numpy as np
import pandas as pd

from buildml import Session


def fit_log1p(frame: pd.DataFrame, params: dict) -> dict:
    return {"columns": list(frame.columns)}


def transform_log1p(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    out = frame.copy()
    for col in state["columns"]:
        out[col] = np.log1p(pd.to_numeric(out[col], errors="coerce").clip(lower=0))
    return out


Session.register_transform(
    "log1p_nonneg",
    fit=fit_log1p,
    transform=transform_log1p,
)

frame = pd.DataFrame({"income": [10, 20, 30, 45, 70, 100, 150, 200]})
session = Session.ingest(frame).set_roles({"income": "feature"}).split(test_size=0.25)
session.apply_custom_transform("log1p_nonneg", columns=["income"])
print(Session.list_transforms())
```

Custom transforms are never fold-local inside `cv_score`. For preprocessing that must fit separately in each cross-validation fold,
use a supported recipe step or implement the transform inside an
sklearn-compatible estimator pipeline.

---

## Resample strategies (train only)

```python
import pandas as pd
from sklearn.datasets import make_classification
from buildml import Session

X, y = make_classification(n_samples=160, n_features=6, n_informative=4,
                           weights=[0.8, 0.2], random_state=0)
frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(6)])
frame["target"] = y
session = (Session.ingest(frame)
    .set_roles({**{c: "feature" for c in frame if c != "target"}, "target": "target"})
    .split(test_size=0.25, stratify=True, random_state=0))

# pip install "buildml[imbalanced]"
for row in session.resample_strategies():
    print(row["name"], row.get("when") or row)

session.resample(sampler="smote", random_state=0)
# also: random_oversample, random_undersample, adasyn, borderline_smote
```

Validation and test prevalence stay as they were. Pipeline bundles record
resample as lineage. Scoring does not re-synthesize minority rows.

---

## Dry-run and plan inspection

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression

from buildml import Session

frame = pd.DataFrame(
    {
        "signup": pd.to_datetime(
            [
                "2023-01-01",
                "2023-02-15",
                "2023-03-10",
                "2023-04-01",
                "2023-05-20",
                "2023-06-11",
                "2023-07-04",
                "2023-08-19",
            ]
        ),
        "age": [21, None, 35, 40, 29, 33, 52, 47],
        "segment": ["gold", "silver", "gold", "bronze", "silver", "gold", "bronze", "silver"],
        "note": [
            "late payment risk",
            "loyal customer",
            "new account",
            "chargeback history",
            "payroll deposit",
            "travel spend",
            "student",
            "payroll deposit",
        ],
        "approved": [0, 1, 0, 1, 0, 1, 1, 0],
    }
)

session = (
    Session.ingest(frame)
    .set_roles(
        {
            "signup": "feature",
            "age": "feature",
            "segment": "feature",
            "note": "feature",
            "approved": "target",
        }
    )
    .split(test_size=0.25, stratify=True, random_state=0)
)

session.impute(strategy="median")
preview = session.dry_run(["impute", "encode", "scale"])
print(preview)
print(session.impute_plan)
print(session.last_preprocess)
```

---

## Score-time replay

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression

from buildml import Session

frame = pd.DataFrame(
    {
        "signup": pd.to_datetime(
            [
                "2023-01-01",
                "2023-02-15",
                "2023-03-10",
                "2023-04-01",
                "2023-05-20",
                "2023-06-11",
                "2023-07-04",
                "2023-08-19",
            ]
        ),
        "age": [21, None, 35, 40, 29, 33, 52, 47],
        "segment": ["gold", "silver", "gold", "bronze", "silver", "gold", "bronze", "silver"],
        "note": [
            "late payment risk",
            "loyal customer",
            "new account",
            "chargeback history",
            "payroll deposit",
            "travel spend",
            "student",
            "payroll deposit",
        ],
        "approved": [0, 1, 0, 1, 0, 1, 1, 0],
    }
)

session = (
    Session.ingest(frame)
    .set_roles(
        {
            "signup": "feature",
            "age": "feature",
            "segment": "feature",
            "note": "feature",
            "approved": "target",
        }
    )
    .split(test_size=0.25, stratify=True, random_state=0)
)

session.extract_dates(include_time=False, drop_original=True)
session.impute(strategy="median")
session.encode(method="onehot", columns=["segment"])
session.text_features(method="tfidf", columns=["note"], max_features=32, ngram_range=(1, 2))
session.scale(method="standard")
session.fit(LogisticRegression(max_iter=800), task="classification")
print(session.evaluate(partition="test").metrics)

# Apply the saved transforms to raw held-out rows, not already-transformed rows.
raw_holdout = frame.iloc[list(session.split_plan.test_indices)].copy()
applied = session.apply_preprocess_plans(raw_holdout, use_session_plans=True)
print(applied.dataset.to_pandas().head())
```

Or one-shot: `predict_from_pipeline(path, data)`
([artifacts](artifacts-checkpoints-bundles.md)).

---

## Failure modes

| Issue | Guidance |
| --- | --- |
| Prep before split | Refused: split first |
| CV after Session prep | Raises LeakageError: see the leakage guide |
| Text features create too many columns | Cap `max_features`; prefer hashing for large vocabularies |
| Target encode + small n | High variance; prefer nested CV / smoothing |
| Custom transform in CV | Not fold-local: redesign protocol |

---

## Related

- [Classical end-to-end](classical-end-to-end.md)
- [Leakage & recipes](leakage-cv-recipes.md)
- [Engines](engines-polars-duckdb.md)
