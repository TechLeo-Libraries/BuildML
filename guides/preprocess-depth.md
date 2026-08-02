# Preprocess depth

> **Install (GitHub 2.x):**
> `pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"`
> Optional: `pip install "buildml[imbalanced]"` for resample.
> See [installation](../docs/installation.rst).

Session-global preparation fits on **train** and freezes plans for other
partitions. For fold-local prep inside CV, use `PreprocessRecipe`
([leakage-cv-recipes](leakage-cv-recipes.md)) instead of calling these methods
before `cv_score`.

---

## Why plans matter

Each step stores a serializable plan (`impute_plan`, `encode_plan`, …).
`save_pipeline` ships those plans with the estimator so score-time rows see the
same frozen transforms. `apply_preprocess_plans` replays them on new frames;
resample plans are **lineage-only** at score time (they do not synthesize rows
for inference).

---

## Use case — mixed numeric + categorical + dates + text

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
session.encode(method="onehot")
session.text_features(method="tfidf", max_features=32, ngram_range=(1, 2))
session.scale(method="standard")
session.fit(LogisticRegression(max_iter=800), task="classification")
print(session.evaluate(partition="test").metrics)
```

---

## Encoding methods

| Method | Behavior | When |
| --- | --- | --- |
| `onehot` | Dense/sparse one-hot from train levels | Low-cardinality categories |
| `ordinal` | Ordered integer codes | Tree models / ordered cats |
| `infrequent` | Pool rare train levels then one-hot | Long-tail categoricals |
| `target` | Smoothed target means on train (OOF-style on train) | High-cardinality with care |

```python
session.encode(method="infrequent", min_frequency=0.2)
# or
session.encode(method="target", smoothing=10.0, n_folds=5, random_state=0)
```

Target encoding on Session train is fine for a **final** model after split. For
CV, put `encode="target"` inside `PreprocessRecipe` so means refit per fold.

---

## Outliers, binning, selection, PCA

```python
session.handle_outliers(method="iqr", action="cap")
session.bin(strategy="quantile", n_bins=4, encode_as="ordinal")
session.select_features(strategy="univariate", k=10)
session.reduce_dimensions(method="pca", n_components=5, prefix="pc")
```

- `action="drop"` on outliers rebuilds splits after removing train rows.
- Feature selection and PCA fit on train only; dropping input columns is
  configurable via method kwargs where exposed.
- Prefer expressing bin/select/reduce inside `PreprocessRecipe` when those
  steps participate in CV/search knobs (`SAFE_RECIPE_KNOBS`).

---

## Custom transforms (Session-global only)

```python
import numpy as np
import pandas as pd

from buildml import Session


def fit_log1p(frame: pd.DataFrame, columns: list[str], params: dict) -> dict:
    return {"columns": list(columns)}


def transform_log1p(frame: pd.DataFrame, columns: list[str], state: dict) -> pd.DataFrame:
    out = frame.copy()
    for col in state["columns"]:
        out[col] = np.log1p(pd.to_numeric(out[col], errors="coerce").clip(lower=0))
    return out


Session.register_transform(
    "log1p_nonneg",
    fit=fit_log1p,
    transform=transform_log1p,
)

session.apply_custom_transform("log1p_nonneg", columns=["income"])
print(Session.list_transforms())
```

**Limit:** custom transforms are never fold-local inside `cv_score`. If you need
fold-local honesty, keep the logic out of CV or accept Session-global bias with
eyes open.

---

## Resample strategies (train only)

```python
# After GitHub 2.x install:
# pip install "buildml[imbalanced]"
for row in session.resample_strategies():
    print(row["sampler"], row.get("when") or row)

session.resample(sampler="smote", random_state=0)
# also: random_oversample, random_undersample, adasyn, borderline_smote
```

Validation/test prevalence is unchanged. Pipeline bundles record resample as
lineage; scoring does not re-synthesize minority rows.

---

## Dry-run and plan inspection

```python
preview = session.dry_run(["impute", "encode", "scale"])
print(preview)
print(session.impute_plan)
print(session.last_preprocess)
```

---

## Score-time replay

```python
from buildml import Session

# After load_pipeline or with plans present:
applied = session.apply_preprocess_plans(
    session.partition("test"),
    use_session_plans=True,
)
```

Or one-shot: `predict_from_pipeline(path, data)`
([artifacts](artifacts-checkpoints-bundles.md)).

---

## Failure modes

| Issue | Guidance |
| --- | --- |
| Prep before split | Refused — split first |
| CV after Session prep | Hard-refuse — see leakage guide |
| Text columns explode width | Cap `max_features`; prefer hashing for huge vocab |
| Target encode + small n | High variance; prefer nested CV / smoothing |
| Custom transform in CV | Not fold-local — redesign protocol |

---

## Related

- [Classical end-to-end](classical-end-to-end.md)
- [Leakage & recipes](leakage-cv-recipes.md)
- [Engines](engines-polars-duckdb.md)
