# Classical end-to-end

> **Install:** PyPI `buildml` is legacy 1.x. Use
> `pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"`
> (or editable checkout). See [installation](../docs/installation.rst).

This guide is the deep classical path: **dirty tabular data → roles → split →
train-fitted preparation → fit → validation diagnostics → holdout evaluate →
pipeline bundle**. For the short on-ramp, see
[quickstart-classical](quickstart-classical.md). For leakage and fold-local CV,
see [leakage-cv-recipes](leakage-cv-recipes.md).

Classical flat calls (`session.fit`, `session.evaluate`, …) remain the preferred
DX; `session.classical.fit` / `session.classical.evaluate` exist as a dual
namespaced path and are not required here.

## Why Session order exists

Supervised ML fails quietly when preparation statistics (medians, category
levels, scale parameters) are computed on rows that later pretend to be
“unseen.” BuildML makes that failure loud:

1. **Roles** declare what each column *is* (feature, target, group, time,
   weight, id, ignore): the library will not infer deployment semantics.
2. **Split** creates partition membership before any fit-capable step.
3. **Preparation** learns on train only and freezes plans for other partitions.
4. **Fit / evaluate** respect partition purpose: validation for choices, test
   for a fixed policy.

`assert_can_fit("train")` backs impute, encode, scale, resample, and `fit`.
Skipping split raises rather than leaking.

Cross-links: [concepts](../docs/concepts.rst),
[workflow-guide](../docs/workflow-guide.rst), [glossary](glossary.md).

---

## Use case A: Loan approval with missing ages

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression

from buildml import Session

frame = pd.DataFrame(
    {
        "age": [21, None, 35, 40, 29, 33, 52, 47, 31, None, 44, 38],
        "income": [40, 55, 60, 80, 50, 70, 90, 65, 48, 72, 88, 61],
        "region": ["N", "S", "N", "W", "S", "N", "W", "S", "N", "S", "W", "N"],
        "approved": [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0],
    }
)

session = Session.ingest(frame)
session.set_roles(
    {
        "age": "feature",
        "income": "feature",
        "region": "feature",
        "approved": "target",
    }
)
session.split(
    test_size=0.25,
    validation_size=0.25,
    stratify=True,
    random_state=42,
)

# Read-only first: findings do not mutate the frame.
eda = session.eda(include_plots=False)
print(eda.findings[0].title if eda.findings else "no findings")

session.impute(strategy="median")
session.encode(method="onehot")
session.scale(method="standard")
session.fit(LogisticRegression(max_iter=500), task="classification")

val = session.evaluate(partition="validation")
test = session.evaluate(
    partition="test",
    include_plots=False,
    export_html="artifacts/loan_eval.html",
)
print(val.metrics, test.metrics)

session.save_pipeline("artifacts/loan_pipeline", evaluate_partition="test")
```

**Why this order:** median age and one-hot levels come from train. Validation
informs threshold / feature choices; test scores the frozen policy once.

---

## Use case B: Fraud-like imbalance

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from buildml import Session

rng = pd.Series(range(500))
frame = pd.DataFrame(
    {
        "amount": rng * 1.2 + 5,
        "velocity": (rng % 11).astype(float),
        "channel": (rng % 3).map({0: "web", 1: "app", 2: "pos"}),
        "is_fraud": (rng % 25 == 0).astype(int),
    }
)

session = (
    Session.ingest(frame)
    .set_roles(
        {
            "amount": "feature",
            "velocity": "feature",
            "channel": "feature",
            "is_fraud": "target",
        }
    )
    .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0)
)

print(session.partition("train")["is_fraud"].mean())  # prevalence

# Requires: pip install "buildml[imbalanced]"  (after GitHub 2.x install)
session.encode(method="onehot")
session.resample(sampler="smote", random_state=0)
session.fit(RandomForestClassifier(n_estimators=100, random_state=0))

print(session.evaluate(partition="validation").metrics.get("f1"))
print(session.evaluate(partition="test").metrics.get("f1"))
```

Resample **never** touches validation/test. Compare against a non-resampled
baseline before trusting F1. Threshold tuning still belongs on validation
([diagnostics](classical-diagnostics-search.md)).

---

## Use case C: House price regression

```python
import pandas as pd
from sklearn.linear_model import Ridge

from buildml import Session

frame = pd.DataFrame(
    {
        "sqft": [850, 920, 1100, 1400, 1600, 1800, 2100, 2400, 1000, 1300],
        "beds": [2, 2, 3, 3, 4, 4, 4, 5, 2, 3],
        "year_built": [1990, 1985, 2001, 2010, 1975, 2015, 2008, 2020, 1998, 2005],
        "price_k": [210, 235, 290, 360, 410, 455, 520, 610, 250, 330],
    }
)

session = (
    Session.ingest(frame)
    .set_roles(
        {
            "sqft": "feature",
            "beds": "feature",
            "year_built": "feature",
            "price_k": "target",
        }
    )
    .split(test_size=0.25, validation_size=0.25, random_state=42)
    .impute(strategy="median")
    .scale(method="standard")
    .fit(Ridge(alpha=1.0), task="regression")
)

print(session.evaluate(partition="test").metrics)
```

Report MAE/RMSE in the target unit. If you log-transform the target yourself,
interpret metrics in that space or back-transform explicitly.

---

## Use case D: Grouped customers (no row-ID leakage)

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression

from buildml import Session

visits = pd.DataFrame(
    {
        "customer_id": [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6],
        "spend": [10, 12, 15, 8, 9, 20, 22, 25, 5, 6, 30, 28, 11, 14],
        "tenure": [1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 1, 2, 1, 2],
        "churned": [0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0],
    }
)

session = (
    Session.ingest(visits)
    .set_roles(
        {
            "customer_id": "group",
            "spend": "feature",
            "tenure": "feature",
            "churned": "target",
        }
    )
    .group_split(test_size=0.25, validation_size=0.25, random_state=0)
    .scale(method="standard")
    .fit(LogisticRegression(max_iter=500), task="classification")
)

print(session.evaluate(partition="test").metrics)
```

Random `split` would put the same customer in train and test. `group` role +
`group_split` keeps entities whole. BuildML does **not** invent your entity key.

---

## Use case E: Chronological holdout

```python
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from buildml import Session

n = 48
frame = pd.DataFrame(
    {
        "ts": pd.date_range("2022-01-01", periods=n, freq="W"),
        "demand": [50 + i * 0.4 + (i % 5) for i in range(n)],
        "promo": [(i % 7 == 0) * 1 for i in range(n)],
    }
)

session = (
    Session.ingest(frame)
    .set_roles({"ts": "time", "demand": "target", "promo": "feature"})
    .time_split(test_size=0.2, validation_size=0.2)
    .fit(GradientBoostingRegressor(random_state=0), task="regression")
)

print(session.evaluate(partition="test").metrics)
```

---

## Use case F: External indices via `inject_split`

```python
import numpy as np
import pandas as pd

from buildml import Session

frame = pd.DataFrame({"x": range(20), "y": [i % 2 for i in range(20)]})
session = Session.ingest(frame).set_roles({"x": "feature", "y": "target"})
session.inject_split(
    train_indices=list(range(0, 12)),
    validation_indices=list(range(12, 16)),
    test_indices=list(range(16, 20)),
)
# BuildML checks overlap/range; it cannot prove your design matches production.
```

---

## Teaching before mutating

```python
before = session.explain("impute", moment="before")
print(before.risks)
preview = session.dry_run(["impute", "encode", "scale", "fit"])
for step in session.workflow():
    if step.status == "blocked":
        print(step.operation, step.reasons)
```

See [EDA / Teaching Studio](eda-teaching-studio.md).

---

## Persistence: checkpoint vs pipeline

```python
# Mid-workflow resume (data + roles + splits + history + optional plans)
session.checkpoint_save("artifacts/ckpt")
restored = Session.checkpoint_load("artifacts/ckpt")
print(restored.reattach_result.status)

# Deployable scoring (plans + estimator + model card): not a checkpoint
session.save_pipeline("artifacts/pipe", evaluate_partition="test")

from buildml.pipeline import predict_from_pipeline

scored = predict_from_pipeline(
    "artifacts/pipe",
    session.partition("test"),
    return_proba=True,
)
```

Full matrix: [artifacts-checkpoints-bundles](artifacts-checkpoints-bundles.md).

---

## Common failure modes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `ValidationError: No split exists` | Prep/fit before split | Call `split` / `group_split` / `time_split` / `inject_split` |
| `LeakageError` | Fit-capable work outside train | Keep prep + `fit` on train scope |
| Weak test after many validation tweaks | Test used for selection | Freeze policy on validation; score test once |
| `MissingExtraError: imbalanced` | Extra not installed | `pip install "buildml[imbalanced]"` on 2.x |
| Pipeline scores differ from Session | Plans missing / resample lineage | Prefer `save_pipeline`; resample is lineage-only at score time |

---

## Related guides

- [Leakage & CV recipes](leakage-cv-recipes.md)
- [Preprocess depth](preprocess-depth.md)
- [Diagnostics & search](classical-diagnostics-search.md)
- [Engines](engines-polars-duckdb.md)
- [Artifacts](artifacts-checkpoints-bundles.md)
- Sphinx: [usage](../docs/usage.rst), [features](../docs/features.rst)
