# Classical quickstart

```bash
pip install buildml
```

This is the main path: a table, a target, a holdout you can trust. The
Session holds the rows, the roles, the split, the preparation that learned
from train only, and the model. If you try to prepare or fit before a
split, it stops you.

A longer walk with dirtier data is
[classical end-to-end](classical-end-to-end.md). Paste
[`examples/classical_loan_loop.py`](../examples/classical_loan_loop.py)
(120 rows, calibration and threshold on validation). The snippet below
is a short table so you can read every cell. The proof on German Credit (OpenML `credit-g` when cached) is
[loan-approval-classical](../proofs/loan-approval-classical/). Wisconsin
breast cancer is
[breast-cancer-classical](../proofs/breast-cancer-classical/) or
[`examples/breast_cancer_classical_loop.py`](../examples/breast_cancer_classical_loop.py).

## A first loop

You have ages, incomes, and an approval label. Some ages are missing.
You want a test number that did not help fit the scaler.

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression

from buildml import Session

frame = pd.DataFrame(
    {
        "age": [21, None, 35, 40, 29, 33, 52, 47],
        "income": [40, 55, 60, 80, 50, 70, 90, 65],
        "approved": [0, 1, 0, 1, 0, 1, 1, 0],
    }
)

session = Session.ingest(frame)
session.set_roles(
    {"age": "feature", "income": "feature", "approved": "target"}
)
session.split(
    test_size=0.25,
    validation_size=0.25,
    stratify=True,
    random_state=42,
)
session.impute(strategy="median")
session.scale(method="standard")
session.fit(LogisticRegression(max_iter=500), task="classification")

validation = session.evaluate(partition="validation")
test = session.evaluate(partition="test")
print(validation.metrics)
print(test.metrics)
```

What the Session actually did:

- Columns you do not name default to `feature`. You still need exactly one
  `target` before `fit`.
- `split` stores row positions. Default `test_size` is 0.2 and
  `random_state` is 42. `stratify=True` keeps class mix in each
  partition; turn it on for classification, especially when one class is
  rare.
- `impute` and `scale` learn from training rows, then apply frozen
  numbers everywhere else. With `columns=None` they touch numeric
  `feature` columns only. `id`, `target`, `group`, `time`, `weight`, and
  `ignore` stay as they are unless you name them.
- `fit` accepts any sklearn-style estimator. `task="auto"` infers
  classification or regression from the target; say it yourself when an
  integer label would look like a quantity.
- `evaluate` defaults to `test`. Use validation while you are still
  choosing. Every extra look at test spends a little of its independence.
  `calibration` and `tune_threshold` default to validation. A split
  without a validation partition raises; pass `partition="test"` only to
  measure a frozen model.

If you need to fill a categorical gap, impute with
`strategy="most_frequent"` (or a constant) on those columns, then
`encode`. Encoding also defaults to `feature`-role categoricals and
leaves the protected roles alone.

## When the positive class is rare

Accuracy will look fine while the rare class is ignored. Read prevalence
on train. Resample **train only** after the split (`buildml[imbalanced]`).

```python
from sklearn.ensemble import RandomForestClassifier

# Requires: pip install "buildml[imbalanced]"
session.resample(sampler="smote", random_state=0)
session.fit(RandomForestClassifier(n_estimators=100, random_state=0))
```

Validation and test rows are never altered. Compare against the same
split without resample before you trust an F1 gain. Thresholds still
belong on validation.

## Regression

Same spine. Metrics come back in the target's units (MAE, RMSE) plus R².

```python
import pandas as pd
from sklearn.linear_model import Ridge

from buildml import Session

frame = pd.DataFrame(
    {
        "sqft": [850, 920, 1100, 1400, 1600, 1800, 2100, 2400],
        "beds": [2, 2, 3, 3, 4, 4, 4, 5],
        "price_k": [210, 235, 290, 360, 410, 455, 520, 610],
    }
)

session = (
    Session.ingest(frame)
    .set_roles({"sqft": "feature", "beds": "feature", "price_k": "target"})
    .split(test_size=0.25, random_state=42)
    .impute(strategy="median")
    .scale(method="standard")
    .fit(Ridge(alpha=1.0), task="regression")
)
print(session.evaluate(partition="test").metrics)
```

Trees do not need scaling. Linear and distance methods do. Scale last,
after impute, encode, and any outlier fences, so the scaler sees the
distribution the model will see.

## Groups and time

Random `split` assumes rows are interchangeable. They are not when the
same customer appears twice, or when you are predicting the future.

`group_split` moves whole entities. `test_size` counts **groups**, not
rows, so the row fraction will not match the number you typed. That is
expected: groups differ in size.

```python
import pandas as pd

from buildml import Session

visits = pd.DataFrame(
    {
        "customer_id": [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5],
        "spend": [10, 12, 15, 8, 9, 20, 22, 25, 5, 6, 30, 28],
        "churned": [0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0],
    }
)

session = (
    Session.ingest(visits)
    .set_roles(
        {
            "customer_id": "group",
            "spend": "feature",
            "churned": "target",
        }
    )
    .group_split(test_size=0.25, random_state=0)
)
```

`time_split` sorts on the `time` role and holds out the most recent
rows. Validation, if you ask for it, still sits before test in time.

If another system already decided membership, pass positional indices
(0 .. n-1, not DataFrame labels) to `inject_split`. Overlap is refused.
BuildML cannot prove your boundary matches deployment.

## Choosing a model without burning test

`compare_models` fits each candidate on train and ranks them on one
partition. The default partition is **test**. The winner becomes the
Session's fitted model. While you are still choosing, pass
`partition="validation"`.

```python
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

comparison = session.compare_models(
    {
        "prevalence": DummyClassifier(strategy="prior"),
        "logistic": LogisticRegression(max_iter=500),
        "forest": RandomForestClassifier(random_state=0),
    },
    partition="validation",
    ranking_metric="f1",
)
```

Cross-validation and search draw folds from **train only**. If you
already ran Session-global `impute` / `encode` / `scale` on the whole
training partition, `cv_score` and `grid_search` refuse. A
`PreprocessRecipe` cannot undo that: it sees the already-transformed
frame. Re-ingest (or load an unpoisoned checkpoint), then put the recipe
inside the CV call.

```python
from buildml.preprocess import PreprocessRecipe

cv = session.cv_score(
    LogisticRegression(max_iter=500),
    cv=5,
    preprocess=PreprocessRecipe(impute="median", scale="standard"),
)
```

`allow_session_global_preprocess=True` is an override for a known-biased
baseline. The score stays leakage-biased. Details and good/bad patterns:
[leakage and recipes](leakage-cv-recipes.md).

After you have a fitted model, calibration, thresholds, and permutation
importance belong on validation. Confirm a fixed threshold on test once.
[Diagnostics and search](classical-diagnostics-search.md).

## Save and reload

A checkpoint stores the table, roles, split, history, and optional
preprocess plans. It does not store the fitted estimator. A pipeline
bundle stores the plans plus the estimator. Neither embeds the other.

```python
session.checkpoint_save("artifacts/checkpoint")
restored = Session.checkpoint_load("artifacts/checkpoint")

session.save_pipeline("artifacts/pipeline", evaluate_partition="test")
```

Loaders that deserialize pickle default to `trusted=False`. Pass
`trusted=True` only for a file you made. `data_only=True` skips plans
without needing that flag. Inspect `reattach_result` after a checkpoint
load.

## When it refuses

| What you see | What happened |
| --- | --- |
| `ValidationError: No split exists` | `impute`, `scale`, `encode`, `resample`, or `fit` before a split |
| `LeakageError` on fit | Something tried to learn outside train |
| `LeakageError` on CV / search | Session-global prep already ran on the full train partition |
| `MissingExtraError` | The named extra is not installed (`imbalanced`, `optuna`, `viz`) |
| Empty partition / stratify error | Sizes left a side empty, or a class is too rare to appear in every partition |

`session.explain("impute", moment="before")` lists prerequisites before
you mutate state. The teaching studio is
[EDA / Teaching Studio](eda-teaching-studio.md), not this page.

## Next

- [Classical end-to-end](classical-end-to-end.md) for messier tables
- [Leakage and recipes](leakage-cv-recipes.md) for fold-local honesty
- [Preprocess depth](preprocess-depth.md) for encode, dates, text, custom transforms
- [Artifacts](artifacts-checkpoints-bundles.md) for checkpoint vs pipeline
- [loan-approval-classical](../proofs/loan-approval-classical/) for the same spine on German Credit (`credit-g` when cached)
