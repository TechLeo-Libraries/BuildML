# Classical quickstart

BuildML centers on a `Session` that keeps data, column roles, partition
membership, fitted preparation, model state, and operation history together.
This guide walks from a first classification example through regression,
imbalance, group and time splits, teaching APIs, failure modes, engines, and
persistence.

For vocabulary and judgment calls (leakage, partitions, metrics), read
[concepts](../docs/concepts.rst) and the [glossary](glossary.md). For the
decision framework at each stage, see [workflow-guide](../docs/workflow-guide.rst).

---

## 1. First loop: loan approval classification

Missing ages and differently scaled numeric columns are common in tabular
classification. BuildML requires a split before fit-capable preprocessing so
medians and scale parameters come from training rows only.

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

# Inspect before changing data. Findings and recommendations are read-only.
eda = session.eda(include_plots=False)

# Learned plans fit on train and apply frozen values to every partition.
session.impute(strategy="median")
session.handle_outliers(method="iqr", action="cap")
session.scale(method="standard")
session.fit(LogisticRegression(max_iter=500), task="classification")

# Use validation while making choices.
validation = session.evaluate(partition="validation")

# Reserve test for the fixed model and decision policy.
test = session.evaluate(
    partition="test",
    include_plots=True,
    export_html="artifacts/evaluation.html",
)
```

**Why order matters:** `impute`, `encode`, `scale`, `resample`, and `fit` call
`assert_can_fit("train")`. Without a split, BuildML raises rather than leak
holdout statistics into preparation.

---

## 2. Imbalanced classification

When the positive class is rare, accuracy can look fine while recall fails.
Read prevalence on each partition. Resample train only after splitting:

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from buildml import Session

rng = pd.Series(range(400))
frame = pd.DataFrame(
    {
        "amount": rng * 1.2 + 5,
        "velocity": (rng % 11).astype(float),
        "is_fraud": (rng % 25 == 0).astype(int),
    }
)

session = (
    Session.ingest(frame)
    .set_roles({"amount": "feature", "velocity": "feature", "is_fraud": "target"})
    .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0)
)

# Requires: pip install "buildml[imbalanced]"
session.resample(sampler="smote", random_state=0)
session.fit(RandomForestClassifier(n_estimators=100, random_state=0))

print(session.evaluate(partition="validation").metrics)
print(session.evaluate(partition="test").metrics)
```

Compare against a run without `resample` on the same split before trusting F1
gains. Threshold and calibration choices still belong on validation, not test.

---

## 3. Regression: house prices

Regression uses the same Session spine with `task="regression"` and
regression metrics (MAE, RMSE, R²):

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

Report MAE/RMSE with the target unit. If you transform the target, interpret
metrics in transformed space or back-transform explicitly in your analysis.

---

## 4. Group and time splits

Random `split` assumes exchangeable rows. When rows share an entity or follow
time, use `group_split` or `time_split`:

```python
import pandas as pd

from buildml import Session

# Multiple rows per customer — random split leaks the same customer across partitions
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

For temporal data, assign a `time` role and call `time_split`. When an external
system already defined memberships, pass positional indices to
`inject_split`. BuildML checks overlap and range; it cannot prove your boundary
matches deployment.

---

## 5. Teaching surfaces: explain, workflow, walkthrough, dry_run

BuildML ships a versioned operation catalog linked to every public Session
method. These APIs expose assumptions and risks; they do not certify domain
correctness.

```python
# Before mutating state: prerequisites, leakage risks, alternatives
before = session.explain("impute", moment="before")
print(before.prerequisite_status)
print(before.risks)

# Full prerequisite graph
workflow = session.workflow()
for step in workflow:
    if step.status == "blocked":
        print(step.operation, step.reasons or step.blockers)

# Preview without history mutation
preview = session.dry_run(["impute", "scale", "fit"])
summary = session.summarize_history()
print(summary.operation_counts)

# After a call: join catalog to latest state transition
session.impute(strategy="median")
after = session.explain("impute", moment="after")

# Offline audit HTML
walkthrough = session.walkthrough(export_html="artifacts/workflow.html")
```

`workflow()` marks steps `available` when prerequisites pass — that is not a
recommendation to run them. `eda()` findings and recommendations are also
read-only; acting on a recommendation still requires an explicit Session call.

---

## 6. Failure modes BuildML surfaces

Common errors and what they mean:

| Symptom | Likely cause | What to do |
| --- | --- | --- |
| `ValidationError: No split exists` | `impute` / `fit` before split | Call `split`, `group_split`, `time_split`, or `inject_split` |
| `LeakageError` | Fit-capable work outside train | Ensure preparation and `fit` use training scope only |
| `MissingExtraError` | Optional dependency not installed | Install the named extra (`imbalanced`, `optuna`, `dashboard`, …) |
| Blocked step in `workflow()` | Prerequisites missing | Read `step.reasons`; call `explain(operation, moment="before")` |
| Weak test metric after many validation tweaks | Test used for selection | Fix choices on validation; run test once |

RAG eval hygiene example: documents marked `eval_only` refuse indexing
(`LeakageError`) so labeled answers do not enter the retrieval corpus.

---

## 7. Engines: pandas, Polars, DuckDB

Pandas is the canonical sklearn path. Polars and DuckDB help ingest, filter,
and aggregate before materialization:

```python
from buildml import Session
from buildml.data import portable_filter_expr

with Session.ingest("data.csv", engine="duckdb") as session:
    pred = portable_filter_expr("amount", ">", 100)
    narrowed = session.dataset.filter_expr(pred)
```

`with session:` closes owned DuckDB connections. Lazy Polars frames collect at
sklearn boundaries — that is not out-of-core training. After Session preprocess
steps, native handles rebuild so `Dataset.project` / `prepare_design_matrix`
can prefer engine ops where implemented.

---

## 8. Cross-validation and hyperparameter search

Folds stay inside train; Session test is not used for ranking:

```python
from sklearn.tree import DecisionTreeClassifier

from buildml.preprocess import PreprocessRecipe

cv = session.cv_score(
    LogisticRegression(max_iter=500),
    cv=5,
    preprocess=PreprocessRecipe(impute="median", scale="standard"),
)
print(cv.mean_metrics[cv.scoring_metric], "±", cv.std_metrics[cv.scoring_metric])

search = session.grid_search(
    DecisionTreeClassifier(random_state=0),
    param_grid={"max_depth": [2, 4], "min_samples_leaf": [1, 5]},
    cv=5,
)

# Optional adaptive search (requires: pip install 'buildml[optuna]')
# optuna_result = session.optuna_search(
#     DecisionTreeClassifier(random_state=0),
#     param_space={"max_depth": {"type": "int", "low": 2, "high": 6}},
#     n_trials=8,
#     cv=3,
# )

# Post-selection estimate: outer folds score winners chosen by inner CV only.
nested = session.nested_cv_score(
    DecisionTreeClassifier(random_state=0),
    param_grid={"max_depth": [2, 4], "min_samples_leaf": [1, 5]},
    outer_cv=3,
    inner_cv=3,
)
print(
    nested.mean_metrics[nested.scoring_metric],
    "±",
    nested.std_metrics[nested.scoring_metric],
)
```

Prefer `PreprocessRecipe` inside CV for fold-local honesty — on data that has
**not** already been Session-globally imputed/encoded/scaled. Resample and
`apply_custom_transform` remain Session-global. If Session-global preprocess
already ran, CV/search refuse **even with** a fold-local recipe (recipes do
not undo poisoned frames). Re-ingest unpoisoned data, or set
`allow_session_global_preprocess=True` only as an explicit override.
`compare_models` ranks on one partition — override the default to
`validation` during iterative selection.

---

## 9. Diagnostics beyond accuracy

After `fit`, inspect calibration, thresholds, and reliance:

```python
session.calibration(partition="validation")
session.tune_threshold(partition="validation", fp_cost=1.0, fn_cost=5.0)
session.feature_importance(partition="validation", n_repeats=10)
session.error_slices(partition="test", by="customer_id")
```

Select thresholds on validation; confirm the fixed policy on test.
Permutation importance measures model reliance, not causal effect.

---

## 10. Checkpoint and pipeline round-trip

```python
session.checkpoint_save("artifacts/checkpoint")
restored = Session.checkpoint_load("artifacts/checkpoint")
print(restored.reattach_result.status)

session.save_pipeline("artifacts/pipeline", evaluate_partition="test")
print(session.model_card.lineage.get("plans_present"))

from buildml.pipeline import predict_from_pipeline

holdout = session.partition("test")
scored = predict_from_pipeline(
    "artifacts/pipeline",
    holdout,
    return_proba=True,
)
```

Checkpoints restore dataset, roles, partitions, history, and optional preprocess
plan objects. They do not embed fitted estimators. Pipeline bundles store plans
plus estimator and model card; neither artifact contains the other. Resample
plans are lineage-only at score time.

Inspect `reattach_result` after load. A `data_only=True` load deliberately
discards prior workflow semantics.

---

## 11. Optional paths on the same Session

Classical APIs stay authoritative. Extras attach without replacing them:

| Path | Extra | Guide |
| --- | --- | --- |
| Tabular Torch | `buildml[torch]` | [quickstart-torch.md](quickstart-torch.md) |
| Retrieval | `buildml[rag]` | [quickstart-rag.md](quickstart-rag.md) |
| LLM operator | `buildml[ai]` | [quickstart-ai.md](quickstart-ai.md) |

---

## Install for development

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,all-classical]"
```

See [workflow-guide](../docs/workflow-guide.rst) and [concepts](../docs/concepts.rst).
