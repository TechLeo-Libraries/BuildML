# BuildML 2.0 alpha quickstart

BuildML 2.x uses a `Session` to keep data, semantic roles, partition
membership, fitted preparation, model state, and operation history together.

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
# Optional: session.bin(...), session.encode(method="infrequent"|"target"),
# session.select_features(...), session.text_features(...),
# session.reduce_dimensions(method="pca"), session.apply_custom_transform(...)
# Preview without mutation: session.dry_run(["impute", "scale", "fit"])
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

Random splitting assumes independent, exchangeable rows. Prefer
`session.group_split(...)` or `session.time_split(...)` when a group or time
role defines the boundary. Use `session.inject_split(...)` for externally
governed memberships.

## Cross-validation and hyperparameter search

```python
from sklearn.tree import DecisionTreeClassifier

from buildml.preprocess import PreprocessRecipe

# Folds stay inside train; test is never used for ranking.
# Fold-local recipes may include dates, text, binning, encode, scale,
# reduce (PCA), select, and outliers. Custom transforms stay Session-global.
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
# Optional Optuna inner search (requires: pip install 'buildml[optuna]')
# nested_optuna = session.nested_cv_score(
#     DecisionTreeClassifier(random_state=0),
#     inner_search="optuna",
#     param_space={"max_depth": {"type": "int", "low": 2, "high": 6}},
#     n_trials=6,
#     outer_cv=3,
#     inner_cv=3,
#     # Opt-in: share Optuna study priors across outer folds (still no test peeking)
#     # warm_start_studies=True,
# )
print(
    nested.mean_metrics[nested.scoring_metric],
    "±",
    nested.std_metrics[nested.scoring_metric],
)

validation = session.evaluate(partition="validation")
test = session.evaluate(partition="test")
```

## Explain and inspect the workflow

```python
before = session.explain("feature_importance", moment="before")
workflow = session.workflow()

importance = session.feature_importance(partition="validation")
after = session.explain("feature_importance", moment="after")

walkthrough = session.walkthrough(export_html="artifacts/workflow.html")
```

`workflow()` reports operations as done, available, blocked, or skipped.
`explain()` joins catalog guidance to current state or the latest recorded
operation. `walkthrough()` can export the full state-and-history view as a
self-contained local HTML file. These APIs expose known assumptions and risks;
they do not prove that the split or model is suitable for the domain.

## Checkpoint and resume

```python
session.checkpoint_save("artifacts/checkpoint")
# Optional native-sidecar knobs (defaults: layout=auto, zstd, 25k rows/part):
# session.checkpoint_save(
#     "artifacts/checkpoint",
#     sidecar_layout="partitioned",
#     sidecar_partition_rows=10_000,
#     sidecar_compression="zstd",
# )
restored = Session.checkpoint_load("artifacts/checkpoint")
print(restored.reattach_result.status)
```

The checkpoint restores dataset, roles, partitions, history, and optional
preprocess plan objects (`plans.joblib`) for mid-loop resume. When a
Polars/DuckDB native handle was attached, an optional native sidecar is written
(`data/native_sidecar.parquet`, or partitioned `data/native_sidecar/` for large
frames; zstd by default) so restore can reattach without always rebuilding
eagerly from the Pandas-exported frame. LazyFrame *plans* are not persisted —
`lazy_intent` restore uses `scan_parquet` / `read_parquet` over the sidecar
bytes. Save the estimator alone with `save_model` / `load_model`, or
save preprocess plans plus a model card with `save_pipeline` / `load_pipeline`.
Pipeline bundles do not embed checkpoint data; checkpoints do not embed fitted
estimators. Model cards list which preprocess plans are present.

DuckDB sessions own a connection on the root dataset. Prefer a context manager
so it closes when you leave the block:

```python
from buildml import Session
from buildml.data import portable_filter_expr

with Session.ingest("data.csv", engine="duckdb") as session:
    pred = portable_filter_expr("amount", ">", 100)
    narrowed = session.dataset.filter_expr(pred)
```

`portable_filter_expr` builds simple quoted comparisons that work for both
Polars and DuckDB `filter_expr` paths. Complex SQL stays engine-specific.

```python
session.save_pipeline("artifacts/pipeline", evaluate_partition="test")
print(session.model_card.lineage.get("plans_present"))

# One-shot score path: load bundle → apply plans → predict (+ optional proba).
from buildml.pipeline import predict_from_pipeline

holdout = session.partition("test")
scored = predict_from_pipeline(
    "artifacts/pipeline",
    holdout,
    return_proba=True,
)
print(scored.n_rows, scored.apply_result.applied if scored.apply_result else ())

# Manual plan replay remains available when you need the transformed Dataset.
from buildml.preprocess import apply_preprocess_plans

applied = apply_preprocess_plans(
    holdout,
    {
        "impute_plan": session.impute_plan,
        "scale_plan": session.scale_plan,
    },
)
print(applied.applied, applied.skipped)
```

Pipeline bundles write `buildml.pipeline_bundle.v2` / `buildml.plans.v2`. Older
unversioned flat `plans.joblib` payloads still load. Soft materialization gates
warn near ~250 MiB; configure `BUILDML_MATERIALIZATION_HARD_LIMIT_BYTES` for a
hard refuse at design-matrix boundaries. With Polars/DuckDB configured,
`session.prepare_design_matrix(...)` and `Dataset.project` /
`Dataset.aggregate` (including `median` / `q25`…`q75`) prefer native
Polars/DuckDB ops before the Pandas design matrix used by sklearn — they do
not enable out-of-core training.

## Install for development

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,all-classical]"
```

See [workflow-guide.rst](./workflow-guide.rst) and [concepts.rst](./concepts.rst).
