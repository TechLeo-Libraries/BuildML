# Multi-task / multi-output

```bash
pip install buildml
# XGBoost / LightGBM / CatBoost multi-target: pip install "buildml[multitask-industry]"
# shared-trunk multi-head: pip install "buildml[torch]"
```

Two or more targets share one feature matrix. Classical `session.fit`
still expects exactly one target. This path is separate.

`session.multitask.fit` defaults to method `multi_output` and base
estimator `logistic_regression`. That pairing stays sklearn even if
XGBoost is installed. Pass `method="multi_output_xgb"` with
`backend=None` to take industry when a GBDT extra imported cleanly.
Pass `method="shared_trunk_multihead"` with `backend=None` to take
torch. `backend="industry"` with the default `multi_output` method is
refused: name an industry method (`multi_output_xgb`,
`multi_output_lgbm`, or `multi_output_catboost`).

The API refuses mixed classification plus regression on sklearn and
industry, fewer than two targets, and a fit without a split. You decide
which columns are targets (`role="target"` or `targets=`), whether to
chain them, and whether mixed heads on torch are what you meant.

Short on-ramp: [multi-task quickstart](quickstart-multi-task.md).
Proof: [multi-target-underwriting](../proofs/multi-target-underwriting/).

## A first joint fit

Prefer two or more `role="target"` columns. `task="auto"` infers
classification vs regression from dtypes and cardinality. Say the task
yourself when an integer label would look like a quantity.

`split(stratify=True)` still goes through the single-target gate. With
several target roles, split without stratification (or stratify on a
temporary single-target setup, then restore roles).

```python
import numpy as np
import pandas as pd

from buildml import Session

rng = np.random.default_rng(0)
n = 240
x0 = rng.normal([-1.0, -1.0], 0.55, size=(n // 2, 2))
x1 = rng.normal([1.2, 1.0], 0.55, size=(n - n // 2, 2))
frame = pd.DataFrame(np.vstack([x0, x1]), columns=["x", "y"])
frame["t1"] = [0] * (n // 2) + [1] * (n - n // 2)
frame["t2"] = ([0, 1] * (n // 2))[:n]

session = (
    Session.ingest(frame)
    .set_roles({"x": "feature", "y": "feature", "t1": "target", "t2": "target"})
    .split(test_size=0.2, validation_size=0.2, random_state=0)
    .scale(method="standard")
)

fit = session.multitask.fit(
    backend="sklearn",
    method="multi_output",
    task="classification",
    base_estimator="logistic_regression",
)
print(fit.backend, fit.n_tasks, fit.target_columns)

ev = session.multitask.evaluate(partition="validation")
print(ev.metrics)
print(ev.per_task_metrics)

session.multitask.save_bundle("artifacts/multitask_bundle")
```

`evaluate` defaults to validation. `predict` defaults to test. Neither
refits. `attach=True` on predict can write columns with prefix
`multitask_pred` (override with `prediction_prefix=`).

## Backends and methods

| Backend | Extra | Methods | Target mix |
| --- | --- | --- | --- |
| `sklearn` | none (`multi_output` default) | `multi_output`, `classifier_chain`, `regressor_chain` | same-type only |
| `industry` | `buildml[multitask-industry]` | `multi_output_xgb`, `multi_output_lgbm`, `multi_output_catboost` | same-type only |
| `torch` | `buildml[torch]` | `shared_trunk_multihead` | mixed cls+reg via separate heads |

Industry is available when at least one of XGBoost, LightGBM, or
CatBoost imports in a subprocess. Chains stay on sklearn. There is no
ClassifierChain on a GBDT backend.

Torch is a shared MLP trunk with per-task heads and joint training.
`epochs` defaults to 60, `batch_size` to 64, `device` to `"cpu"`. It is
not a task-affinity search product.

```python
session.multitask.capability_matrix()
```

## Metrics

`evaluate` returns:

- `per_task_metrics[task]`: accuracy / F1 (classification) or MAE / RMSE / R² (regression)
- `metrics`: unweighted means across tasks of each kind (`mean_accuracy`, `mean_mae`, ...)

A mixed torch plan reports classification and regression aggregates
separately. Holdout is never used for fitting.

## Bundles

`buildml.multitask_bundle.v1` stores the `MultiTaskPlan`: estimator,
target contract, per-task label encoders, backend metadata. A Session
checkpoint does not embed it. Reload the table with `checkpoint_load`.
Reload the learner with `session.multitask.load_bundle`. `trusted=True`
only for a file you made.

[Artifacts](artifacts-checkpoints-bundles.md)

## When it refuses

| What you see | What happened |
| --- | --- |
| Fit before a split | Train-only contract |
| Fewer than two targets | Pass roles or `targets=` |
| Mixed cls+reg on sklearn/industry | Use torch `shared_trunk_multihead`, or split the problem |
| `session.fit` with several target roles | Classical fit still calls `require_target()` |
| Missing extra | Named GBDT or torch method without the extra |
| Null features | Impute (and usually scale) first |

This is not causal multi-task, federated MTL, or a multi-label
binary-relevance zoo.

[Multi-task quickstart](quickstart-multi-task.md)
