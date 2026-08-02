# Multi-task / multi-output deep guide

## What this is

BuildML multi-task learning fits **multiple targets that share one feature
matrix** using sklearn façades:

| Method | Task | Wrapper |
| --- | --- | --- |
| `multi_output` | classification | `MultiOutputClassifier` |
| `multi_output` | regression | `MultiOutputRegressor` |
| `classifier_chain` | classification | `ClassifierChain` |
| `regressor_chain` | regression | `RegressorChain` |

Base estimators (core sklearn):

- Classification: `logistic_regression`, `hist_gradient_boosting`
- Regression: `ridge`, `hist_gradient_boosting_regressor`

This is **not** a universal MTL research platform (no multi-head Torch rewrite,
no task-affinity search, no multi-label binary-relevance zoo, no causal
multi-task).

## Leakage discipline

1. `fit_multitask` requires a `SplitPlan` and fits **train only**.
2. Validation / test are evaluation-only (`evaluate_multitask` /
   `predict_multitask` never refit).
3. Classical `Session.fit` still calls `require_target()` and expects **exactly
   one** target — multi-task is a distinct Session path.
4. `split(stratify=True)` also uses `require_target()` (single target). With
   multiple target roles, split without stratification (or stratify on a
   temporary single-target setup).

## Targets

- Prefer multiple `role="target"` columns, or pass `targets=[...]`.
- Need **≥ 2** targets.
- `task="auto"` infers classification vs regression from column dtypes /
  cardinality; **mixed kinds are refused**.
- Force `task="classification"` or `task="regression"` only when every target is
  compatible.

## Metrics

`evaluate_multitask` returns:

- `per_task_metrics[task]`: accuracy / F1 (cls) or MAE / RMSE / R² (reg)
- `metrics`: unweighted means across tasks (`mean_accuracy`, `mean_mae`, …)

## Bundle boundary

`buildml.multitask_bundle.v1` stores `MultiTaskPlan` (estimator + target
contract + per-task label encoders). Session checkpoints do **not** embed it.
Reload tabular workflow via `checkpoint_load`; reload the learner via
`load_multitask_bundle`.

See [Artifacts](artifacts-checkpoints-bundles.md).

## Teaching surfaces

- Concepts: `multitask-multi-output`, `multitask-chain`,
  `multitask-target-roles`, `multitask-bundle-boundary`
- Session ops: `fit_multitask`, `predict_multitask`, `evaluate_multitask`,
  `save_multitask_bundle`, `load_multitask_bundle`
- Walkthrough: `multitask_status`
- AI allowlist: fit / evaluate / save / load

## Phase tracker

Phase 2 items 1–5 done (semi → self-supervised → active → online → **multi-task**).
Next depth-first item: **meta-learning**.
