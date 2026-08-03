# Multi-task / multi-output deep guide

## What this is

BuildML multi-task learning fits **multiple targets that share one feature
matrix** with honest backend routing:

| Backend | Extra | Methods | Targets |
| --- | --- | --- | --- |
| `sklearn` | core | `multi_output`, `classifier_chain`, `regressor_chain` | same-type only |
| `industry` | `multitask-industry` | `multi_output_xgb`, `multi_output_lgbm`, `multi_output_catboost` | same-type only |
| `torch` | `torch` | `shared_trunk_multihead` | mixed cls+reg supported |

Inspect defaults and availability:

```python
from buildml.multitask import multitask_capability_matrix

multitask_capability_matrix()
```

When extras are installed, industry defaults to XGBoost multi-target; core
sklearn remains the fallback when extras are missing.

This is **not** a universal MTL research platform (no task-affinity search,
no multi-label binary-relevance zoo, no causal multi-task).

## Leakage discipline

1. `fit_multitask` requires a `SplitPlan` and fits **train only**.
2. Validation / test are evaluation-only (`evaluate_multitask` /
   `predict_multitask` never refit).
3. Classical `Session.fit` still calls `require_target()` and expects **exactly
   one** target: multi-task is a distinct Session path.
4. `split(stratify=True)` also uses `require_target()` (single target). With
   multiple target roles, split without stratification (or stratify on a
   temporary single-target setup).

## Targets

- Prefer multiple `role="target"` columns, or pass `targets=[...]`.
- Need **≥ 2** targets.
- `task="auto"` infers classification vs regression from column dtypes /
  cardinality.
- **Sklearn/industry:** mixed classification+regression is refused.
- **Torch `shared_trunk_multihead`:** mixed targets get separate heads and
  joint training.

## Metrics

`evaluate_multitask` returns:

- `per_task_metrics[task]`: accuracy / F1 (cls) or MAE / RMSE / R² (reg)
- `metrics`: unweighted means across tasks of each kind (`mean_accuracy`,
  `mean_mae`, …). Mixed torch plans report cls and reg aggregates separately.

## Bundle boundary

`buildml.multitask_bundle.v1` stores `MultiTaskPlan` (estimator + target
contract + per-task label encoders + backend metadata). Session checkpoints do
**not** embed it. Reload tabular workflow via `checkpoint_load`; reload the
learner via `load_multitask_bundle`.

See [Artifacts](artifacts-checkpoints-bundles.md).

## Teaching surfaces

- Concepts: `multitask-multi-output`, `multitask-chain`,
  `multitask-target-roles`, `multitask-bundle-boundary`
- Session ops: `fit_multitask`, `predict_multitask`, `evaluate_multitask`,
  `save_multitask_bundle`, `load_multitask_bundle`
- Walkthrough: `multitask_status` (includes capability matrix)
- AI allowlist: fit / evaluate / save / load

## Benchmark

```bash
python benchmarks/multitask/multi_target_quality.py
```

Writes `benchmarks/multitask/results/multi_target_quality.json` comparing
sklearn vs industry/torch when extras are installed.

## Phase tracker

Phase 2 item 5 (**multi-task industry depth**, R6.4) is done.
Next depth-first item: **meta-learning** (R6.5).
