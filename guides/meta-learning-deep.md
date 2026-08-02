# Meta-learning deep guide

Practical Session-facing meta-learning for tabular few-shot / episodic
protocols. This is **not** a foundation-model meta-learning platform and does
**not** claim MAML-at-scale.

## Mental model

1. A **task** is the set of rows sharing a task/group id.
2. **Meta-train** runs on the train partition only (optionally holding out a
   fraction of train task ids for internal checks).
3. An **episode** samples a balanced support set (`k_shot` per class) and a
   query set from one task.
4. **Adapt** freezes the meta-train plan and fits only on a support set.
5. **Evaluate** prefers novel task ids on holdout partitions; overlapping
   task ids are allowed only with a clear disclosure.

## Algorithms

### `prototypical`

Tabular nearest-centroid few-shot (ProtoNet-style geometry without a learned
neural embedding):

- Support → class mean vectors in feature space
- Query → nearest prototype (euclidean)

Features may already be Session-scaled / reduced. Honesty: this is a complete
practical baseline, not a claimed neural ProtoNet.

### `warm_start`

1. Fit a pooled `logistic_regression` or `sgd_classifier` on meta-train rows
   (`init_estimator_`).
2. `adapt_to_task` clones that init and refits on the support set.

Honesty: transfer via warm initialization + fast adapt — **not** MAML /
Reptile second-order meta-gradients.

## Session API

| Method | Role |
| --- | --- |
| `fit_metalearning` | Meta-train on train tasks |
| `adapt_to_task` | Fast adapt to one task support set |
| `evaluate_metalearning` | Episodic holdout metrics |
| `save_metalearning_bundle` / `load_metalearning_bundle` | `buildml.metalearning_bundle.v1` |

Properties: `metalearning_plan`, `metalearning_fit_result`,
`metalearning_adapt_result`, `metalearning_eval_result`.

## Task column

- Prefer `role="group"` on the task identifier column, **or**
- Pass `task_column=` explicitly.

The task column is excluded from features. Exactly one `role="target"` is
required (multi-target joint fitting belongs on `fit_multitask`).

## Leakage discipline

- Meta-train requires a split and uses **train only**.
- Validation/test are never used for meta-training.
- `evaluate_metalearning` discloses when holdout task ids overlap meta-train
  (not true out-of-task generalization). Prefer task-disjoint splits or the
  fit-time `held_out_task_ids` when evaluating `partition="train"`.
- Null features are refused (impute/scale first).

## Bundles vs checkpoints

`buildml.metalearning_bundle.v1` stores `MetaLearningPlan` (protocol +
feature/task contract + label encoder + optional warm-start init). Session
checkpoints do **not** embed the meta-learner. See
[Artifacts](artifacts-checkpoints-bundles.md).

## AI allowlist

Teaching-critical tools: `fit_metalearning`, `adapt_to_task`,
`evaluate_metalearning`, plus save/load bundle.

## Residuals (honest)

- Classification-focused surface (shared global label space).
- No Torch meta-learning / learned embeddings.
- No MAML/Reptile second-order loops.
- Random row splits may place the same task id in train and holdout — disclosed.

Next Phase 2 item after meta-learning (now shipped): **federated learning**
(see [Federated deep](federated-deep.md)).
