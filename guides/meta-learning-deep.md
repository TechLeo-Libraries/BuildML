# Meta-learning

```bash
pip install buildml
# tabular ProtoNet encoder: pip install "buildml[torch]"
# learn2learn MAML / Reptile: pip install "buildml[metalearning-industry,torch]"
```

Few-shot on a task or group column. Each task id is a small dataset.
You meta-train on train tasks, adapt on a support set, then score
episodes you tried not to meta-train on.

`session.metalearning.fit` defaults to method `prototypical`. That name
is a sklearn method, so `backend=None` stays sklearn even if torch is
installed. Pass `method="prototypical_torch"` with `backend=None` to
take torch. Pass `method="maml"` or `method="reptile"` with
`backend=None` to take industry when torch imported cleanly (learn2learn
if present, otherwise a native first-order SGD loop).
`backend="industry"` with the default `prototypical` method is refused.

The API refuses meta-training on validation/test, a fit without a split,
and null features. You decide the task column, `k_shot` / `n_query`,
and whether overlapping task ids on a random split are acceptable (they
are disclosed, not hidden).

Short on-ramp: [meta-learning quickstart](quickstart-meta-learning.md).
Proof: [few-shot-domain-adapt](../proofs/few-shot-domain-adapt/).
This is tabular few-shot, not Mini-ImageNet, not MAML-at-scale, and not
[causal](causal-deep.md) meta.

## Mental model

A **task** is the rows that share a task/group id (`role="group"` or
`task_column=`). You need at least two distinct train task ids.

**Meta-train** runs on the train partition only. By default
`task_holdout_fraction=0.25` holds some train task ids out internally
(`held_out_task_ids` on the plan).

An **episode** samples `k_shot` support rows per class plus `n_query`
query rows. Defaults: `k_shot=5`, `n_query=10`, `n_episodes=20`.
`n_way` is inferred when you leave it `None`.

**Adapt** freezes the meta-train plan and fits only on one task's
support set. **Evaluate** prefers novel task ids on holdout partitions
(`prefer_novel_tasks=True`) and lists `overlapping_task_ids` when a
random row split put the same id in more than one partition.

```python
import numpy as np
import pandas as pd

from buildml import Session

rng = np.random.default_rng(0)
rows = []
for task in range(8):
    shift = rng.normal(0, 1.0, size=2)
    for i in range(40):
        label = i % 2
        center = shift + (1.2 if label else -1.2)
        x = rng.normal(center, 0.45, size=2)
        rows.append(
            {
                "x": float(x[0]),
                "y": float(x[1]),
                "label": label,
                "task_id": f"t{task}",
            }
        )
frame = pd.DataFrame(rows)

session = (
    Session.ingest(frame)
    .set_roles(
        {
            "x": "feature",
            "y": "feature",
            "label": "target",
            "task_id": "group",
        }
    )
    .split(test_size=0.2, validation_size=0.2, random_state=0)
    .scale(method="standard")
)

fit = session.metalearning.fit(
    method="prototypical",
    k_shot=3,
    n_query=6,
    n_episodes=20,
    task_holdout_fraction=0.25,
)
print(fit.n_meta_train_tasks, fit.meta_train_accuracy)

adapt = session.metalearning.adapt(
    task_id=session.metalearning.plan.train_task_ids[0],
    partition="train",
    max_support_per_class=3,
)
print(adapt.n_support, adapt.n_classes_adapted)

ev = session.metalearning.evaluate(partition="train", k_shot=3)
print(ev.metrics)
print(ev.novel_task_ids, ev.overlapping_task_ids)

session.metalearning.save_bundle("artifacts/metalearning_bundle")
```

`evaluate` defaults to `partition="validation"`. Scoring on
`partition="train"` with the internally held-out task ids is the
cleaner task-disjoint check when a random split mixed ids.

## Methods

| Backend | Extra | Methods |
| --- | --- | --- |
| `sklearn` | none (`prototypical` default) | `prototypical`, `warm_start` |
| `torch` | `buildml[torch]` | `prototypical_torch` |
| `industry` | torch required; `buildml[metalearning-industry]` for learn2learn | `maml`, `reptile` |

`prototypical` is nearest-centroid on tabular features. No learned
neural embedding.

`warm_start` pools a `logistic_regression` / `sgd_classifier` meta-init,
then clones and refits on support. That is honest warm initialization,
not second-order MAML.

`prototypical_torch` is a small MLP encoder trained with episodic
prototype cross-entropy. Adapt and eval use embedding-space nearest
prototypes. Tabular ProtoNet, not vision ProtoNet claims.
`meta_epochs` defaults to 40.

`maml` / `reptile` are first-order tabular loops. Inner-loop SGD on
support (`inner_lr=0.05`, `inner_steps=5`), meta-update across
episodes. learn2learn when it imports; otherwise a native first-order
SGD loop, disclosed in the matrix. Not full second-order MAML-at-scale.

```python
session.metalearning.capability_matrix()
```

A working torch install is what makes `maml` / `reptile` callable.
learn2learn is optional. The capability matrix says which of those
actually imported.

## Episodic metrics

| Metric | Where |
| --- | --- |
| `meta_train_accuracy` | Fit result: episodic query accuracy during meta-train |
| `mean_accuracy`, `mean_f1_macro`, `n_tasks_scored` | Eval result |
| `held_out_task_ids` | Plan: internal train-task holdout from fit |
| `novel_task_ids` / `overlapping_task_ids` | Eval disclosure |

The surface is classification with a shared global label space.
Regression few-shot is not implemented here.

## Bundles

`buildml.metalearning_bundle.v1` stores the `MetaLearningPlan`:
backend, protocol, feature/task contract, label encoder, optional
warm-start init or torch/industry meta-learner. A Session checkpoint
does not embed it. `trusted=True` only for a file you made.

[Artifacts](artifacts-checkpoints-bundles.md)

## When it refuses

| What you see | What happened |
| --- | --- |
| Fit before a split | Meta-train is train only |
| Too few task ids | Need at least two distinct train tasks |
| Null features | Impute / scale first |
| `MissingExtraError` | Torch method without `buildml[torch]` |
| Unknown method | See the table above |

Random row splits may place the same task id in train and holdout.
`evaluate` tells you. Prefer `group_split` on the task column when you
need a hard boundary.

[Meta-learning quickstart](quickstart-meta-learning.md)
