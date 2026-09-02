# Meta-learning quickstart

```bash
pip install buildml
```

Episodic few-shot on a task/group column. You need at least two distinct
train task ids. Default is tabular prototypical. Holdout is never used
for meta-train. This is not MAML-at-scale.

[Meta-learning deep](meta-learning-deep.md) ·
[few-shot-domain-adapt](../proofs/few-shot-domain-adapt/)

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
        rows.append({"x": float(x[0]), "y": float(x[1]), "label": label, "task_id": f"t{task}"})
frame = pd.DataFrame(rows)

session = (
    Session.ingest(frame)
    .set_roles({"x": "feature", "y": "feature", "label": "target", "task_id": "group"})
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

# Prefer train held-out tasks for a true task-disjoint episodic check
ev = session.metalearning.evaluate(partition="train", k_shot=3)
print(ev.metrics)
print(ev.novel_task_ids, ev.overlapping_task_ids)

session.metalearning.save_bundle("artifacts/metalearning_bundle")
```

## Honest boundaries

| In scope | Out of scope |
| --- | --- |
| Episodic few-shot via task/group column | Foundation-model / LLM meta-learning |
| `prototypical` nearest-centroid on tabular features | Learned ProtoNet embeddings / Torch rewrite |
| `warm_start` pooled sklearn init + support adapt | Full MAML / Reptile second-order meta-gradients |
| Train-only meta-train | Meta-training on validation/test |
| Distinct `buildml.metalearning_bundle.v1` | Session checkpoint embedding the plan |

Related next: federated learning
(see [Federated quickstart](quickstart-federated.md)).
