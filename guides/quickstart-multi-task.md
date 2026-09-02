# Multi-task / multi-output quickstart

```bash
pip install buildml
```

Two or more target columns of the same type. Mixed classification plus
regression is refused on sklearn/industry. Classical `session.fit` stays
single-target. Default is sklearn `multi_output`.

[Multi-task deep](multi-task-deep.md) ·
[multi-target-underwriting](../proofs/multi-target-underwriting/)

```bash
pip install buildml
# optional industry depth:
pip install "buildml[multitask-industry,torch]"
```

```python
import numpy as np
import pandas as pd

from buildml import Session
from buildml.multitask import multitask_capability_matrix

print(multitask_capability_matrix()["default_backend_when_installed"])

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
print(ev.metrics)           # unweighted means across tasks
print(ev.per_task_metrics)  # per-target accuracy / F1

session.multitask.save_bundle("artifacts/multitask_bundle")
```

## Honest boundaries

| In scope | Out of scope |
| --- | --- |
| sklearn / industry GBDT / torch shared-trunk on shared features | Deep MTL research platform |
| Same-type tasks on sklearn/industry | Mixed cls+reg except torch multi-head |
| ≥2 targets via roles or `targets=` | Auto-switching classical `Session.fit` |
| Per-task + aggregate holdout metrics | Causal multi-task / federated MTL |
| Distinct `buildml.multitask_bundle.v1` | Session checkpoint embedding the plan |

Related next: meta-learning
(see [Meta-learning quickstart](quickstart-meta-learning.md)).
