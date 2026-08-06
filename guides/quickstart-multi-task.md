# Multi-task / multi-output quickstart

> **Install:** Install Session 2.x with `pip install buildml` (2.5.x on PyPI). Legacy 1.x remains available as `pip install "buildml==1.0.9"`.
> Core sklearn MultiOutput / Chain needs no extra; industry GBDT and torch
> multi-head use optional extras. See [installation](../docs/installation.rst).

Shared-feature multi-target fitting: assign multiple `role="target"` columns
(or pass `targets=`), `session.multitask.fit` on train only, then
`session.multitask.evaluate` / `session.multitask.predict` on holdout, and save a distinct
bundle. Backends: **sklearn** (core), **industry** (`buildml[multitask-industry]`),
**torch** (`buildml[torch]` for shared-trunk multi-head). Classical
`Session.fit` remains single-target.

**Proof:** [multi-target-underwriting](../proofs/multi-target-underwriting/) (+ Tier C MultiOutputClassifier twin).

**Go deeper:** [Multi-task deep](multi-task-deep.md) ·
[Artifacts](artifacts-checkpoints-bundles.md).

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
