# Multi-task / multi-output quickstart

> **Install first (GitHub):** PyPI `buildml` is still legacy 1.x and does **not**
> install Session 2.x. Install 2.x from GitHub (or an editable checkout).
> Multi-task uses core sklearn MultiOutput / Chain façades — no optional extra.
> See [installation](../docs/installation.rst).

Shared-feature multi-target fitting: assign multiple `role="target"` columns
(or pass `targets=`), `fit_multitask` on train only, then
`evaluate_multitask` / `predict_multitask` on holdout, and save a distinct
bundle. Honesty: sklearn MultiOutput / Chain — **not** a deep multi-head MTL
research platform. Classical `Session.fit` remains single-target.

**Go deeper:** [Multi-task deep](multi-task-deep.md) ·
[Artifacts](artifacts-checkpoints-bundles.md).

```bash
pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"
```

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

fit = session.fit_multitask(
    method="multi_output",
    task="classification",
    base_estimator="logistic_regression",
)
print(fit.n_tasks, fit.target_columns)

ev = session.evaluate_multitask(partition="validation")
print(ev.metrics)           # unweighted means across tasks
print(ev.per_task_metrics)  # per-target accuracy / F1

session.save_multitask_bundle("artifacts/multitask_bundle")
```

## Honest boundaries

| In scope | Out of scope |
| --- | --- |
| sklearn `MultiOutput*` / `*Chain` on shared features | Deep multi-head MTL / Torch rewrite |
| Same-type tasks (all cls or all reg) | Mixed classification + regression targets |
| ≥2 targets via roles or `targets=` | Auto-switching classical `Session.fit` |
| Per-task + aggregate holdout metrics | Causal multi-task / federated MTL |
| Distinct `buildml.multitask_bundle.v1` | Session checkpoint embedding the plan |

Next Phase 2 item after multi-task (now shipped): **meta-learning**
(see [Meta-learning quickstart](quickstart-meta-learning.md)).
