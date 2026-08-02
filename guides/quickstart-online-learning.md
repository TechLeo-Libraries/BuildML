# Online / continual learning quickstart

> **Install first (GitHub):** PyPI `buildml` is still legacy 1.x and does **not**
> install Session 2.x. Install 2.x from GitHub (or an editable checkout).
> Online learning uses core sklearn `partial_fit` estimators — no optional extra.
> See [installation](../docs/installation.rst).

Incremental updates on **train chunks**: `fit_online` warm-starts on an initial
chunk, `partial_fit_online` updates on subsequent chunks, `evaluate_online`
scores held-out validation/test (never used for updates), then save a distinct
bundle. Honesty: this is batch/stream-chunk Session updating — **not** a
distributed streaming platform or lifelong-learning research suite.

**Go deeper:** [Online learning deep](online-learning-deep.md) ·
[Artifacts](artifacts-checkpoints-bundles.md).

```bash
pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"
```

```python
import numpy as np
import pandas as pd

from buildml import Session

rng = np.random.default_rng(0)
x0 = rng.normal([-1.0, -1.0], 0.55, size=(160, 2))
x1 = rng.normal([1.2, 1.0], 0.55, size=(160, 2))
frame = pd.DataFrame(np.vstack([x0, x1]), columns=["x", "y"])
frame["label"] = [0] * 160 + [1] * 160

session = (
    Session.ingest(frame)
    .set_roles({"x": "feature", "y": "feature", "label": "target"})
    .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0)
    .scale(method="standard")
)

# Warm-start on the first train chunk; classes discovered from train targets.
fit = session.fit_online(
    estimator="sgd_classifier",
    chunk_size=40,
    n_init=40,
)
print(fit.n_init_rows, fit.n_remaining_train)

# Stream remaining train in chunks (cursor advances automatically).
while True:
    plan = session.online_plan
    assert plan is not None
    remaining = plan.n_train_rows - plan.cursor
    if remaining <= 0:
        break
    update = session.partial_fit_online(n_rows=min(40, remaining))
    print(update.n_updates, update.n_seen_rows, update.update_mode)

ev = session.evaluate_online(partition="validation")
print(ev.metrics)

session.save_online_bundle("artifacts/online_bundle")
```

## Honest boundaries

| In scope | Out of scope |
| --- | --- |
| sklearn `partial_fit` family on Session train chunks | Distributed streaming / Kafka / Flink |
| Explicit `classes=` (or train-target discovery) | Silent full `.fit` pretending to be online |
| Holdout eval never used for updates | Lifelong / continual research suites (EWC, replay zoos, …) |
| Optional lite drift disclosure vs init chunk | Full production drift platform |

Next Phase 2 item after multi-task: **meta-learning**.
