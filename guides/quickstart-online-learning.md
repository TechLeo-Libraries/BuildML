# Online / continual learning quickstart

> **Install:** Install Session 2.x with `pip install buildml` (2.5.x). Legacy 1.x remains available as `pip install "buildml==1.0.9"`. Install 2.x from GitHub (or an editable checkout).
> Online learning uses core sklearn `partial_fit` estimators: no optional extra.
> See [installation](../docs/installation.rst).

Incremental updates on **train chunks**: `session.online.fit` warm-starts on an initial
chunk, `session.online.partial_fit` updates on subsequent chunks, `session.online.evaluate`
scores held-out validation/test (never used for updates), then save a distinct
bundle. Honesty: this is batch/stream-chunk Session updating: **not** a
distributed streaming platform or lifelong-learning research suite.

**Go deeper:** [Online learning deep](online-learning-deep.md) ·

**Proof:** [stream-fraud-online](../proofs/stream-fraud-online/) (+ Tier C SGD partial_fit). Cross-domain: [aegis-fraud-platform](../proofs/aegis-fraud-platform/).
[Artifacts](artifacts-checkpoints-bundles.md).

```bash
pip install buildml
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
fit = session.online.fit(
    estimator="sgd_classifier",
    chunk_size=40,
    n_init=40,
)
print(fit.n_init_rows, fit.n_remaining_train)

# Stream remaining train in chunks (cursor advances automatically).
while True:
    plan = session.online.plan
    assert plan is not None
    remaining = plan.n_train_rows - plan.cursor
    if remaining <= 0:
        break
    update = session.online.partial_fit(n_rows=min(40, remaining))
    print(update.n_updates, update.n_seen_rows, update.update_mode)

ev = session.online.evaluate(partition="validation")
print(ev.metrics)

session.online.save_bundle("artifacts/online_bundle")
```

## Honest boundaries

| In scope | Out of scope |
| --- | --- |
| sklearn `partial_fit` family on Session train chunks | Distributed streaming / Kafka / Flink |
| Explicit `classes=` (or train-target discovery) | Silent full `.fit` pretending to be online |
| Holdout eval never used for updates | Lifelong / continual research suites (EWC, replay zoos, …) |
| Optional lite drift disclosure vs init chunk | Full production drift platform |

Related next: meta-learning.
