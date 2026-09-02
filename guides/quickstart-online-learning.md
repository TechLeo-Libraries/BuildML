# Online / continual learning quickstart

```bash
pip install buildml
```

`fit` on a train chunk, then `partial_fit` on later train chunks.
Validation and test are never updated. Silent full refits are refused
(`allow_refit_fallback` is off). Default estimator is SGD classifier.
This is not a distributed streaming product.

[Online deep](online-learning-deep.md) ·
Paste: [`examples/online_partial_fit_loop.py`](../examples/online_partial_fit_loop.py) ·
Evidence: [stream-fraud-online](../proofs/stream-fraud-online/)

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
