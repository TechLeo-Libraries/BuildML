# Online / continual learning

```bash
pip install buildml
# River streaming + ADWIN / Page-Hinkley: pip install "buildml[online-industry]"
# replay / EWC tabular MLP: pip install "buildml[torch]"
```

Rows arrive in chunks. You warm-start on the first train slice, then
`partial_fit` on later train slices. Validation and test are for
scoring, never for updates.

`session.online.fit` defaults to estimator `sgd_classifier`. That name
is a sklearn estimator, so `backend=None` stays sklearn even when River
is installed. Pass `estimator="river_logistic"` with `backend=None` to
take industry if `buildml[online-industry]` imported cleanly. Pass
`estimator="replay_mlp"` with `backend=None` to take torch.
`backend="industry"` with the default `sgd_classifier` is refused:
pick a `river_*` estimator for that backend.

The API refuses holdout indices on `partial_fit`, and it refuses a
silent full `.fit` pretending to be online (`allow_refit_fallback`
defaults to False). You decide chunk size, whether to pass `classes=`
yourself, and whether to opt into a disclosed full-refit fallback.

Short on-ramp: [online quickstart](quickstart-online-learning.md).
Proof: [stream-fraud-online](../proofs/stream-fraud-online/).

## A first stream

`chunk_size` defaults to 50. `n_init` defaults to that same size when
you leave it `None`. Classifiers need a class vocabulary on first fit:
pass `classes=` or let it read the **full train target column** (labels
only, not holdout).

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

fit = session.online.fit(
    estimator="sgd_classifier",
    chunk_size=40,
    n_init=40,
)
print(fit.n_init_rows, fit.n_remaining_train)

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

`evaluate` defaults to validation and does not call `partial_fit`.
`predict` defaults to test. Same rule: inference only.

## Backends and estimators

| Backend | Extra | Estimators | Drift hooks |
| --- | --- | --- | --- |
| `sklearn` | none (`sgd_classifier` default) | `sgd_classifier`, `sgd_regressor`, `passive_aggressive_classifier`, `passive_aggressive_regressor`, `perceptron`, `multinomial_nb`, `bernoulli_nb` | `mean_shift` |
| `industry` | `buildml[online-industry]` | `river_logistic`, `river_hoeffding`, `river_pa`, `river_linear_regression`, `river_hoeffding_regressor` | `mean_shift`, `adwin`, `page_hinkley` |
| `torch` | `buildml[torch]` | `replay_mlp`, `ewc_mlp` | `mean_shift` |

Industry needs a successful River import, not just a wheel on disk.
Torch replay/EWC is a small tabular MLP, not a lifelong-learning
research suite.

`drift_detector=None` picks `adwin` on industry when River is there,
otherwise `mean_shift`. `adwin` and `page_hinkley` refuse on
sklearn/torch. Set `drift_detector="none"` if you do not want the
disclosure. `drift_disclose` defaults to True.

```python
session.online.capability_matrix()
```

## How chunks are ingested

| Source | Call | Cursor |
| --- | --- | --- |
| Next unused train rows | `session.online.partial_fit(n_rows=...)` | Advances |
| Explicit train indices | `session.online.partial_fit(indices=...)` | Advances past the max index |
| External aligned frame | `session.online.partial_fit(frame=...)` | Unchanged |

The external frame must carry the same feature and target columns as
the plan. Validation/test indices are refused. `evaluate(...,
drift_check=True)` (the default) can flag mean-shift against the init
chunk on every backend, and River error-stream detectors on industry.

Results expose `drift_detected` and `drift_notes` on the update and
eval objects. That is a disclosure, not a production drift platform.

## Bundles

`buildml.online_bundle.v1` stores the `OnlinePlan`: backend, estimator,
cursor, seen indices, update history, classes. A Session checkpoint
does not embed it. `trusted=False` on load unless the file is yours.

[Artifacts](artifacts-checkpoints-bundles.md)

## When it refuses

| What you see | What happened |
| --- | --- |
| Fit before a split | `assert_can_fit("train")` |
| `partial_fit` on validation/test indices | Updates are train (or your aligned frame) only |
| Missing extra for `river_*` / `replay_mlp` | Install the extra, or pick a sklearn estimator |
| Estimator without `partial_fit` | Refused unless `allow_refit_fallback=True` (disclosed full refit) |
| Classifier without a class vocabulary | Pass `classes=` or keep train targets readable on first fit |

This is not Kafka, Flink, or a distributed streaming product.

[Online quickstart](quickstart-online-learning.md)
