# Semi-supervised (deep)

```bash
pip install buildml
# GBDT pseudo-label: pip install "buildml[semisupervised-industry]"
# FixMatch: pip install "buildml[torch]"
```

Scarce labels on train, unlabeled train features still used. Unlabeled
means target NaN (or your `unlabeled_marker`). Internally that is sklearn
`-1`. Default `method="label_propagation"` stays sklearn even if XGBoost
is installed. Holdout labels are for evaluation only.

This is not novelty detection (`session.anomaly`), not pretext
(`session.ssl`), and not an oracle (`session.active_learning`).

Quickstart: [semi-supervised quickstart](quickstart-semisupervised.md).

## Backends

| Backend | Extra | Methods |
| --- | --- | --- |
| `sklearn` | core | `label_propagation`, `label_spreading`, `self_training` |
| `industry` | `semisupervised-industry` | `pseudo_label_xgb`, `pseudo_label_lgbm` |
| `torch` | `torch` | `fixmatch_tabular`, `mixmatch_tabular` |
| `hf` | `ssl` | `text_pseudo_label` |

```python
import numpy as np
import pandas as pd

from buildml import Session

rng = np.random.default_rng(0)
x0 = rng.normal([-1.0, -1.0], 0.6, size=(120, 2))
x1 = rng.normal([1.2, 1.0], 0.6, size=(120, 2))
frame = pd.DataFrame(np.vstack([x0, x1]), columns=["x", "y"])
frame["label"] = [0] * 120 + [1] * 120

session = (
    Session.ingest(frame)
    .set_roles({"x": "feature", "y": "feature", "label": "target"})
    .split(test_size=0.25, stratify=True, random_state=0)
    .scale(method="standard")
)

# Scarce labels on TRAIN only (holdout remains fully labeled).
full = session.to_pandas().copy()
train_idx = list(session.split_plan.train_indices)
blank = rng.choice(train_idx, size=int(0.7 * len(train_idx)), replace=False)
full.loc[blank, "label"] = np.nan
# Re-ingest the masked table while preserving the original row assignments.
session = (Session.ingest(full)
    .set_roles(dict(session.dataset.roles))
    .inject_split(train_indices=session.split_plan.train_indices,
                  validation_indices=session.split_plan.validation_indices,
                  test_indices=session.split_plan.test_indices))

session.semisupervised.fit(
    backend="industry",
    method="pseudo_label_xgb",
    threshold=0.75,
    max_self_train_iter=10,
)
```

## The loop

Split first. If you need `stratify=True`, start from fully labeled data,
then blank **train** targets to simulate scarce labels. Fit on train.
Predict / evaluate holdout. Bundle: `buildml.semisupervised_bundle.v1`.
A Session checkpoint does not embed the plan.

Read `n_labeled_*` / `n_unlabeled_*` beside every metric.

## With a pretext

Self-supervised pretext can run on all train rows (labels optional).
Semi-supervised fit then uses partial labels on those representations:

```python
import numpy as np
import pandas as pd

from buildml import Session

rng = np.random.default_rng(0)
x0 = rng.normal([-1.0, -1.0], 0.6, size=(120, 2))
x1 = rng.normal([1.2, 1.0], 0.6, size=(120, 2))
frame = pd.DataFrame(np.vstack([x0, x1]), columns=["x", "y"])
frame["label"] = [0] * 120 + [1] * 120

session = (
    Session.ingest(frame)
    .set_roles({"x": "feature", "y": "feature", "label": "target"})
    .split(test_size=0.25, stratify=True, random_state=0)
    .scale(method="standard")
)

# Scarce labels on TRAIN only (holdout remains fully labeled).
full = session.to_pandas().copy()
train_idx = list(session.split_plan.train_indices)
blank = rng.choice(train_idx, size=int(0.7 * len(train_idx)), replace=False)
full.loc[blank, "label"] = np.nan
# Re-ingest the masked table while preserving the original row assignments.
session = (Session.ingest(full)
    .set_roles(dict(session.dataset.roles))
    .inject_split(train_indices=session.split_plan.train_indices,
                  validation_indices=session.split_plan.validation_indices,
                  test_indices=session.split_plan.test_indices))

session.ssl.fit_pretext(method="simclr_tabular", latent_dim=16, epochs=30)
session.ssl.transform(attach=True, partition="all")
session.semisupervised.fit(
    method="self_training",
    columns=list(session.ssl.plan.representation_columns),
    prefer_reduce_components=False,
)
```

`session.ssl.finetune_head` is labeled train only. `session.semisupervised.fit`
uses unlabeled train rows via propagation or pseudo-labels.

## What usually goes wrong

- Fewer than two labeled train rows, or a single class among labels.
- Null feature columns: impute/scale first.
- Missing extra for a non-sklearn backend: `MissingExtraError`.
- Using validation/test unlabeled rows to invent labels for selection.

[Self-supervised](selfsupervised-deep.md) · [Active learning](active-learning-deep.md)
