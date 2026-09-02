# Semi-supervised quickstart

```bash
pip install buildml
```

Scarce labels plus unlabeled train rows. Unlabeled means target NaN
(mapped to sklearn `-1`). Default is label propagation. Evaluate uses
labeled holdout rows only. This is not active learning and not SSL
pretext.

[Semi-supervised deep](semisupervised-deep.md) ·
[semi-label-efficiency](../proofs/semi-label-efficiency/)

```bash
pip install buildml
# Optional industry depth:
pip install "buildml[semisupervised-industry,torch,ssl]"
```

Recommended recipe: split on fully labeled data (so stratification works), then
blank a fraction of **train** targets only. Holdout stays labeled for honest eval.

```python
import numpy as np
import pandas as pd

from buildml import Session
from buildml.data.dataset import Dataset
from buildml.ingest.detect import schema_from_dataframe

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
session._dataset = Dataset.from_transformed(
    session.dataset,
    full,
    schema=schema_from_dataframe(full),
    roles=dict(session.dataset.roles),
)

fit = session.semisupervised.fit(method="label_propagation", n_neighbors=7)
print(fit.n_labeled_train, fit.n_unlabeled_train, fit.backend, fit.method)

preds = session.semisupervised.predict(partition="test")
print(preds.n_rows, preds.predictions[:5])

ev = session.semisupervised.evaluate(partition="test")
print(ev.n_labeled_eval, ev.metrics)

bundle = session.semisupervised.save_bundle("artifacts/semisupervised_bundle")
```

Industry pseudo-label (when XGBoost installed):

```python
session.semisupervised.fit(
    backend="industry",
    method="pseudo_label_xgb",
    threshold=0.8,
    max_self_train_iter=10,
)
print(session.semisupervised.evaluate(partition="test").metrics)
```

SSL → semi-supervised pipeline:

```python
session.ssl.fit_pretext(method="simclr_tabular", latent_dim=8, epochs=20)
session.ssl.transform(attach=True, partition="all")
session.semisupervised.fit(
    method="self_training",
    columns=list(session.ssl.plan.representation_columns),
    prefer_reduce_components=False,
)
```

**Not this API:** anomaly novelty (normal-only detector fit), active learning
([quickstart-active-learning](quickstart-active-learning.md)), or pure SSL pretext
without partial labels.
