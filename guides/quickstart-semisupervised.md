# Semi-supervised quickstart

> **Install first (GitHub):** PyPI `buildml` is still legacy 1.x and does **not**
> install Session 2.x. Install 2.x from GitHub (or an editable checkout).
> Semi-supervised methods use core sklearn — no optional extra is required.
> See [installation](../docs/installation.rst).

Scarce labels + abundant unlabeled train rows on the same `Session`: history,
explain catalog, and a distinct semi-supervised bundle. Unlabeled targets are
**NaN missingness** by default (mapped to sklearn `-1` internally).

**Go deeper:** [Semi-supervised deep](semisupervised-deep.md) ·
[Artifacts](artifacts-checkpoints-bundles.md) ·
[Self-supervised](quickstart-selfsupervised.md).

```bash
pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"
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

fit = session.fit_semisupervised(method="label_propagation", n_neighbors=7)
print(fit.n_labeled_train, fit.n_unlabeled_train, fit.method)

preds = session.predict_semisupervised(partition="test")
print(preds.n_rows, preds.predictions[:5])

ev = session.evaluate_semisupervised(partition="test")
print(ev.n_labeled_eval, ev.metrics)

bundle = session.save_semisupervised_bundle("artifacts/semisupervised_bundle")
```

Self-training variant:

```python
session.fit_semisupervised(
    method="self_training",
    base_estimator="logistic_regression",
    threshold=0.8,
)
print(session.evaluate_semisupervised(partition="test").metrics)
```

**Not this API:** anomaly novelty (normal-only detector fit), self-supervised
pretext (`fit_ssl_pretext`), or active learning
([quickstart-active-learning](quickstart-active-learning.md)).
