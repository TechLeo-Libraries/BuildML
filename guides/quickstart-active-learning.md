# Active learning quickstart

> **Install first (GitHub):** PyPI `buildml` is still legacy 1.x and does **not**
> install Session 2.x. Install 2.x from GitHub (or an editable checkout).
> Active learning uses core sklearn: no optional extra is required.
> See [installation](../docs/installation.rst).

Human-in-the-loop labeling on the **train** pool: fit a learner on scarce seed
labels, `session.active_learning.suggest_query` the most uncertain train rows, `session.active_learning.label_rows` with
**user-provided** labels, refit, evaluate labeled holdout, save a distinct
bundle. Pool convention matches semi-supervised: **NaN targets** mark unlabeled
train rows. Validation/test are never the query pool. Core never invents an
oracle (examples/tests may simulate one).

**Go deeper:** [Active learning deep](active-learning-deep.md) ·

**Proof:** [active-labeling-budget](../proofs/active-labeling-budget/) (+ Tier C margin twin). Cross-domain: [atlas-label-studio](../proofs/atlas-label-studio/).
[Artifacts](artifacts-checkpoints-bundles.md) ·
[Semi-supervised](quickstart-semisupervised.md).

```bash
pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"
```

```python
import numpy as np
import pandas as pd

from buildml import Session
from buildml.data.dataset import Dataset
from buildml.ingest.detect import schema_from_dataframe

rng = np.random.default_rng(0)
x0 = rng.normal([-1.0, -1.0], 0.55, size=(140, 2))
x1 = rng.normal([1.2, 1.0], 0.55, size=(140, 2))
frame = pd.DataFrame(np.vstack([x0, x1]), columns=["x", "y"])
frame["label"] = [0] * 140 + [1] * 140
# Keep a hidden copy for the *example* oracle only (not used by library core).
truth = frame["label"].copy()

session = (
    Session.ingest(frame)
    .set_roles({"x": "feature", "y": "feature", "label": "target"})
    .split(test_size=0.25, stratify=True, random_state=0)
    .scale(method="standard")
)

# Seed: blank most TRAIN labels (holdout stays labeled for honest eval).
full = session.to_pandas().copy()
train_idx = list(session.split_plan.train_indices)
blank = rng.choice(train_idx, size=int(0.85 * len(train_idx)), replace=False)
full.loc[blank, "label"] = np.nan
session._dataset = Dataset.from_transformed(
    session.dataset,
    full,
    schema=schema_from_dataframe(full),
    roles=dict(session.dataset.roles),
)

fit = session.active_learning.fit(
    strategy="margin",
    base_estimator="logistic_regression",
    batch_size=8,
    label_budget=24,
)
print(fit.n_labeled_train, fit.n_unlabeled_pool, fit.strategy)

for round_i in range(3):
    q = session.active_learning.suggest_query(batch_size=8)
    if not q.indices:
        break
    # Example-only simulated oracle: production code must use a human labeler.
    human_labels = [int(truth.loc[i]) for i in q.indices]
    labeled = session.active_learning.label_rows(indices=q.indices, labels=human_labels)
    print(round_i, labeled.n_newly_labeled, labeled.n_labeled_now, labeled.budget_remaining)

ev = session.active_learning.evaluate(partition="test")
print(ev.n_labeled_eval, ev.metrics)

bundle = session.active_learning.save_bundle("artifacts/activelearning_bundle")
```

**Strategies:** `least_confidence`, `margin`, `entropy`, `committee` (bagged
vote entropy), `expected_model_change_lite`.

**Not this API:** semi-supervised propagation (`session.semisupervised.fit`),
self-supervised pretext (`session.ssl.fit_pretext`), or online/`partial_fit` streams
(`session.online.fit` / `session.online.partial_fit`)
(see active-learning / online guides).
