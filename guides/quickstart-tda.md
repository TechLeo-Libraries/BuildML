# TDA quickstart

```bash
pip install "buildml[tda]"
```

Local Vietoris-Rips on train neighborhoods, then a sklearn head. Default
vectorization is a persistence image. Needs `buildml[tda]`. If
`buildml[tda-industry]` is installed, `backend=None` picks giotto for
that default image. Not a Mapper suite.

[TDA deep](tda-deep.md) ·
[credit-tda-shape](../proofs/credit-tda-shape/)

---

## Fit → evaluate → bundle

```python
import numpy as np
import pandas as pd
from buildml import Session

rng = np.random.default_rng(0)
# Two blobs with different local geometry
a = rng.normal(size=(120, 4)) + np.array([0, 0, 0, 0])
b = rng.normal(size=(120, 4)) * 1.8 + np.array([3, 0, 0, 0])
x = np.vstack([a, b])
y = np.array([0] * 120 + [1] * 120)
frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(4)])
frame["y"] = y

session = (
    Session.ingest(frame)
    .set_roles({**{f"f{i}": "feature" for i in range(4)}, "y": "target"})
    .split(test_size=0.2, validation_size=0.2, random_state=0, stratify=True)
    .scale(method="standard")
)

fit = session.tda.fit(
    vectorization="persistence_image",
    knn=12,
    n_bins=12,
    head="logistic_regression",
)
print(fit.feature_dim, fit.train_score)

feats = session.tda.transform(partition="test")
print(feats.features.shape)

ev = session.tda.evaluate(partition="validation")
print(ev.metrics)

session.tda.save_bundle("artifacts/tda_demo_bundle")
```

---

## Vectorization choices

| `vectorization` | Backend | Notes |
|-----------------|---------|-------|
| `persistence_image` | persim / gtda | Default; birth×persistence raster |
| `landscape` | in-tree / gtda | Layered tents on a train-fitted t-grid |
| `silhouette` | in-tree | Weighted average of tents (native only) |
| `betti_curve` | gtda | Industry backend only |
| `persistence_landscape` | gtda | Industry backend only |

```python
# Preferred on a Session instance; flat Session.*_capability_matrix still works.
session.tda.capability_matrix()  # honest backend / vectorization matrix
```

---

## Leakage / honesty

| Rule | Behavior |
|------|----------|
| Fit | Train-only NN index, vectorizer ranges, optional head |
| Transform / eval | Frozen pipeline; no refit on holdout |
| Extra | `buildml[tda]` native; `buildml[tda-industry]` giotto |
| Scope | PH + vectorization → sklearn: not Mapper-at-scale |

---

## Related

TDA industry depth is shipped. See also
[recommendation systems](quickstart-recommenders.md), search/LTR, knowledge
graphs, optimisation helpers, and synthetic-data guides.
