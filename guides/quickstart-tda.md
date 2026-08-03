# Quickstart: Topological Data Analysis (TDA)

**Proof:** [credit-tda-shape](../proofs/credit-tda-shape/) (+ Tier C logistic twin on raw features).

> **Install (GitHub 2.x):**
> `pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"`
> TDA path: `pip install "buildml[tda]"` (ripser + persim)
> Industry: `pip install "buildml[tda-industry]"` (giotto-tda + Betti curves)
> See [installation](../docs/installation.rst).

Local Vietoris–Rips persistence on kNN train neighborhoods, train-fitted
vectorization (persistence images / landscapes / silhouettes), and an optional
sklearn head. **Not** a Mapper research suite or every TDA paper.

Runnable mirror: [`examples/tda_loop.py`](../examples/tda_loop.py).
Deep guide: [tda-deep.md](tda-deep.md).

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

fit = session.fit_tda(
    vectorization="persistence_image",
    knn=12,
    n_bins=12,
    head="logistic_regression",
)
print(fit.feature_dim, fit.train_score)

feats = session.transform_tda(partition="test")
print(feats.features.shape)

ev = session.evaluate_tda(partition="validation")
print(ev.metrics)

session.save_tda_bundle("artifacts/tda_demo_bundle")
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
Session.tda_capability_matrix()  # honest backend / vectorization matrix
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

## Next

TDA **PASS (R5.9 industry depth)**. Phase 3 application systems start at
[recommendation systems](quickstart-recommenders.md); after that PASS:
search/LTR, knowledge graphs, optimisation helpers, synthetic-data: one at a
time.
