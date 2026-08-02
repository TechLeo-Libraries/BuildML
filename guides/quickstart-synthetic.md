# Quickstart — Synthetic-data systems

> **Install (GitHub 2.x):**
> `pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"`
> Core path — bootstrap + Gaussian copula use numpy/scipy/sklearn.
> `method='smote'` needs `pip install "buildml[imbalanced]"`.
> See [installation](../docs/installation.rst).

Session path for **train-fitted tabular generators**: bootstrap / smoothed
bootstrap, Gaussian copula (mixed types), and optional SMOTE wrap.

**Distinct from** `Session.resample` (class-balance preprocess).
**Not** differential privacy. **Not** an SDV/CTGAN stack in core.

Runnable mirror: [`examples/synthetic_copula_loop.py`](../examples/synthetic_copula_loop.py).
Deep guide: [synthetic-deep.md](synthetic-deep.md).

---

## Fit → sample / evaluate → bundle

```python
import pandas as pd
from sklearn.datasets import make_classification

from buildml import Session

x, y = make_classification(
    n_samples=400,
    n_features=6,
    n_informative=4,
    weights=[0.7, 0.3],
    random_state=0,
)
frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
frame["y"] = y
frame["grp"] = pd.Series(y).map({0: "A", 1: "B"})  # mixed-type column

session = (
    Session.ingest(frame)
    .set_roles(
        {
            **{c: "feature" for c in frame.columns if c.startswith("f")},
            "grp": "feature",
            "y": "target",
        }
    )
    .split(test_size=0.25, validation_size=0.25, random_state=0)
)

fit = session.fit_synthesizer(method="gaussian_copula", random_state=0)
print(fit.to_dict())

# Default: return a Frame — does not mutate Session roles/splits
sample = session.sample_synthetic(n=200, random_state=1)
print(sample.frame.head())

# Fidelity vs real test; or mode='tstr' for train-on-synthetic test-on-real
fid = session.evaluate_synthetic(mode="fidelity", partition="test")
print(fid.metrics)

tstr = session.evaluate_synthetic(mode="tstr", partition="test")
print(tstr.metrics)

# Optional: append into train with provenance (role=ignore)
session.sample_synthetic(n=50, merge_mode="extend_train", provenance_column="_synthetic")

session.save_synthetic_bundle("artifacts/synthetic_demo_bundle")
```

### Cross-link: `resample`

```python
# Class balance only (mutates train membership) — still supported:
# session.resample(sampler="smote")
# Reusable SMOTE generator (optional extra):
# session.fit_synthesizer(method="smote")
```

---

## Honesty

| Claim | Reality |
| --- | --- |
| Train-only fit | Always; never fits on validation/test |
| Privacy | **Not** DP; bootstrap can near-duplicate rows |
| Merge | Explicit `merge_mode`; default returns Frame only |
| Heavy stacks | No SDV/CTGAN required in core |

Phase tracker: recommenders / LTR / KG / optimize / **synthetic** = Phase-3 PASS.
