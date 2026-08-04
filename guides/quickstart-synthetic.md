# Quickstart: Synthetic-data systems

> **Install (GitHub 2.x):**
> `pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"`
> Core path: bootstrap + Gaussian copula use numpy/scipy/sklearn.
> `method='smote'` needs `pip install "buildml[imbalanced]"`.
> SDV CTGAN/TVAE/CopulaGAN needs `pip install "buildml[synthetic-industry]"`.
> See [installation](../docs/installation.rst).

Session path for **train-fitted tabular generators**: native bootstrap / copula /
SMOTE, plus optional SDV industry backends when installed.

**Distinct from** `Session.resample` (class-balance preprocess).
**Not** differential privacy.

**Proof:** [synthetic-privacy-utility](../proofs/synthetic-privacy-utility/) (+ Tier C column-bootstrap twin).

Runnable mirror: [`examples/synthetic_copula_loop.py`](../examples/synthetic_copula_loop.py).
Deep guide: [synthetic-deep.md](synthetic-deep.md).

---

## Capability matrix

```python
import pandas as pd

from buildml import Session

# Preferred: session.synthetic.capability_matrix on a Session instance.
# Flat Session.*_capability_matrix classmethods still work without data.
session = Session.ingest(pd.DataFrame({"x": [0.0]}))
print(session.synthetic.capability_matrix())
```

---

## Fit → sample / evaluate → bundle (native)

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
frame["grp"] = pd.Series(y).map({0: "A", 1: "B"})

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

fit = session.synthetic.fit(backend="native", method="gaussian_copula", random_state=0)
sample = session.synthetic.sample(n=200, random_state=1, validate=True)
fid = session.synthetic.evaluate(mode="fidelity", partition="test", eval_backend="auto")
tstr = session.synthetic.evaluate(mode="tstr", partition="test")
session.synthetic.save_bundle("artifacts/synthetic_demo_bundle")
```

## SDV industry path (when installed)

```python
# pip install "buildml[synthetic-industry]"
session.synthetic.fit(backend="sdv", method="ctgan", epochs=100, batch_size=256)
session.synthetic.sample(n=300)
session.synthetic.evaluate(mode="fidelity", eval_backend="auto")  # + SDMetrics when installed
```

### Cross-link: `resample`

Prefer `Session.resample` for class balance only. Prefer `session.synthetic.fit` for
reusable sampling, fidelity/TSTR, and controlled augmentation with provenance.

---

## Honesty

| Claim | Reality |
| --- | --- |
| Train-only fit | Always; never fits on validation/test |
| Privacy | **Not** DP; bootstrap/SDV can memorize train structure |
| Merge | Explicit `merge_mode`; default returns Frame only |
| Industry | SDV optional: native copula always available |

Industry depth is shipped. Benchmark: `benchmarks/synthetic/tstr_quality.py`.
