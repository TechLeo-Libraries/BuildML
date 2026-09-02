# Decisions / optimisation quickstart

```bash
pip install buildml
```

Thresholds, cost matrices, top-K, knapsack, LP. Threshold / cost-matrix
paths need a prior `session.fit`. Tuning defaults to validation. Tuning
on test is refused unless `allow_test_tuning=True`. Not a general MIP
platform.

[Decisions deep](optimize-deep.md) ·
Paste: [`examples/decision_threshold_loop.py`](../examples/decision_threshold_loop.py) ·
Evidence: [cost-sensitive-collections](../proofs/cost-sensitive-collections/)

**Not** a general operations-research platform, arbitrary MIP suite, or digital twin.

---

## Capability matrix

```python
import pandas as pd

from buildml import Session

# Preferred: session.decision.capability_matrix on a Session instance.
# Flat Session.*_capability_matrix classmethods still work without data.
session = Session.ingest(pd.DataFrame({"score": [0.5], "y": [0]}))
print(session.decision.capability_matrix()["default_backend_when_installed"])
```

---

## Fit → apply / evaluate → bundle

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from buildml import Session

x, y = make_classification(
    n_samples=400,
    n_features=8,
    n_informative=5,
    weights=[0.7, 0.3],
    random_state=0,
)
frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
frame["y"] = y
frame["cost"] = np.where(y == 1, 2.0, 1.0)

session = (
    Session.ingest(frame)
    .set_roles({**{c: "feature" for c in frame.columns if c.startswith("f")}, "y": "target"})
    .split(test_size=0.25, validation_size=0.25, random_state=0)
    .fit(LogisticRegression(max_iter=500), task="classification")
)

# Prefer validation for policy selection (test requires allow_test_tuning=True)
fit = session.decision.fit(
    method="threshold",
    partition="validation",
    fp_cost=1.0,
    fn_cost=5.0,
    backend="native",  # or "xgb" / "calibrated" when installed
)
print(fit.to_dict())

applied = session.decision.apply(partition="test")
eval_result = session.decision.evaluate(partition="test")
print(eval_result.to_dict())

# MIP knapsack when optimize-industry is installed
session.decision.fit(
    method="knapsack",
    partition="validation",
    budget=40.0,
    cost_column="cost",
    score_source="model_proba",
    backend="pulp",  # auto-defaults to pulp/ortools when installed
)
print(session.decision.apply(partition="test").selected_ids[:10])

session.decision.save_bundle("artifacts/decision_demo_bundle")
```

---

## Leakage rules

- Default tuning partition is **`validation`**.
- Tuning on Session **test** requires `allow_test_tuning=True` and emits a
  dangerous-opt-in disclosure.
- Confirm a frozen policy once with `session.decision.evaluate(partition="test")`.

---

## Honesty

Decision helpers for ML scores/costs/allocations: scoped PuLP/OR-Tools MIP
knapsack and CVXPY LP only; not a general OR platform or digital twin.
`tune_threshold` remains the classical diagnostic sweep;
`session.decision.fit(method="threshold")` persists the chosen operating point.

Related: [quickstart-synthetic.md](quickstart-synthetic.md).
