# Quickstart — Optimisation / decision helpers

> **Install (GitHub 2.x):**
> `pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"`
> Core path — uses sklearn + transitive `scipy.optimize` for LP allocation.
> No PuLP / OR-Tools extra. See [installation](../docs/installation.rst).

Session decision helpers over **ML scores, costs, and constrained allocations**.
Cost-sensitive thresholds wrap the same engine as classical `tune_threshold`;
multiclass cost matrices, top-K capacity, knapsack-lite, and continuous LP
budget shares persist as a `DecisionPlan` bundle.

**Not** a general operations-research platform, MIP suite, or digital twin.

Runnable mirror: [`examples/decision_threshold_loop.py`](../examples/decision_threshold_loop.py).
Deep guide: [optimize-deep.md](optimize-deep.md).

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
fit = session.fit_decision_policy(
    method="threshold",
    partition="validation",
    fp_cost=1.0,
    fn_cost=5.0,
)
print(fit.to_dict())

# Classical diagnostic explorer still available (same cost engine):
# session.tune_threshold(partition="validation", fp_cost=1.0, fn_cost=5.0)

applied = session.apply_decisions(partition="test")
print(applied.to_dict())

eval_result = session.evaluate_decisions(partition="test")
print(eval_result.to_dict())

# Allocation example: top-K under capacity using model scores + row costs
session.fit_decision_policy(
    method="knapsack",
    partition="validation",
    budget=40.0,
    cost_column="cost",
    score_source="model_proba",
)
print(session.apply_decisions(partition="test").selected_ids[:10])

session.save_decision_bundle("artifacts/decision_demo_bundle")
```

---

## Leakage rules

- Default tuning partition is **`validation`**.
- Tuning on Session **test** requires `allow_test_tuning=True` and emits a
  dangerous-opt-in disclosure.
- Confirm a frozen policy once with `evaluate_decisions(partition="test")`.

---

## Honesty

Decision helpers for ML scores/costs/allocations — not PuLP/OR-Tools, not a
digital twin. `tune_threshold` remains the classical diagnostic sweep;
`fit_decision_policy(method="threshold")` persists the chosen operating point.

Phase-3 synthetic-data systems: **PASS** → [quickstart-synthetic.md](quickstart-synthetic.md).
