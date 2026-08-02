# Quickstart — Optimisation / decision helpers

**Proof:** [cost-sensitive-collections](../proofs/cost-sensitive-collections/) · Tier B allocation in [harbor-demand-desk](../proofs/harbor-demand-desk/), [aegis-fraud-platform](../proofs/aegis-fraud-platform/), and [ledger-underwriting-studio](../proofs/ledger-underwriting-studio/).

> **Install (GitHub 2.x):**
> `pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"`
> Core path — sklearn + transitive `scipy.optimize` for LP allocation.
> Industry solvers: `pip install 'buildml[optimize-industry]'` (PuLP, OR-Tools,
> CVXPY, XGBoost). See [installation](../docs/installation.rst).

Session decision helpers over **ML scores, costs, and constrained allocations**.
Cost-sensitive thresholds wrap the same engine as classical `tune_threshold`
(or XGB/calibrated when installed); multiclass cost matrices, top-K capacity,
knapsack (native or MIP), and continuous LP budget shares persist as a
`DecisionPlan` bundle.

**Not** a general operations-research platform, arbitrary MIP suite, or digital twin.

Runnable mirror: [`examples/decision_threshold_loop.py`](../examples/decision_threshold_loop.py).
Deep guide: [optimize-deep.md](optimize-deep.md).

---

## Capability matrix

```python
from buildml import Session

print(Session.decision_capability_matrix()["default_backend_when_installed"])
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
fit = session.fit_decision_policy(
    method="threshold",
    partition="validation",
    fp_cost=1.0,
    fn_cost=5.0,
    backend="native",  # or "xgb" / "calibrated" when installed
)
print(fit.to_dict())

applied = session.apply_decisions(partition="test")
eval_result = session.evaluate_decisions(partition="test")
print(eval_result.to_dict())

# MIP knapsack when optimize-industry is installed
session.fit_decision_policy(
    method="knapsack",
    partition="validation",
    budget=40.0,
    cost_column="cost",
    score_source="model_proba",
    backend="pulp",  # auto-defaults to pulp/ortools when installed
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

Decision helpers for ML scores/costs/allocations — scoped PuLP/OR-Tools MIP
knapsack and CVXPY LP only; not a general OR platform or digital twin.
`tune_threshold` remains the classical diagnostic sweep;
`fit_decision_policy(method="threshold")` persists the chosen operating point.

Phase-3 synthetic-data systems: **PASS** → [quickstart-synthetic.md](quickstart-synthetic.md).
