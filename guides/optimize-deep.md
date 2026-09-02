# Decisions under costs and capacity

```bash
pip install buildml
# PuLP / OR-Tools knapsack, CVXPY LP, XGB thresholds:
# pip install "buildml[optimize-industry]"
```

The classifier is already fitted. You still have to act: approve, chase,
spend a budget, pick a top-K. `session.decision.*` turns scores into a
frozen policy. It is not a general MIP suite, not a fleet scheduler, and
not a replacement for Optuna or `session.fit`.

Default method is `threshold`. Default tuning partition is
`validation`. Default `score_source` is `model_proba`. Threshold and
cost-matrix need a prior `session.fit`. Tuning on test raises
`LeakageError` unless you pass `allow_test_tuning=True` (disclosed as a
dangerous opt-in). Allocation (`topk`, `knapsack`, `lp_allocate`) can
read scores and costs from columns instead of the model.

Resolver when `backend=None`:

- F1 threshold (no `fp_cost` / `fn_cost`) → `native`
- Cost-sensitive threshold → `xgb` if installed, else `native`
- `knapsack` → `pulp`, then `ortools`, then `native`
- `lp_allocate` → `native` (scipy HiGHS). Pass `backend="cvxpy"` yourself.

You choose costs, capacity, and whether a test-tuned number is worth
the leak. The API refuses a missing split, threshold/cost-matrix without
`session.fit`, and test tuning without the flag.

Short on-ramp: [decisions quickstart](quickstart-optimize.md). Proof:
[cost-sensitive-collections](../proofs/cost-sensitive-collections/).

## Threshold after a classical fit

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
    .set_roles(
        {**{c: "feature" for c in frame.columns if c.startswith("f")}, "y": "target"}
    )
    .split(test_size=0.25, validation_size=0.25, random_state=0)
    .fit(LogisticRegression(max_iter=500), task="classification")
)

fit = session.decision.fit(
    method="threshold",
    partition="validation",
    fp_cost=1.0,
    fn_cost=5.0,
    backend="native",
)
print(fit.threshold, fit.recommendation_basis)

applied = session.decision.apply(partition="test")
eval_result = session.decision.evaluate(partition="test")
print(eval_result.metrics)
```

With `fp_cost` / `fn_cost`, the sweep minimizes expected cost:

`fp_cost·FP + fn_cost·FN − tp_benefit·TP − tn_benefit·TN`

Without those costs it recommends best F1 on the tuning partition.
`session.tune_threshold` remains the diagnostic explorer.
`session.decision.fit(method="threshold")` uses the same sweep (or an
industry scorer), stores a reusable `DecisionPlan`, and also updates
the last diagnostic report on the native/calibrated path.

`session.fit` stays the tabular model. Industry threshold backends
train an auxiliary estimator stored on the plan for apply.

## Backends

| Backend | Methods | Extra | Notes |
| --- | --- | --- | --- |
| `native` | all | core | threshold sweep, numpy knapsack, scipy linprog |
| `calibrated` | `threshold` | core | CalibratedClassifierCV on train, then cost sweep |
| `xgb` | `threshold` | `optimize-industry` | `scale_pos_weight` + validation sweep |
| `pulp` | `knapsack` | `optimize-industry` | Exact 0-1 MIP (CBC) |
| `ortools` | `knapsack` | `optimize-industry` | Exact 0-1 MIP |
| `cvxpy` | `lp_allocate` | `optimize-industry` | Convex LP, same class as linprog |

`cost_matrix`, `topk` always use native routing. Ask for `pulp` or
`cvxpy` without the extra and you get `MissingExtraError`.

## Methods

### `threshold`

Binary probabilistic classifiers. Needs `session.fit` because
`score_source` defaults to `model_proba`. Pass `backend="calibrated"`
or `"xgb"` when you want those auxiliary heads.

### `cost_matrix`

You supply a square `C[true, action]`. For each row the plan chooses

`argmin_a Σ_y P(y|x) C[y, a]`

using `predict_proba`. The matrix is not estimated from test labels.
Also requires `session.fit`.

```python
session.decision.fit(
    method="cost_matrix",
    partition="validation",
    cost_matrix=[[0.0, 1.0], [5.0, 0.0]],
    class_labels=["0", "1"],
)
```

### `topk`

Select up to `capacity` highest scores, optional `min_score` floor.
Scores from `model_proba`, `model_decision_function`, or columns
(`score_source="column"` plus `score_column`).

### `knapsack`

Maximize value under `budget`. Native: exact integer DP when costs are
near-integral and the state stays bounded; otherwise density-greedy,
disclosed. Industry: exact 0-1 MIP via PuLP or OR-Tools. Native
`knapsack_solver` is `"dp"` (or `"greedy"`).

```python
session.decision.fit(
    method="knapsack",
    partition="validation",
    budget=40.0,
    cost_column="cost",
    score_source="model_proba",
)
print(session.decision.apply(partition="test").selected_ids[:10])
```

When `backend=None`, knapsack prefers PuLP, then OR-Tools, then
native. Pass `backend="native"` if you want the DP/greedy path on a
machine that has PuLP.

### `lp_allocate`

Continuous shares `0 ≤ x_i ≤ lp_max_fraction` (default 1.0) under a
budget. Native: scipy linprog (HiGHS). CVXPY only if you pass
`backend="cvxpy"`. Fractional by design: not integer MIP.

Allocation can skip `session.fit` when you pass `score_column` or
`value_column`. Model scores without a fit raise.

## Test tuning

Default partition is `validation`. `partition="test"` without
`allow_test_tuning=True` is refused. With the flag, the result carries
a dangerous-opt-in warning. Confirm a frozen plan once with
`session.decision.evaluate(partition="test")`. Evaluating on the same
partition you tuned on also warns.

```python
try:
    session.decision.fit(
        method="threshold",
        partition="test",
        fp_cost=1.0,
        fn_cost=5.0,
    )
except Exception as exc:
    print(type(exc).__name__, exc)
```

## Bundle

`session.decision.save_bundle` writes `buildml.decision_bundle.v1`:
threshold or matrix or allocation rules, plus any auxiliary industry
estimator. Session checkpoints do not embed `DecisionPlan`. Applying
from model scores still needs a compatible `session.fit` unless the
plan carries that auxiliary estimator.

```python
session.decision.save_bundle("artifacts/decision_bundle")
```

## Benchmark

`benchmarks/optimize/policy_value.py` compares a validation-tuned
cost-optimal policy with a fixed 0.5 baseline on held-out expected
cost.
