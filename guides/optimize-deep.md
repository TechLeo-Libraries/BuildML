# Optimisation / decision helpers: deep guide

Session path for turning model scores into **decisions** under costs and
capacity constraints. Industry depth: PuLP/OR-Tools MIP knapsack, CVXPY
LP, XGB/calibrated cost-sensitive thresholds: with honest native fallback.

## What this is / is not

| Is | Is not |
| --- | --- |
| Cost-sensitive binary thresholds (native + XGB/calibrated) | General MIP / digital-twin OR platform |
| Multiclass Bayes actions under a user cost matrix | Causal decision analysis |
| Top-K, knapsack (native DP/greedy or PuLP/OR-Tools MIP) | Production scheduler / fleet OR |
| Continuous LP allocation (scipy linprog or CVXPY) | Replacement for Optuna HPO |
| Persisted `DecisionPlan` (`buildml.decision_bundle.v1`) | PuLP/OR-Tools for arbitrary constraint models |

Dependency policy: core stays light (numpy/scipy via sklearn). Industry solvers
live behind `buildml[optimize-industry]` and become **defaults when installed**.

```python
from buildml.optimize import decision_capability_matrix
decision_capability_matrix()["default_backend_when_installed"]
```

## Backends (`backend=` on `fit_decision_policy`)

| Backend | Methods | Extra | Notes |
| --- | --- | --- | --- |
| `native` | all | core | threshold_report, numpy knapsack, scipy linprog |
| `pulp` | `knapsack` | optimize-industry | Exact 0-1 MIP via PuLP/CBC |
| `ortools` | `knapsack` | optimize-industry | Exact 0-1 MIP via OR-Tools |
| `cvxpy` | `lp_allocate` | optimize-industry | Convex LP (same problem class as linprog) |
| `calibrated` | `threshold` | core | CalibratedClassifierCV + cost sweep |
| `xgb` | `threshold` | optimize-industry | XGB `scale_pos_weight` + validation sweep |

When `backend=None`:
- **F1 threshold** (no costs) → `native`
- **Cost threshold** with costs → `xgb` when installed, else `native`
- **Knapsack** → `pulp` > `ortools` > `native`
- **LP allocate** → `native` (scipy linprog); pass `backend='cvxpy'` explicitly when needed

## Cross-link: `tune_threshold`

Classical `Session.tune_threshold` remains the **diagnostic explorer**
(`DiagnosticReport` threshold sweep). `fit_decision_policy(method="threshold")`
calls the same `threshold_report` engine (or industry scorers), stores a
reusable `DecisionPlan`, and also updates the Session's last diagnostic report
for continuity when using the native/calibrated path.

Prefer:

1. `fit_decision_policy(..., partition="validation", fp_cost=..., fn_cost=...)`
2. `evaluate_decisions(partition="test")` once
3. `save_decision_bundle(...)`

## Methods

### `threshold`

Binary probabilistic classifiers. With `fp_cost`/`fn_cost`, minimizes expected
cost on the tuning partition; otherwise recommends best F1. Formula:

`fp_cost·FP + fn_cost·FN − tp_benefit·TP − tn_benefit·TN`

Industry backends train auxiliary estimators stored on the `DecisionPlan` for
apply: Session.fit remains the primary tabular fit.

### `cost_matrix`

User-supplied square `C[true, action]`. For each row, choose

`argmin_a Σ_y P(y|x) C[y, a]`

using `predict_proba`. The matrix is **not** estimated from test labels.

### `topk`

Select up to `capacity` highest scores (optional `min_score` floor). Scores from
`model_proba` / `model_decision_function` or columns.

### `knapsack`

Maximize value under `budget`. Native: exact integer DP when costs are
near-integral and state is bounded; else density-greedy with disclosure.
Industry: exact 0-1 MIP via PuLP or OR-Tools (`backend='pulp'|'ortools'`).

### `lp_allocate`

Continuous `0 ≤ x_i ≤ lp_max_fraction` budget shares. Native: HiGHS linprog.
Industry: CVXPY convex LP (`backend='cvxpy'`). Fractional by design: not
integer MIP.

## Leakage

| Rule | Behavior |
| --- | --- |
| Default partition | `validation` |
| `partition="test"` | Requires `allow_test_tuning=True` + warning |
| Evaluate | Frozen plan on holdout; warn if eval == fit partition |

## Bundle boundary

`buildml.decision_bundle.v1` stores the `DecisionPlan` (threshold / matrix /
allocation rules + optional auxiliary estimator). Session checkpoints do **not**
embed it. Model-score apply still needs a compatible `Session.fit(...)` unless
the plan carries an auxiliary industry estimator.

## Walkthrough / audit

`walkthrough().decision_status` discloses plan presence, capability matrix, and
boundary text. Audit suggests `fit_decision_policy` when a classification fit
exists without a threshold/decision step.

## Benchmark

`benchmarks/optimize/policy_value.py` compares validation-tuned cost-optimal
policies vs a fixed 0.5 baseline on held-out expected cost.

