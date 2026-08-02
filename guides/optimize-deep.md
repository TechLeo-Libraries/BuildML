# Optimisation / decision helpers — deep guide

Session path for turning model scores into **decisions** under costs and
capacity constraints. Phase-1 bar: Session API, leakage discipline, tests,
explain/catalog, guides, bundle, AI allowlist, honest docs.

## What this is / is not

| Is | Is not |
| --- | --- |
| Cost-sensitive binary thresholds | General MIP / PuLP / OR-Tools suite |
| Multiclass Bayes actions under a user cost matrix | Causal decision analysis / digital twin |
| Top-K, knapsack-lite, continuous LP allocation | Production scheduler / fleet OR |
| Persisted `DecisionPlan` (`buildml.decision_bundle.v1`) | Replacement for Optuna HPO |

Dependency policy: core stays light. Knapsack uses numpy DP/greedy.
`lp_allocate` uses `scipy.optimize.linprog` (transitive via scikit-learn).
No `buildml[optimize]` extra — PuLP was not justified at this depth.

## Cross-link: `tune_threshold`

Classical `Session.tune_threshold` remains the **diagnostic explorer**
(`DiagnosticReport` threshold sweep). `fit_decision_policy(method="threshold")`
calls the same `threshold_report` engine, stores a reusable `DecisionPlan`,
and also updates the Session's last diagnostic report for continuity.

Prefer:

1. `fit_decision_policy(..., partition="validation", fp_cost=..., fn_cost=...)`
2. `evaluate_decisions(partition="test")` once
3. `save_decision_bundle(...)`

## Methods

### `threshold`

Binary probabilistic classifiers. With `fp_cost`/`fn_cost`, minimizes expected
cost on the tuning partition; otherwise recommends best F1. Formula:

`fp_cost·FP + fn_cost·FN − tp_benefit·TP − tn_benefit·TN`

### `cost_matrix`

User-supplied square `C[true, action]`. For each row, choose

`argmin_a Σ_y P(y|x) C[y, a]`

using `predict_proba`. The matrix is **not** estimated from test labels.

### `topk`

Select up to `capacity` highest scores (optional `min_score` floor). Scores from
`model_proba` / `model_decision_function` or columns.

### `knapsack`

Maximize value under `budget`. Exact integer DP when costs are near-integral
and the state space is bounded; otherwise density-greedy with disclosure.
`knapsack_solver='greedy'` forces the approximation.

### `lp_allocate`

Continuous `0 ≤ x_i ≤ lp_max_fraction` budget shares via HiGHS linprog.
Fractional by design — not integer MIP.

## Leakage

| Rule | Behavior |
| --- | --- |
| Default partition | `validation` |
| `partition="test"` | Requires `allow_test_tuning=True` + warning |
| Evaluate | Frozen plan on holdout; warn if eval == fit partition |

## Bundle boundary

`buildml.decision_bundle.v1` stores the `DecisionPlan` (threshold / matrix /
allocation rules). Session checkpoints do **not** embed it. Model-score apply
still needs a compatible `Session.fit(...)`.

## Walkthrough / audit

`walkthrough().decision_status` discloses plan presence and boundary text.
Audit suggests `fit_decision_policy` when a classification fit exists without
a threshold/decision step.

## Tracker

Recommenders / LTR / KG: **PASS**. This module: Phase-1 bar.
Phase-3 synthetic-data systems: **PASS** (see [synthetic-deep.md](synthetic-deep.md)).
