# Federated learning (deep)

Practical Session-facing **federated learning simulation** for research,
teaching, and workflows. This is **not** a distributed FL network stack
(Flower/OpenFL) and does **not** implement cryptographic secure aggregation.

## What BuildML means by “federated”

Rows sharing a client/group id act as one simulated client. Each federation
round:

1. Sample a fraction of clients (`client_fraction`).
2. Clone the global `coef_` / `intercept_` to each selected client.
3. Run `local_epochs` of train-only updates on that client's **train** rows.
4. Aggregate with sample-size weights (FedAvg). Optional FedProx proximal pull
   (`method="fedprox"`, `mu > 0`) after each local epoch.

Validation/test partitions are never used for local updates.

## Algorithms (depth over breadth)

| Method | Behavior |
| --- | --- |
| `fedavg` | Weighted-by-n coefficient averaging |
| `fedprox` | FedAvg + proximal pull toward the round's global weights |

Supported estimators (must expose `coef_` / `intercept_`):

- Classification: `sgd_classifier`, `logistic_regression`
- Regression: `sgd_regressor`, `ridge`, `linear_regression`

SGD paths use `partial_fit`; full-fit models use `.fit` (with `warm_start`
when available). Prefer completing this FedAvg path deeply over stubbing a
zoo of FL algorithms.

## Session API

| API | Role |
| --- | --- |
| `fit_federated` | Train-only federated rounds |
| `evaluate_federated` | Global + optional per-client holdout metrics |
| `predict_federated` | Global predictions (no update) |
| `save_federated_bundle` / `load_federated_bundle` | `buildml.federated_bundle.v1` |

Properties: `federated_plan`, `federated_fit_result`,
`federated_eval_result`, `federated_predict_result`.

Client identity: single `role="group"` column, or explicit `client_column=`.
The client column is excluded from features.

## Leakage discipline

- Requires a split; local updates use **train only**.
- Clients see only their own train rows during local updates.
- `evaluate_federated` / `predict_federated` never call local training.
- Class vocabulary for classifiers is discovered from the full **train**
  target column (labels only), disclosed on the plan.

## Privacy honesty

Aggregation is **in-process**. The orchestrator sees client coefficient
updates. Do **not** claim differential privacy, secure multi-party
computation, or cryptographic secure aggregation from this path.

## Bundle boundary

`buildml.federated_bundle.v1` stores `FederatedPlan` (global estimator +
client contract + round history). Session checkpoints store
data/roles/splits/history — they do **not** embed the federated model.
Reload tabular workflow via `checkpoint_load`; reload the learner via
`load_federated_bundle`.

## AI / explain / walkthrough

Teaching-critical tools: `fit_federated`, `evaluate_federated`,
`predict_federated`, plus save/load bundle. Walkthrough exposes
`federated_status`. Explain overlays cover leakage, privacy limits, and
bundle boundaries.

## Explicit non-goals

- No Flower / OpenFL / gRPC client runtime.
- No cryptographic secure aggregation.
- No FedOpt / SCAFFOLD / neural FedAvg zoo (unless later implemented for real).
- No causal APIs.

Next Phase 2 item: **Bayesian / probabilistic ML**.
