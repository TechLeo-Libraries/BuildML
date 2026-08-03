# Federated learning (deep)

Practical Session-facing **federated learning simulation** for research,
teaching, and workflows. Both backends are honest **local simulations** on
Session data: not turnkey production FL networking, and **no** cryptographic
secure aggregation.

## What BuildML means by “federated”

Rows sharing a client/group id act as one simulated client. Each federation
round:

1. Sample a fraction of clients (`client_fraction`).
2. Clone the global `coef_` / `intercept_` to each selected client.
3. Run `local_epochs` of train-only updates on that client's **train** rows.
4. Aggregate with sample-size weights (FedAvg). Optional FedProx proximal pull
   (`method="fedprox"`, `mu > 0`) after each local epoch.

Validation/test partitions are never used for local updates.

## Backends

| Backend | Extra | Behavior |
| --- | --- | --- |
| `native` | none (core) | In-process weighted `coef_` / `intercept_` averaging |
| `flower` | `buildml[federated-industry]` | Flower NumPyClient wrappers over Session partitions + `flwr` weighted aggregation |

Install industry extra:

```bash
pip install 'buildml[federated-industry]'  # flwr>=1.5
```

Capability matrix:

```python
from buildml.federated import federated_capability_matrix
print(federated_capability_matrix())
```

When `flwr` is installed and `backend=` is omitted, `fit_federated` defaults
to `flower`. Pass `backend="native"` to force the core path.

**Honesty:** `backend="flower"` still runs in-process on Session partitions
unless you deploy a real Flower ServerApp/ClientApp yourself. Do not claim
gRPC networking, Ray production sim, or secure aggregation from Session APIs.

## Algorithms (depth over breadth)

| Method | Behavior |
| --- | --- |
| `fedavg` | Weighted-by-n coefficient averaging |
| `fedprox` | FedAvg + proximal pull toward the round's global weights |

Supported estimators (must expose `coef_` / `intercept_`):

- Classification: `sgd_classifier`, `logistic_regression`
- Regression: `sgd_regressor`, `ridge`, `linear_regression`

SGD paths use `partial_fit`; full-fit models use `.fit` (with `warm_start`
when available).

## Session API

| API | Role |
| --- | --- |
| `fit_federated(backend=...)` | Train-only federated rounds |
| `evaluate_federated(backend=...)` | Global + optional per-client holdout metrics |
| `predict_federated(backend=...)` | Global predictions (no update) |
| `save_federated_bundle` / `load_federated_bundle` | `buildml.federated_bundle.v1` |
| `export_round_history(path)` | JSON export of round metrics/weights (Session facade; module-level `export_round_history(plan, path)` also available) |

Properties: `federated_plan`, `federated_fit_result`,
`federated_eval_result`, `federated_predict_result`.

Client identity: single `role="group"` column, or explicit `client_column=`.
The client column is excluded from features.

Round history includes `client_weights`, `total_weight`, and `weighting`:
`sample_size` for auditability.

## Leakage discipline

- Requires a split; local updates use **train only**.
- Clients see only their own train rows during local updates.
- `evaluate_federated` / `predict_federated` never call local training.
- Class vocabulary for classifiers is discovered from the full **train**
  target column (labels only), disclosed on the plan.

## Privacy honesty

Aggregation is **in-process**. The orchestrator sees client coefficient
updates. Do **not** claim differential privacy, secure multi-party
computation, or cryptographic secure aggregation from either backend.

## Bundle boundary

`buildml.federated_bundle.v1` stores `FederatedPlan` (global estimator +
client contract + round history + `backend`). Session checkpoints store
data/roles/splits/history: they do **not** embed the federated model.
Reload tabular workflow via `checkpoint_load`; reload the learner via
`load_federated_bundle`.

## AI / explain / walkthrough

Teaching-critical tools: `fit_federated`, `evaluate_federated`,
`predict_federated`, plus save/load bundle. Walkthrough exposes
`federated_status` with `backend`. Explain overlays cover leakage, privacy
limits, native vs Flower honesty, and bundle boundaries.

## Benchmark

```bash
python benchmarks/federated/fedavg_convergence.py
```

Writes `benchmarks/federated/results/fedavg_convergence.json` with native and
optional Flower convergence curves.

## Explicit non-goals

- No turnkey Flower gRPC / Ray production deployment from Session alone.
- No cryptographic secure aggregation on any backend.
- No FedOpt / SCAFFOLD / neural FedAvg zoo (not offered as a production path).
- No causal APIs.

See also: [knowledge graphs quickstart](quickstart-kg.md).
