# Federated learning

```bash
pip install buildml
# Flower NumPyClient + aggregation: pip install "buildml[federated-industry]"
```

Rows that share a client or group id act as one simulated client. Each
round samples clients, copies the global linear weights, trains on that
client's **train** rows, then averages with sample-size weights.

Default method is `fedavg`, estimator `sgd_classifier`, `n_rounds=5`,
`local_epochs=1`, `client_fraction=1.0`. `fedavg` is valid on both
backends, so `backend=None` picks Flower when `flwr` imports cleanly,
otherwise native. That is different from domains whose default method
locks you to sklearn. Flower here is still an in-process simulation on
Session partitions. It is not a gRPC network, not Ray production FL,
and not cryptographic secure aggregation. Pass `backend="native"` if
you want the core path regardless of extras.

The API refuses a fit without a split, local updates on holdout,
missing client identity, and more than one target. You decide the
client column, how many rounds, `client_fraction`, and (for FedProx)
`mu`.

Short on-ramp: [federated quickstart](quickstart-federated.md).
Proof: [federated-hospital-sim](../proofs/federated-hospital-sim/).

## A local FedAvg loop

Need a `role="group"` column or `client_column=`, exactly one target,
and at least two clients with `min_client_rows` train rows (default 2).
The client column is excluded from features.

```python
import numpy as np
import pandas as pd

from buildml import Session

rng = np.random.default_rng(0)
rows = []
for client in range(8):
    shift = rng.normal(0, 0.8, size=2)
    for i in range(40):
        label = i % 2
        center = shift + (1.1 if label else -1.1)
        x = rng.normal(center, 0.35, size=2)
        rows.append(
            {
                "x": float(x[0]),
                "y": float(x[1]),
                "label": int(label),
                "client_id": f"c{client}",
            }
        )
frame = pd.DataFrame(rows)

session = (
    Session.ingest(frame)
    .set_roles(
        {
            "x": "feature",
            "y": "feature",
            "label": "target",
            "client_id": "group",
        }
    )
    .split(test_size=0.2, validation_size=0.2, random_state=0)
    .scale(method="standard")
)

fit = session.federated.fit(
    backend="native",
    method="fedavg",
    estimator="sgd_classifier",
    n_rounds=5,
    local_epochs=2,
)
print(fit.backend, fit.n_clients, fit.final_train_metric, len(fit.round_history))

ev = session.federated.evaluate(partition="validation", per_client=True)
print(ev.metrics, ev.n_clients_evaluated)

preds = session.federated.predict(partition="test")
print(len(preds.predictions))

session.federated.save_bundle("artifacts/federated_bundle")
session.federated.export_round_history("artifacts/federated_rounds.json")
```

`evaluate` defaults to validation with `per_client=True`. `predict`
defaults to test. Neither runs local training.

Round history records `client_weights`, `total_weight`, and
`weighting: sample_size` so you can audit the average.

## What a round does

1. Sample a fraction of eligible clients (`client_fraction`).
2. Clone the global `coef_` / `intercept_` onto each selected client.
3. Run `local_epochs` of updates on that client's train rows.
4. Aggregate with sample-size weights (FedAvg). FedProx adds a proximal
   pull toward the round's global weights after each local epoch.

SGD estimators use `partial_fit`. Full-fit models use `.fit` (with
`warm_start` when the estimator has it).

## Backends

| Backend | Extra | Aggregation |
| --- | --- | --- |
| `native` | none | In-process weighted `coef_` / `intercept_` averaging |
| `flower` | `buildml[federated-industry]` | Flower `NumPyClient` wrappers + `flwr` weighted ndarray aggregation, still in-process |

Flower `available` requires `flwr` to import, not just sit on disk.
A broken install reports unavailable.

```python
session.federated.capability_matrix()
```

## Methods and estimators

| Method | Behavior |
| --- | --- |
| `fedavg` | Weighted-by-n coefficient averaging (default) |
| `fedprox` | FedAvg plus proximal pull. Requires `mu > 0` |

Estimators must expose `coef_` / `intercept_`:

- Classification: `sgd_classifier`, `logistic_regression`
- Regression: `sgd_regressor`, `ridge`, `linear_regression`

No tree FedAvg, no neural FedAvg zoo, no FedOpt / SCAFFOLD on this
surface.

Class vocabulary for classifiers is discovered from the full **train**
target column (labels only) and stored on the plan.

## Privacy

Aggregation is in-process. The orchestrator sees client coefficient
updates. Neither backend gives differential privacy, secure
multi-party computation, or cryptographic secure aggregation. If you
deploy a real Flower ServerApp/ClientApp yourself, that is your
deployment, not `session.federated.fit`.

## Bundles

`buildml.federated_bundle.v1` stores the `FederatedPlan`: global
estimator, client contract, round history, backend. A Session
checkpoint stores data, roles, splits, and history. It does not embed
the federated model. `export_round_history` writes JSON (optional
`include_disclosures=True`). `trusted=True` only for a file you made.

[Artifacts](artifacts-checkpoints-bundles.md)

## When it refuses

| What you see | What happened |
| --- | --- |
| Fit before a split | Local updates are train only |
| No group / `client_column` | Client identity is required |
| Zero or several targets | Federated simulation is single-target (`session.multitask.fit` is the other path) |
| `method="fedprox"` with `mu=0` | Set `mu > 0` |
| `MissingExtraError` for flower | `backend="flower"` without a working `flwr` |
| Too few eligible clients | Raise `min_client_rows` clients, or lower the threshold |

[Federated quickstart](quickstart-federated.md)
