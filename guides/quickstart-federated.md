# Quickstart: Federated learning (local simulation)

Local FedAvg-style simulation on Session data partitioned by a client/group
column: `fit_federated` runs train-only local updates, aggregates
`coef_` / `intercept_`, then `evaluate_federated` / `predict_federated` on
holdout. Persist via `buildml.federated_bundle.v1`.

**Backends:** `native` (core) or `flower` with `pip install 'buildml[federated-industry]'`.
Both are honest in-process simulations — not production FL networking; **not**
cryptographic secure aggregation.

**Go deeper:** [Federated learning deep](federated-deep.md) ·
[Artifacts](artifacts-checkpoints-bundles.md)

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

fit = session.fit_federated(
    backend="native",
    method="fedavg",
    estimator="sgd_classifier",
    n_rounds=5,
    local_epochs=2,
)
print(fit.backend, fit.n_clients, fit.final_train_metric, len(fit.round_history))

ev = session.evaluate_federated(partition="validation", per_client=True)
print(ev.metrics, ev.n_clients_evaluated)

preds = session.predict_federated(partition="test")
print(len(preds.predictions))

session.save_federated_bundle("artifacts/federated_bundle")
```

| In scope | Out of scope |
| --- | --- |
| Native + optional Flower (`flwr`) backends | Turnkey gRPC / Ray FL deployment |
| Local FedAvg / FedProx on client/group column | Cryptographic secure aggregation |
| sklearn linear/SGD coefficient averaging | Non-linear trees / neural FedAvg |
| Train-only local updates; holdout eval | Claiming production FL from simulation |
| Distinct `buildml.federated_bundle.v1` | Session checkpoint embedding the plan |

Next Phase 2 item: **Knowledge graphs (KG)**.
