# Federated learning quickstart

```bash
pip install buildml
```

In-process FedAvg / FedProx on a client/group column. You need at least
two eligible clients and one target. Default method is FedAvg with
`sgd_classifier`. `backend=None` picks Flower when
`buildml[federated-industry]` is installed, otherwise native. Flower is
still a local simulation, not a network stack and not secure
aggregation.

[Federated deep](federated-deep.md) ·
[federated-hospital-sim](../proofs/federated-hospital-sim/)

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
# Roundtrip: load_bundle(..., trusted=True) then evaluate again on the same split.
# Holdout metrics include accuracy / f1_macro / balanced_accuracy (+ roc_auc when binary).
```

| In scope | Out of scope |
| --- | --- |
| Native + optional Flower (`flwr`) backends | Turnkey gRPC / Ray FL deployment |
| Local FedAvg / FedProx on client/group column | Cryptographic secure aggregation |
| sklearn linear/SGD coefficient averaging | Non-linear trees / neural FedAvg |
| Train-only local updates; holdout eval | Claiming production FL from simulation |
| Distinct `buildml.federated_bundle.v1` | Session checkpoint embedding the plan |
| Flower disclosed as **local-sim** | Networked Flower ServerApp from Session |

Flower (`backend='flower'`) uses `flwr` NumPyClient-shaped wiring + weighted
aggregation helpers but still runs in-process on Session partitions unless you
operate a real Flower deployment yourself.
