"""Mirror of guides/quickstart-federated.md — local FedAvg, not a network stack."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session


def main() -> None:
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
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
    print(
        f"backend={fit.backend} method={fit.method} estimator={fit.estimator_name} "
        f"n_clients={fit.n_clients} "
        f"final_train_metric={fit.final_train_metric}"
    )
    print(f"rounds={fit.round_history}")

    ev = session.federated.evaluate(partition="validation", per_client=True)
    print(f"global metrics={ev.metrics}")
    print(f"n_clients_evaluated={ev.n_clients_evaluated}")

    preds = session.federated.predict(partition="test")
    print(f"n_predictions={len(preds.predictions)}")

    out = Path("artifacts") / "federated_fedavg_bundle"
    session.federated.save_bundle(out)
    print(f"saved bundle -> {out}")


if __name__ == "__main__":
    main()
