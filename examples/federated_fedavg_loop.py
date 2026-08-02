"""Federated example: fit_federated → evaluate → predict → bundle.

Honesty: local FedAvg-style simulation on a client/group column — not a
distributed FL platform (Flower/OpenFL) and not cryptographic secure
aggregation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session


def main() -> None:
    rng = np.random.default_rng(11)
    rows: list[dict[str, object]] = []
    for client in range(8):
        shift = rng.normal(0, 0.9, size=2)
        for i in range(45):
            label = i % 2
            center = shift + (1.2 if label else -1.2)
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
        method="fedavg",
        estimator="sgd_classifier",
        n_rounds=6,
        local_epochs=2,
        client_fraction=1.0,
    )
    print(
        f"method={fit.method} estimator={fit.estimator_name} "
        f"n_clients={fit.n_clients} "
        f"final_train_metric={fit.final_train_metric}"
    )
    print(f"rounds={fit.round_history}")

    ev = session.evaluate_federated(partition="validation", per_client=True)
    print(f"global metrics={ev.metrics}")
    print(f"n_clients_evaluated={ev.n_clients_evaluated}")

    preds = session.predict_federated(partition="test")
    print(f"n_predictions={len(preds.predictions)}")

    out = Path("artifacts") / "federated_fedavg_bundle"
    session.save_federated_bundle(out)
    print(f"saved bundle -> {out}")


if __name__ == "__main__":
    main()
