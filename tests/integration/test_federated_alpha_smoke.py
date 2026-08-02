"""Integration smoke for federated Session loop + walkthrough + bundle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session


def test_federated_alpha_smoke(tmp_path: Path) -> None:
    rng = np.random.default_rng(19)
    rows: list[dict[str, object]] = []
    for client in range(10):
        shift = rng.normal(0, 1.0, size=2)
        for i in range(40):
            label = i % 2
            center = shift + (1.15 if label else -1.15)
            x = rng.normal(center, 0.4, size=2)
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
        n_rounds=4,
        local_epochs=2,
        client_fraction=0.8,
    )
    assert fit.n_clients >= 2
    assert len(fit.round_history) >= 1

    ev = session.evaluate_federated(partition="validation", per_client=True)
    assert ev.n_rows > 0
    assert "accuracy" in ev.metrics

    preds = session.predict_federated(partition="test")
    assert len(preds.predictions) > 0

    walk = session.walkthrough()
    status = walk.federated_status
    assert status.get("has_federated_plan") is True
    assert status.get("method") == "fedavg"

    bundle = session.save_federated_bundle(tmp_path / "fed_bundle")
    other = Session.ingest(frame).set_roles(
        {
            "x": "feature",
            "y": "feature",
            "label": "target",
            "client_id": "group",
        }
    )
    other._split_plan = session.split_plan
    other._dataset = session.dataset
    other.load_federated_bundle(bundle)
    ev2 = other.evaluate_federated(partition="test", per_client=False)
    assert ev2.method == "fedavg"
