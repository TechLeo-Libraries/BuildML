"""Runnable IsolationForest anomaly loop: fit → score → evaluate → bundle.

Requires a GitHub / editable BuildML 2.x install (core sklearn; no extra).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session


def main() -> None:
    rng = np.random.default_rng(0)
    n_normal, n_fraud = 200, 20
    normal = rng.normal(0.0, 1.0, size=(n_normal, 2))
    fraud = rng.normal(4.0, 0.6, size=(n_fraud, 2))
    frame = pd.DataFrame(np.vstack([normal, fraud]), columns=["x", "y"])
    frame["is_fraud"] = [0] * n_normal + [1] * n_fraud

    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "y": "feature", "is_fraud": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )

    fit = session.fit_anomaly(
        method="isolation_forest",
        mode="unsupervised",
        contamination=0.1,
        random_state=0,
    )
    print(
        "fit:",
        fit.method,
        "threshold=",
        round(fit.threshold, 4),
        "train_alert_rate=",
        round(fit.train_alert_rate, 4),
    )

    scored = session.score_anomalies(partition="test")
    print("score:", scored.n_flagged, "flagged; alert_rate=", round(scored.alert_rate, 4))

    ev = session.evaluate_anomaly(partition="test", positive_label=1)
    print("eval labeled:", {k: round(v, 4) for k, v in ev.labeled_metrics.items()})

    out = Path(".buildml-artifacts") / "anomaly_bundle"
    path = session.save_anomaly_bundle(out)
    print("bundle:", path)


if __name__ == "__main__":
    main()
