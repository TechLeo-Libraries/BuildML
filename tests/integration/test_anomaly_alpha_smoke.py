"""Anomaly alpha-gate smoke: split → fit → score → eval → bundle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session


def test_anomaly_alpha_gate_smoke(tmp_path: Path) -> None:
    rng = np.random.default_rng(17)
    n_normal, n_fraud = 220, 25
    normal = rng.normal([0.0, 0.0], 0.9, size=(n_normal, 2))
    fraud = rng.normal([4.0, 4.0], 0.5, size=(n_fraud, 2))
    frame = pd.DataFrame(np.vstack([normal, fraud]), columns=["x", "y"])
    frame["is_fraud"] = [0] * n_normal + [1] * n_fraud

    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "y": "feature", "is_fraud": "target"})
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0)
        .scale(method="standard")
    )

    fit = session.fit_anomaly(
        method="isolation_forest",
        mode="unsupervised",
        contamination=0.1,
        random_state=0,
    )
    assert fit.method == "isolation_forest"
    assert session.anomaly_plan is not None
    assert session.anomaly_fit_result is not None
    assert fit.threshold_policy == "contamination"

    scored = session.score_anomalies(partition="validation")
    assert scored.partition == "validation"
    assert 0.0 <= scored.alert_rate <= 1.0

    ev = session.evaluate_anomaly(partition="validation", positive_label=1)
    assert ev.partition == "validation"
    assert "average_precision" in ev.labeled_metrics
    assert session.anomaly_eval_result is not None

    before = session.explain("fit_anomaly", moment="before")
    assert before.operation == "fit_anomaly"
    assert before.prerequisite_status.get("split") is True

    bundle = session.save_anomaly_bundle(tmp_path / "anomaly_bundle")
    assert (bundle / "meta.json").is_file()
    assert (bundle / "anomaly_plan.joblib").is_file()

    restored = (
        Session.ingest(session.to_pandas())
        .set_roles({"x": "feature", "y": "feature", "is_fraud": "target"})
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0)
        .scale(method="standard")
    )
    restored.load_anomaly_bundle(bundle, trusted=True)
    again = restored.score_anomalies(partition="validation")
    assert again.flags == scored.flags

    # Novelty path still works on the same recipe
    novelty = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "y": "feature", "is_fraud": "target"})
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0)
        .scale(method="standard")
    )
    nov = novelty.fit_anomaly(
        method="lof",
        mode="novelty",
        normal_label_value=0,
        contamination=0.1,
        n_neighbors=12,
    )
    assert nov.n_fit_rows < nov.n_train_rows
    nov_metrics = novelty.evaluate_anomaly(partition="test")
    assert "precision_at_k" in nov_metrics.labeled_metrics
