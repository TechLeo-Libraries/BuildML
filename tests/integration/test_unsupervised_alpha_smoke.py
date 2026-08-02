"""Unsupervised alpha-gate smoke: scale → PCA → cluster → eval → bundle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session


def test_unsupervised_alpha_gate_smoke(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    a = rng.normal([0.0, 0.0], 0.3, size=(50, 2))
    b = rng.normal([2.5, 2.5], 0.3, size=(50, 2))
    frame = pd.DataFrame(np.vstack([a, b]), columns=["x", "y"])
    frame["segment"] = [0] * 50 + [1] * 50

    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "y": "feature", "segment": "ignore"})
        .split(test_size=0.2, validation_size=0.2, random_state=0)
        .scale(method="standard")
        .reduce_dimensions(method="pca", n_components=2, prefix="pc")
    )

    fit = session.fit_clusters(method="kmeans", n_clusters=2, random_state=0)
    assert fit.used_reduce_components is True
    assert session.cluster_plan is not None
    assert session.cluster_fit_result is not None

    assigned = session.assign_clusters(partition="validation")
    assert assigned.n_rows > 0
    assert set(assigned.labels).issubset({0, 1})

    metrics = session.evaluate_clusters(
        partition="validation",
        external_label_column="segment",
    )
    assert metrics.partition == "validation"
    assert "silhouette" in metrics.metrics
    assert "adjusted_rand_index" in metrics.external_metrics
    assert session.cluster_eval_result is not None

    before = session.explain("fit_clusters", moment="before")
    assert before.operation == "fit_clusters"
    assert before.prerequisite_status.get("split") is True

    bundle = session.save_unsupervised_bundle(tmp_path / "unsup_bundle")
    assert (bundle / "meta.json").is_file()
    assert (bundle / "cluster_plan.joblib").is_file()

    restored = (
        Session.ingest(session.to_pandas())
        .set_roles({"pc_1": "feature", "pc_2": "feature", "segment": "ignore"})
        .split(test_size=0.2, validation_size=0.2, random_state=0)
    )
    restored.load_unsupervised_bundle(bundle)
    again = restored.assign_clusters(partition="validation")
    assert again.labels == assigned.labels
