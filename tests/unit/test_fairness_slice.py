"""Fairness disparity reporting slice tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.fairness.evaluate import evaluate_fairness


def _session() -> Session:
    rng = np.random.default_rng(0)
    n = 240
    group = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
    x = rng.normal(size=n)
    # Group B has a shifted decision boundary → measurable gap.
    logits = x + np.where(group == "B", -0.8, 0.0)
    y = (logits > 0).astype(int)
    frame = pd.DataFrame({"x": x, "group": group, "y": y})
    return (
        Session.ingest(frame)
        .set_roles({"x": "feature", "group": "ignore", "y": "target"})
        .split(test_size=0.25, validation_size=0.25, stratify=True, random_state=0)
        .fit(LogisticRegression(max_iter=500), task="classification")
    )


def test_evaluate_fairness_core_metrics() -> None:
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])
    sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    report = evaluate_fairness(y_true, y_pred, sensitive, positive_label=1)
    assert report.n_rows == 8
    assert set(report.groups) == {"A", "B"}
    assert "selection_rate_by_group" in report.to_dict()
    assert report.demographic_parity_difference == report.demographic_parity_difference
    assert any("Observational" in d for d in report.disclosures)


def test_session_evaluate_fairness_holdout() -> None:
    session = _session()
    report = session.evaluate_fairness(sensitive_column="group", partition="test")
    assert report.n_rows > 0
    assert session.last_fairness is report
    matrix = Session.fairness_capability_matrix()
    assert matrix["default_backend"] == "native"
    assert "non_goals" in matrix


def test_walkthrough_lazy_skips_inactive_domains() -> None:
    session = _session()
    report = session.walkthrough(capability_probe="lazy")
    assert report.audit_summary.get("capability_probe") == "lazy"
    # Classical fit is active via history, but unused domains stay idle.
    assert report.rag_status.get("status") == "idle" or report.rag_status.get("probed") is False
    assert report.capability_introspection_status.get("capability_probe") == "lazy"
