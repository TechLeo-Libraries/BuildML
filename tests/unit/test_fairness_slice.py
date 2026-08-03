"""Fairness disparity reporting slice tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.fairness.evaluate import evaluate_fairness, validate_positive_label


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


def test_evaluate_fairness_core_metrics_numeric_fixture() -> None:
    # A: pred pos on 2/4; B: pred pos on 1/4 → DP = 0.25; DI = 0.5
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])
    sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    report = evaluate_fairness(y_true, y_pred, sensitive, positive_label=1)
    assert report.n_rows == 8
    assert set(report.groups) == {"A", "B"}
    assert report.selection_rate_by_group["A"] == pytest.approx(0.75)
    assert report.selection_rate_by_group["B"] == pytest.approx(0.25)
    assert report.demographic_parity_difference == pytest.approx(0.5)
    assert report.disparate_impact_ratio == pytest.approx(0.25 / 0.75)
    # A: yt=[0,0,1,1] yp=[0,1,1,1] → TPR=1.0, FPR=0.5
    # B: yt=[0,1,0,1] yp=[0,0,0,1] → TPR=0.5, FPR=0.0
    assert report.equalized_odds_tpr_difference == pytest.approx(0.5)
    assert report.equalized_odds_fpr_difference == pytest.approx(0.5)
    assert any("Observational" in d for d in report.disclosures)
    assert any("positive_label is validated" in d for d in report.disclosures)


def test_evaluate_fairness_string_labels_require_positive_label() -> None:
    y_true = np.array(["denied", "denied", "approved", "approved"] * 2)
    y_pred = np.array(["denied", "approved", "approved", "approved"] * 2)
    sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    with pytest.raises(ValidationError, match="positive_label"):
        evaluate_fairness(y_true, y_pred, sensitive)  # default 1
    report = evaluate_fairness(
        y_true, y_pred, sensitive, positive_label="approved"
    )
    assert report.selection_rate_by_group["A"] == pytest.approx(0.75)
    assert report.selection_rate_by_group["B"] == pytest.approx(0.75)
    assert report.demographic_parity_difference == pytest.approx(0.0)


def test_validate_positive_label_hard_error() -> None:
    with pytest.raises(ValidationError, match="does not appear"):
        validate_positive_label(
            np.array(["yes", "no"]),
            np.array(["yes", "no"]),
            positive_label=1,
        )


def test_session_evaluate_fairness_holdout() -> None:
    session = _session()
    report = session.evaluate_fairness(sensitive_column="group", partition="test")
    assert report.n_rows > 0
    assert session.last_fairness is report
    assert report.demographic_parity_difference == report.demographic_parity_difference
    matrix = Session.fairness_capability_matrix()
    assert matrix["default_backend"] == "native"
    assert matrix["positive_label_validated"] is True
    assert "non_goals" in matrix


def test_walkthrough_lazy_skips_inactive_domains() -> None:
    session = _session()
    report = session.walkthrough(capability_probe="lazy")
    assert report.audit_summary.get("capability_probe") == "lazy"
    # Classical fit is active via history, but unused domains stay idle.
    assert report.rag_status.get("status") == "idle" or report.rag_status.get("probed") is False
    assert report.capability_introspection_status.get("capability_probe") == "lazy"
