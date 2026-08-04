"""Fairness disparity reporting slice tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.fairness.evaluate import evaluate_fairness, validate_positive_label
from buildml.fairness.groups import compose_group_keys
from buildml.fairness.mitigation import (
    apply_group_thresholds,
    suggest_group_thresholds,
    suggest_reweighing_weights,
)
from buildml.fairness.stability import estimate_gap_stability


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


def _intersectional_session() -> Session:
    rng = np.random.default_rng(1)
    n = 320
    group = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
    region = np.array(["urban", "rural"] * (n // 2))
    x = rng.normal(size=n)
    logits = x + np.where(group == "B", -0.7, 0.0) + np.where(region == "rural", -0.3, 0.0)
    y = (logits > 0).astype(int)
    frame = pd.DataFrame({"x": x, "group": group, "region": region, "y": y})
    return (
        Session.ingest(frame)
        .set_roles(
            {
                "x": "feature",
                "group": "ignore",
                "region": "ignore",
                "y": "target",
            }
        )
        .split(test_size=0.25, validation_size=0.2, stratify=True, random_state=1)
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
    assert report.classical_metrics_by_group["A"]["accuracy"] == pytest.approx(0.75)
    assert "accuracy" in report.to_dict()["classical_metrics_by_group"]["A"]
    md = report.to_markdown()
    assert "Demographic parity" in md
    assert report.scope["legal_audit"] is False


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


def test_intersectional_group_keys_and_report() -> None:
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    g = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    r = np.array(["u", "u", "r", "r", "u", "u", "r", "r"])
    keys = compose_group_keys(g, r)
    assert keys[0] == "A|u"
    report = evaluate_fairness(
        y_true,
        y_pred,
        np.column_stack([g, r]),
        positive_label=1,
        sensitive_column=["group", "region"],
    )
    assert report.intersectional is True
    assert report.sensitive_columns == ("group", "region")
    assert "A|u" in report.groups
    assert any("Intersectional" in d for d in report.disclosures)


def test_stability_bootstrap_bands() -> None:
    rng = np.random.default_rng(0)
    n = 200
    sens = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
    y_true = rng.integers(0, 2, size=n)
    y_pred = y_true.copy()
    y_pred[:20] = 1 - y_pred[:20]
    stab = estimate_gap_stability(
        y_true,
        y_pred,
        sens,
        positive_label=1,
        n_resamples=40,
        method="bootstrap",
        random_state=0,
    )
    assert stab.method == "bootstrap"
    band = stab.metrics["demographic_parity_difference"]
    assert band["ci_low"] is not None
    assert band["ci_high"] is not None
    assert band["ci_low"] <= band["ci_high"]

    report = evaluate_fairness(
        y_true,
        y_pred,
        sens,
        positive_label=1,
        bootstrap_samples=30,
        random_state=1,
    )
    assert report.stability is not None
    assert "demographic_parity_difference" in report.stability.metrics
    assert "Stability bands" in report.to_markdown()


def test_stability_stratified_subsample() -> None:
    y_true = np.array([0, 1] * 40)
    y_pred = np.array([0, 1, 1, 0] * 20)
    sens = np.array(["A", "B"] * 40)
    stab = estimate_gap_stability(
        y_true,
        y_pred,
        sens,
        n_resamples=25,
        method="stratified_subsample",
        subsample_fraction=0.75,
        random_state=2,
    )
    assert stab.subsample_fraction == pytest.approx(0.75)
    assert stab.metrics["demographic_parity_difference"]["n_finite_draws"] > 0


def test_classical_bridge_with_scores() -> None:
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])
    scores = np.array([0.1, 0.6, 0.9, 0.8, 0.2, 0.4, 0.3, 0.7])
    sens = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    report = evaluate_fairness(
        y_true, y_pred, sens, positive_label=1, y_score=scores
    )
    assert report.classical_metrics_by_group["A"]["roc_auc"] is not None
    assert report.scope["scores_used_for_auc"] is True


def test_mitigation_threshold_equalization() -> None:
    y_true = np.array([0, 0, 1, 1, 0, 0, 1, 1] * 4)
    # Group A scores higher on average
    scores = np.array([0.8, 0.7, 0.9, 0.85, 0.2, 0.3, 0.4, 0.35] * 4)
    sens = np.array(["A", "A", "A", "A", "B", "B", "B", "B"] * 4)
    suggestion = suggest_group_thresholds(
        y_true, scores, sens, positive_label=1, target="demographic_parity"
    )
    assert set(suggestion.thresholds_by_group) == {"A", "B"}
    assert any("not legal certification" in d for d in suggestion.disclosures)
    y_hat = apply_group_thresholds(
        scores, sens, suggestion.thresholds_by_group, positive_label=1, negative_label=0
    )
    assert len(y_hat) == len(scores)


def test_mitigation_reweighing_weights() -> None:
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1] * 5)
    sens = np.array(["A", "A", "A", "A", "B", "B", "B", "B"] * 5)
    suggestion = suggest_reweighing_weights(y_true, sens, positive_label=1)
    assert len(suggestion.weights) == len(y_true)
    assert pytest.approx(float(suggestion.weights.mean()), abs=1e-9) == 1.0
    assert "A" in suggestion.weight_table
    assert any("does not auto-fit" in d for d in suggestion.disclosures)


def test_session_evaluate_fairness_holdout() -> None:
    session = _session()
    report = session.fairness.evaluate(sensitive_column="group", partition="test")
    assert report.n_rows > 0
    assert session.fairness.last_report is report
    assert session.last_fairness is report
    assert report.demographic_parity_difference == report.demographic_parity_difference
    assert report.classical_metrics_by_group
    matrix = session.fairness.capability_matrix()
    assert matrix["default_backend"] == "native"
    assert matrix["positive_label_validated"] is True
    assert matrix["depth"] == "high"
    assert matrix["supports_intersectional"] is True
    assert "non_goals" in matrix
    assert "suggest_thresholds" in str(matrix["session_paths"])


def test_session_intersectional_and_stability_facade() -> None:
    session = _intersectional_session()
    report = session.fairness.evaluate(
        sensitive_column=["group", "region"],
        partition="test",
        bootstrap_samples=20,
        random_state=0,
    )
    assert report.intersectional is True
    assert "|" in report.groups[0]
    assert report.stability is not None
    assert report.stability.n_resamples == 20


def test_session_attach_to_last_eval() -> None:
    session = _session()
    _ = session.evaluate(partition="validation")
    assert session._last_evaluate_partition == "validation"
    report = session.fairness.attach_to_last_eval(sensitive_column="group")
    assert report.partition == "validation"
    assert session.fairness.last_report is report


def test_session_mitigation_facades() -> None:
    session = _session()
    thr = session.fairness.suggest_thresholds(
        sensitive_column="group",
        partition="validation",
        target="demographic_parity",
    )
    assert thr.thresholds_by_group
    assert any("Opt-in" in d for d in thr.disclosures)
    weights = session.fairness.suggest_reweighing(
        sensitive_column="group", partition="train"
    )
    assert len(weights.weights) > 0
    assert session._fairness_mitigation_suggestion is weights


def test_walkthrough_lazy_skips_inactive_domains() -> None:
    session = _session()
    report = session.walkthrough(capability_probe="lazy")
    assert report.audit_summary.get("capability_probe") == "lazy"
    # Classical fit is active via history, but unused domains stay idle.
    assert report.rag_status.get("status") == "idle" or report.rag_status.get("probed") is False
    assert report.capability_introspection_status.get("capability_probe") == "lazy"
