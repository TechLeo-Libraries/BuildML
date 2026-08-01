from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.explain import OPERATION_CATALOG
from buildml.model.evidence import evidence_for_plot_board
from buildml.model.html_diagnostics import export_diagnostics_html


def _fitted_session() -> Session:
    return (
        Session.ingest(
            pd.DataFrame(
                {
                    "x": list(range(40)),
                    "z": [value % 3 for value in range(40)],
                    "y": [0] * 20 + [1] * 20,
                }
            )
        )
        .set_roles({"x": "feature", "z": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale()
        .fit(LogisticRegression(max_iter=500), task="classification")
    )


def test_diagnostic_findings_link_evidence_and_api_actions() -> None:
    report = _fitted_session().calibration()

    assert report.findings
    assert report.recommendation_details
    finding_keys = {finding.key for finding in report.findings}
    for recommendation in report.recommendation_details:
        assert recommendation.based_on
        assert set(recommendation.based_on) <= finding_keys
        assert recommendation.action is not None
        assert recommendation.action.operation in OPERATION_CATALOG
    assert report.to_dict()["findings"][0]["evidence"]


def test_classification_and_regression_html_use_shared_offline_shell(tmp_path: Path) -> None:
    classification = tmp_path / "classification.html"
    regression = tmp_path / "regression.html"
    asset = tmp_path / "plot.png"
    asset.write_bytes(b"\x89PNG\r\n\x1a\nembedded-test")

    class_findings, class_recs, class_limits = evidence_for_plot_board(
        "classification",
        "test",
        {"accuracy_proxy": 0.8, "n_labels": 2},
        [{"panel": "roc_curve", "reason": "estimator lacks predict_proba"}],
    )
    export_diagnostics_html(
        {
            "kind": "eval_plot_board",
            "task": "classification",
            "partition": "test",
            "metrics": {"accuracy": 0.8},
            "findings": [item.to_dict() for item in class_findings],
            "recommendation_details": [item.to_dict() for item in class_recs],
            "limitations": class_limits,
            "methods": ["Confusion matrix on test rows."],
            "skipped": [
                {"panel": "roc_curve", "reason": "estimator lacks predict_proba"}
            ],
            "figure_paths": {"confusion_matrix": str(asset)},
        },
        classification,
        title="Unsafe <classification>",
    )
    reg_findings, reg_recs, reg_limits = evidence_for_plot_board(
        "regression",
        "test",
        {"rmse": 1.2, "residual_bias": 0.1, "heteroscedasticity_correlation": 0.2},
        [],
    )
    export_diagnostics_html(
        {
            "kind": "eval_plot_board",
            "task": "regression",
            "partition": "test",
            "metrics": {"rmse": 1.2, "residual_bias": 0.1},
            "findings": [item.to_dict() for item in reg_findings],
            "recommendation_details": [item.to_dict() for item in reg_recs],
            "limitations": reg_limits,
            "methods": ["Residual analysis."],
            "skipped": [
                {"panel": "roc_curve", "reason": "not applicable to regression"}
            ],
        },
        regression,
    )

    class_html = classification.read_text(encoding="utf-8")
    reg_html = regression.read_text(encoding="utf-8")
    for document in (class_html, reg_html):
        assert 'class="bml-skip-link"' in document
        assert 'name="viewport"' in document
        assert 'name="generator" content="BuildML"' in document
        assert 'class="bml-nav"' in document
        assert 'class="bml-theme"' in document
        assert 'class="bml-reading-frame"' in document
        assert "What was examined" in document
        assert "body.bml-dark" in document
        assert "@media print" in document
        assert "http://" not in document
        assert "https://" not in document
        assert "<link " not in document
        assert "Exact metrics and data" in document
        assert "Skipped and degraded panels" in document
        assert 'id="summary"' in document
        assert 'id="evidence"' in document
        assert 'id="visuals"' in document
    assert "data:image/png;base64," in class_html
    assert "Unsafe &lt;classification&gt;" in class_html
    assert "predict_proba" in class_html
    assert "not applicable to regression" in reg_html
    assert "bml-finding severity-" in class_html


def test_walkthrough_covers_status_history_choices_and_offline_html(
    tmp_path: Path,
) -> None:
    session = _fitted_session()
    path = tmp_path / "walkthrough.html"
    report = session.walkthrough(export_html=path)

    assert report.status_counts["done"] >= 4
    assert report.status_counts["available"] > 0
    assert report.status_counts["blocked"] >= 0
    assert report.status_counts["skipped"] > 0
    assert [row["sequence"] for row in report.timeline] == list(
        range(1, len(report.timeline) + 1)
    )
    assert {row["choice_origin"] for row in report.timeline} >= {
        "automatic",
        "explicit",
    }
    assert any(row["state_changes"] for row in report.timeline)
    assert report.unresolved_risks
    assert report.concept_links
    assert report.next_actions
    assert all(row["api_action"].startswith("Session.explain") for row in report.next_actions)
    assert session.last_walkthrough is report

    document = path.read_text(encoding="utf-8")
    assert "BuildML Session Workflow Walkthrough" in document
    assert "Done, available, blocked, and skipped operations" in document
    assert "Timeline and state transitions" in document
    assert "Explicit, recommended, and automatic choices" in document
    assert 'class="bml-skip-link"' in document
    assert "@media print" in document
    assert "http://" not in document
    assert "https://" not in document
