from pathlib import Path

import pandas as pd

from buildml import Session
from buildml.eda import EDAReport
from buildml.explain.catalog import OPERATION_CATALOG


def test_eda_research_grade_report_and_html(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "age": [21, 25, 30, 35, 40, 45, None, 55, 60, 22, 28, 33],
            "income": [40, 55, 60, 80, 50, 70, 65, 90, 95, 42, 48, 58],
            "city": ["a", "b", "a", "b", "a", "b", "a", "a", "b", "a", "b", "a"],
            "const": [1] * 12,
            "id_like": list(range(12)),
            "y": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles(
            {
                "age": "feature",
                "income": "feature",
                "city": "feature",
                "y": "target",
                "id_like": "id",
            }
        )
        .split(test_size=0.25, stratify=True, random_state=0)
    )
    html_path = tmp_path / "eda.html"
    report = session.eda(
        include_plots=False,
        show=False,
        export_html=html_path,
        html_format="research",
    )
    assert isinstance(report, EDAReport)
    assert report.narrative
    assert report.findings
    assert report.recommendation_details
    assert all(finding.evidence for finding in report.findings)
    assert all(
        key in {finding.key for finding in report.findings}
        for recommendation in report.recommendation_details
        for key in recommendation.based_on
    )
    assert report.drift.get("available") is True
    assert "flagged_columns" in report.drift
    assert "mutual_information_vs_target" in report.bivariate
    assert report.quality.get("completeness_score") is not None
    assert report.quality.get("constant_columns")
    assert html_path.exists()
    html = html_path.read_text(encoding="utf-8")
    assert "BuildML EDA" in html
    assert "Findings and next steps" in html
    assert "What was examined" in html
    assert "Observed result" in html
    assert "Why it matters" in html
    assert "What next" in html
    assert "<table" in html
    assert "Filter rows" in html
    assert "Sort by" in html
    assert "bml-theme" in html
    if any(not isinstance(value, dict) for value in report.figures.values()):
        assert "data:image/png;base64," in html
    else:
        assert "Visualization" in html or "visualization" in html
    assert "https://" not in html
    assert "http://" not in html

    eligible = set(report.overview["eligible_feature_columns"])
    assert "y" not in eligible
    assert "id_like" not in eligible
    assert "const" not in eligible
    assert "id_like" in report.overview["explicit_role_exclusions"]
    assert "id_like" not in report.overview["heuristic_id_exclusions"]
    assert report.overview["feature_exclusion_reasons"]["id_like"] == [
        "explicit role: id"
    ]
    assert "y" not in report.bivariate["mutual_information_vs_target"]
    assert "id_like" not in report.bivariate["mutual_information_vs_target"]
    assert "const" not in report.bivariate["mutual_information_vs_target"]
    assert all(row["column"] in eligible for row in report.multivariate["vif"])
    feature_plot_columns = {
        column
        for spec in report.adaptive_plan
        for column in spec.get("columns", [spec.get("column")])
        if column is not None and spec.get("kind") != "target_balance"
    }
    assert "id_like" not in feature_plot_columns
    assert "const" not in feature_plot_columns


def test_regression_eda_retains_target_samples_and_noncausal_language() -> None:
    frame = pd.DataFrame(
        {
            "x": list(range(30)),
            "x2": [value * 2 + (value % 3) for value in range(30)],
            "identifier": [f"id-{value}" for value in range(30)],
            "y": [value * 1.5 for value in range(30)],
        }
    )
    report = (
        Session.ingest(frame)
        .set_roles(
            {
                "x": "feature",
                "x2": "feature",
                "identifier": "id",
                "y": "target",
            }
        )
        .eda(include_plots=False, show=False)
    )
    assert report.target["summary"]["type"] == "regression_target"
    assert report.target["n_rows"] == 30
    assert report.target["non_missing_target_rows"] == 30
    assert "identifier" not in report.bivariate["feature_columns_analyzed"]
    assert "y" not in report.multivariate["feature_columns_analyzed"]
    assert any("not evidence of causality" in finding.detail for finding in report.findings)


def test_empty_eda_html_reports_degraded_analyses(tmp_path: Path) -> None:
    frame = pd.DataFrame({"label": ["a", "b", None]})
    html_path = tmp_path / "empty.html"
    report = Session.ingest(frame).eda(
        export_html=html_path,
        include_plots=False,
        html_format="research",
    )
    text = html_path.read_text(encoding="utf-8")
    assert report.findings
    assert "Skipped and degraded analyses" in text
    assert "No eligible result was produced" in text
    assert "Raw technical appendix" in text


def test_heuristic_identifier_is_excluded_from_all_feature_analyses() -> None:
    n_rows = 60
    frame = pd.DataFrame(
        {
            "record_number": list(range(10_000, 10_000 + n_rows)),
            "signal": [value % 7 for value in range(n_rows)],
            "signal_2": [(value % 5) * 2 for value in range(n_rows)],
            "target": [value % 2 for value in range(n_rows)],
        }
    )
    report = (
        Session.ingest(frame)
        .set_roles({"signal": "feature", "signal_2": "feature", "target": "target"})
        .eda(include_plots=False, show=False)
    )

    assert "record_number" in report.quality["id_like_columns"]
    assert "record_number" not in report.overview["eligible_feature_columns"]
    assert report.overview["heuristic_id_exclusions"] == ["record_number"]
    assert "record_number" not in report.overview["explicit_role_exclusions"]
    assert report.overview["feature_exclusion_reasons"]["record_number"] == [
        "heuristic identifier-like detection"
    ]
    assert "record_number" not in report.bivariate["feature_columns_analyzed"]
    assert "record_number" not in report.bivariate["mutual_information_vs_target"]
    assert all(row["column"] != "record_number" for row in report.multivariate["vif"])
    assert all(
        "record_number"
        not in {
            spec.get("column"),
            spec.get("feature"),
            *(spec.get("columns") or []),
        }
        for spec in report.adaptive_plan
    )


def test_normality_diagnostics_are_primary_and_caveated(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "feature": [float((value % 9) ** 2) for value in range(80)],
            "target": [value % 2 for value in range(80)],
        }
    )
    destination = tmp_path / "normality.html"
    report = (
        Session.ingest(frame)
        .set_roles({"feature": "feature", "target": "target"})
        .eda(
            export_html=destination,
            include_plots=False,
            max_plots=0,
            html_format="research",
        )
    )
    profile = report.univariate["per_column"]["feature"]

    assert profile["normality_method"]
    assert profile["normality_sample_size"] == 80
    assert profile["normality_stat"] is not None
    assert profile["normality_pvalue"] is not None
    assert isinstance(profile["appears_non_normal"], bool)
    assert any("does not prove" in item for item in profile["normality_assumptions"])
    html = destination.read_text(encoding="utf-8")
    assert "normality_method" in html
    assert "normality_sample_size" in html
    assert "appears_non_normal" in html
    assert "non-significance does not prove normality" in html


def test_every_eda_recommendation_action_uses_a_catalog_operation() -> None:
    cases = [
        pd.DataFrame({"x": [1.0, None, 2.0] * 8, "target": [0, 1, 0] * 8}),
        pd.DataFrame({"x": [value % 4 for value in range(24)], "target": [0, 1] * 12}),
    ]
    for frame in cases:
        report = (
            Session.ingest(frame)
            .set_roles({"x": "feature", "target": "target"})
            .eda(include_plots=False, show=False)
        )
        for recommendation in report.recommendation_details:
            if recommendation.action is not None:
                assert recommendation.action.operation in OPERATION_CATALOG
