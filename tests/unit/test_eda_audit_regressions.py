"""Regression checks for insight correctness and analysis scope."""

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.eda.profile import explore_dataset


def test_continuous_features_remain_in_analysis_and_detect_drift():
    rng = np.random.default_rng(7)
    x = rng.normal(size=100)
    session = Session.ingest(pd.DataFrame({"x": x, "related": 2 * x, "y": 3 * x}))
    session.set_roles({"x": "feature", "related": "feature", "y": "target"})
    report = session.eda()
    assert report.overview["eligible_feature_columns"] == ["x", "related"]
    assert report.bivariate["top_abs_pearson_pairs"][0]["corr"] == pytest.approx(1)
    assert report.target["top_numeric_associations"]
    assert report.bivariate["mutual_information_vs_target"]


def test_explicit_feature_overrides_identifier_heuristic():
    session = Session.ingest(pd.DataFrame({"record_number": range(40), "y": [0, 1] * 20}))
    session.set_roles({"record_number": "feature", "y": "target"})
    report = session.eda()
    assert "record_number" in report.quality["id_like_columns"]
    assert "record_number" in report.overview["eligible_feature_columns"]


def test_small_continuous_target_is_regression():
    session = Session.ingest(pd.DataFrame({"x": range(10), "y": np.arange(10) + .3}))
    session.set_roles({"y": "target"})
    report = session.eda()
    assert report.target["summary"]["type"] == "regression_target"
    assert any(spec["kind"] == "numeric_distribution" and spec.get("column") == "y" for spec in report.adaptive_plan)
    assert not any(spec["kind"] == "target_balance" for spec in report.adaptive_plan)


def test_missing_labels_are_not_an_extra_class():
    session = Session.ingest(pd.DataFrame({"x": range(10), "y": [0] * 4 + [1] * 4 + [None] * 2}))
    session.set_roles({"y": "target"})
    summary = session.eda().target["summary"]
    assert len(summary["class_counts"]) == summary["n_classes"] == 2
    assert summary["imbalance_ratio"] == 1
    assert sorted(summary["class_rates"].values()) == [.5, .5]
    assert summary["missing_target_rows"] == 2


def test_train_scope_excludes_holdout_from_quality_and_target():
    session = Session.ingest(pd.DataFrame({"x": range(40), "y": [0, 1] * 20}))
    session.set_roles({"y": "target"}).split(test_size=.25, random_state=1)
    report = explore_dataset(session.dataset, split_plan=session._split_plan, partition="train")
    assert report.overview["n_rows"] == 30
    assert report.target["n_rows"] == 30
    assert report.overview["analysis_partition"] == "train"
    assert report.drift["train_rows"] == 30
    assert report.drift["test_rows"] == 10
    default = session.eda()
    assert default.overview["n_rows"] == 40
    assert any("held-out rows" in warning for warning in default.warnings)


def test_no_eligible_drift_is_unavailable_not_reassuring():
    session = Session.ingest(pd.DataFrame({"id": range(40), "y": [0, 1] * 20}))
    session.set_roles({"id": "id", "y": "target"}).split(test_size=.25)
    assert session.eda().drift["available"] is False


def test_partition_requires_split():
    session = Session.ingest(pd.DataFrame({"x": [1, 2]}))
    with pytest.raises(ValueError, match="split"):
        explore_dataset(session.dataset, partition="train")


def test_export_discloses_train_analysis_and_heldout_drift(tmp_path):
    from buildml.eda.html_report import export_eda_html

    session = Session.ingest(pd.DataFrame({"x": range(40), "y": [0, 1] * 20}))
    session.set_roles({"y": "target"}).split(test_size=.25)
    report = session.eda(partition="train")
    html = export_eda_html(report.to_dict(), tmp_path / "scope.html")
    rendered = html.read_text(encoding="utf-8")
    assert "Analysis scope: train rows" in rendered
    assert "Drift scope: full train versus test partitions" in rendered


def test_nonfinite_features_and_target_are_disclosed_and_excluded():
    frame = pd.DataFrame({
        "x": [1., 2., np.inf, -np.inf, np.nan, 6., 7., 8., 9., 10., 11., 12.],
        "other": list(range(12)),
        "y": [1.5, 2.5, 3.5, 4.5, 5.5, np.inf, 7.5, 8.5, 9.5, 10.5, 11.5, np.nan],
    })
    session = Session.ingest(frame).set_roles({"x": "feature", "other": "feature", "y": "target"})
    session.inject_split(train_indices=list(range(6)), test_indices=list(range(6, 12)))
    report = session.eda(max_plots=0)
    assert report.quality["missing_cell_count"] == 2
    assert report.quality["nonfinite_cell_count"] == 3
    assert report.quality["nonfinite_by_column"]["x"] == 2
    assert any(finding.key == "quality.nonfinite" for finding in report.findings)
    assert report.univariate["per_column"]["x"]["count"] == 9
    assert report.target["nonfinite_target_rows"] == 1
    assert report.target["missing_target_rows"] == 1
    assert report.target["non_missing_target_rows"] == 10
    assert any("infinite numeric values" in warning for warning in report.warnings)
    pd.testing.assert_frame_equal(session.to_pandas(), frame)


@pytest.mark.parametrize("values", [[1., 2., np.inf, 4.], [np.inf, -np.inf, np.nan, np.inf]])
def test_nonfinite_numeric_target_does_not_crash(values):
    session = Session.ingest(pd.DataFrame({"x": [1., 2., 3., 4.], "y": values})).set_roles({"y": "target"})
    report = session.eda(max_plots=0)
    assert report.quality["nonfinite_cell_count"] > 0
