"""Statistical evidence must survive CSV and chart consumer boundaries."""
import csv
import io

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.dashboard.exports import export_csv, list_csv_sections


def rows(report, section):
    return list(csv.DictReader(io.StringIO(export_csv(report, section)[1])))


@pytest.mark.parametrize("shape", ["per_column", "mapping", "list"])
@pytest.mark.parametrize("kind", ["numeric", "categorical"])
def test_univariate_supported_shapes_preserve_evidence(shape, kind):
    stats = {"kind": kind, "count": 10, "missing_rate": 0.2}
    if kind == "numeric":
        stats.update(mean=3.5, std=1.2)
    else:
        stats.update(mode="common", unique=2, top_values={"rare": 1, "common": 9})
    if shape == "per_column":
        report = {"univariate": {"per_column": {"feature": stats}}}
    elif shape == "mapping":
        report = {"univariate": {kind: {"feature": stats}}}
    else:
        report = {"univariate": {kind: [{"column": "feature", **stats}]}}
    result = rows(report, "univariate_" + kind)
    assert len(result) == 1
    assert result[0]["column"] == "feature"
    assert float(result[0]["count"]) == 10
    if kind == "numeric":
        assert float(result[0]["mean"]) == 3.5
    elif shape == "per_column":
        assert result[0]["mode"] == result[0]["top_value"] == "common"
        assert int(result[0]["top_count"]) == 9


@pytest.mark.parametrize("shape", ["matrix", "mapping"])
@pytest.mark.parametrize("method,section,key", [("pearson", "correlations", "pearson_r"), ("spearman", "spearman", "spearman_rho")])
def test_correlation_pairs_match_independent_statistics(shape, method, section, key):
    frame = pd.DataFrame({"z": [0, 1, 2, 4], "a": [4, 2, 1, 0], "m": [0, 2, 0, 1]})
    matrix = frame.corr(method=method)
    payload = {"columns": list(frame.columns), "matrix": matrix.values.tolist()} if shape == "matrix" else matrix.to_dict()
    result = rows({"bivariate": {method: payload}}, section)
    assert len(result) == 3
    seen = set()
    for row in result:
        a, b = row["feature_a"], row["feature_b"]
        seen.add(frozenset((a, b)))
        assert float(row[key]) == pytest.approx(matrix.loc[a, b])
    assert len(seen) == 3


def test_real_eda_missingness_survives_csv():
    frame = pd.DataFrame({"x": [1., 2., np.nan, 4., 5., 6.], "group": ["a", "a", "b", "b", "b", None]})
    report = Session.ingest(frame).eda(include_plots=False, show=False).to_dict()
    actual = {row["column"]: row for row in rows(report, "missing_rates")}
    for column in frame:
        assert int(actual[column]["missing_count"]) == frame[column].isna().sum()
        assert float(actual[column]["missing_rate"]) == pytest.approx(frame[column].isna().mean())
    for section in list_csv_sections(report):
        assert int(section["rows"]) == len(rows(report, section["key"]))


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("ratios", [[0.5, 0.3, 0.2], [0.1] * 10])
def test_pca_chart_preserves_variance_and_cumulative_sum(theme, ratios):
    pytest.importorskip("plotly")
    from buildml.dashboard.charts import build_chart_figures
    figures = build_chart_figures({"multivariate": {"pca": {"explained_variance_ratio": ratios}}}, theme=theme)
    figure = figures["pca_variance"]
    np.testing.assert_allclose(figure.data[0].y, ratios)
    np.testing.assert_allclose(figure.data[1].y, np.cumsum(ratios))


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_categorical_association_chart_orders_evidence(theme):
    pytest.importorskip("plotly")
    from buildml.dashboard.charts import build_chart_figures
    report = {"bivariate": {"categorical_pairs": [{"a": "a", "b": "b", "cramers_v": 0.1}, {"a": "c", "b": "d", "cramers_v": 0.8}]}}
    figures = build_chart_figures(report, theme=theme)
    figure = figures["cramers_v_bars"]
    assert list(figure.data[0].x) == [0.8, 0.1]
    assert "not proof" in " ".join(a.text for a in figure.layout.annotations)


def test_real_target_summary_exports_statistics_and_methodological_context():
    target = np.arange(40, dtype=float) ** 2
    frame = pd.DataFrame({"x": np.arange(40, dtype=float), "y": target})
    report = Session.ingest(frame).set_roles({"x": "feature", "y": "target"}).eda(include_plots=False, show=False).to_dict()
    actual = {row["key"]: row["value"] for row in rows(report, "target_summary")}
    assert actual["summary:type"] == "regression_target"
    assert float(actual["summary:mean"]) == pytest.approx(np.mean(target))
    assert float(actual["summary:std"]) == pytest.approx(np.std(target, ddof=1))
    assert actual["interpretation:purpose"] == "exploratory"
    assert "does not establish causation" in actual["interpretation:limitations"]


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_real_mixed_report_chart_missingness_matches_original_rows(theme):
    pytest.importorskip("plotly")
    from buildml.dashboard.charts import build_chart_figures
    frame = pd.DataFrame({
        "x": [float(i) if i % 4 else np.nan for i in range(40)],
        "z": np.sin(np.arange(40)),
        "category": ["a" if i % 3 else "b" for i in range(40)],
        "other": ["c" if i % 2 else "d" for i in range(40)],
        "target": [i % 2 for i in range(40)],
    })
    report = Session.ingest(frame).set_roles({"x": "feature", "z": "feature", "category": "feature", "other": "feature", "target": "target"}).eda(include_plots=False, show=False).to_dict()
    figures = build_chart_figures(report, theme=theme)
    trace = figures["missing_rates"].data[0]
    actual = dict(zip(trace.y, trace.x, strict=True))
    expected = {name: value for name, value in frame.isna().mean().items() if value > 0}
    assert actual == pytest.approx(expected)
    assert len(figures["cramers_v_bars"].data[0].x) > 0
