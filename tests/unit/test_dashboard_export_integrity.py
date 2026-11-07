"""CSV exports must preserve evidence carried by heterogeneous report rows."""

import csv
import io

import pytest

from buildml.dashboard.exports import export_csv


@pytest.mark.parametrize(
    "section,report,expected",
    [
        (
            "drift",
            {"drift": {
                "flagged_columns": ["temperature"],
                "numeric_drift": [{"column": "temperature", "ks_statistic": 0.4}],
                "categorical_drift": [{"column": "site", "js_divergence": 0.2}],
                "scores": {"temperature": 0.7},
            }},
            [
                {"column": "temperature", "flagged": "True"},
                {"column": "temperature", "ks_statistic": "0.4"},
                {"column": "site", "js_divergence": "0.2"},
                {"column": "temperature", "score": "0.7", "flagged": "True"},
            ],
        ),
        (
            "outliers",
            {"outliers": {
                "per_column": {"temperature": {"iqr_outlier_count": 2, "iqr_bounds": [1, 8]}},
                "multivariate": {"method": "isolation_forest", "anomaly_count": 3,
                                 "anomaly_rate": 0.03, "n_rows_scored": 100},
            }},
            [
                {"column": "temperature", "iqr_outlier_count": "2", "iqr_lower": "1",
                 "iqr_upper": "8"},
                {"column": "__multivariate__", "method": "isolation_forest",
                 "anomaly_count": "3", "anomaly_rate": "0.03", "n_rows_scored": "100"},
            ],
        ),
        (
            "vif",
            {"multivariate": {"vif": [
                {"column": "x", "vif": 2.5},
                {"column": "z", "vif": 100.0, "warning": "near collinearity"},
            ]}},
            [{"column": "x", "vif": "2.5"},
             {"column": "z", "vif": "100.0", "warning": "near collinearity"}],
        ),
    ],
)
def test_mixed_evidence_survives_csv_roundtrip(section, report, expected):
    filename, content = export_csv(report, section)
    assert filename == f"buildml_eda_{section}.csv"
    rows = list(csv.DictReader(io.StringIO(content)))
    assert [{key: value for key, value in row.items() if value != ""} for row in rows] == expected


def test_csv_preserves_quoted_column_names_and_text():
    column = 'sensor,"east"\nlevel'
    warning = 'near,"duplicate"\ncheck source'
    _, content = export_csv(
        {"multivariate": {"vif": [{"column": column, "vif": 5, "warning": warning}]}},
        "vif",
    )
    assert list(csv.DictReader(io.StringIO(content))) == [
        {"column": column, "vif": "5", "warning": warning}
    ]


@pytest.mark.parametrize("section", ["drift", "outliers", "unknown"])
def test_missing_evidence_is_not_exported_as_an_empty_success(section):
    with pytest.raises(KeyError):
        export_csv({}, section)
