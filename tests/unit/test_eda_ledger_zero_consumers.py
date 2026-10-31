"""Measured zero mutual information survives real report consumers."""
import copy

import pandas as pd
import pytest
from sklearn.metrics import mutual_info_score

from buildml.dashboard.sheet import build_ledger
from buildml.eda.html_report import _ledger_block, _ledger_groups
from buildml.eda.sheet_coverage import build_ledger_groups


@pytest.mark.parametrize("row", [
    {"feature": "category", "mi": 0.},
    {"column": "category", "score": 0.},
    {"key": "category", "value": 0.},
    {"feature": "category", "mi": 0., "score": 0.9},
    {"feature": "category", "mi": None, "score": 0.},
])
def test_independent_zero_mi_retained_in_shared_static_and_cockpit(row):
    # Cartesian support gives exact empirical independence, not an assumed zero.
    frame = pd.DataFrame([(x, y) for _ in range(12) for x in ["a", "b"] for y in [0, 1]],
                         columns=["category", "target"])
    measured = mutual_info_score(frame["category"], frame["target"])
    assert measured == pytest.approx(0., abs=1e-15)
    report = {"overview": {"n_rows": len(frame), "analysis_rows": len(frame), "n_columns": 2},
              "bivariate": {"mutual_information_vs_target": [row]}}
    original = copy.deepcopy(report)
    for groups in [build_ledger_groups(report), _ledger_groups(report, [])]:
        group = next(group for group in groups if group["key"] == "mi")
        assert group["items"] == [("category", "0")]
        markup = _ledger_block([group])
        assert "not available" not in markup
        assert "category" in markup
        assert 'title="0">0</span>' in markup
    cockpit = next(group for group in build_ledger(report) if group["key"] == "mi")
    assert [item for column in cockpit["cols"] for item in column["items"]] == [{"k": "category", "v": "0"}]
    assert report == original


def test_dict_and_list_metrics_have_identical_values():
    values = {"independent": 0., "weak": 0.125, "strong": 0.75}
    dict_report = {"bivariate": {"mutual_information": values}}
    list_report = {"bivariate": {"mutual_information": [
        {"feature": key, "mi": value} for key, value in values.items()
    ]}}
    def mi_items(report):
        return dict(next(group for group in build_ledger_groups(report) if group["key"] == "mi")["items"])
    assert mi_items(dict_report) == mi_items(list_report)
