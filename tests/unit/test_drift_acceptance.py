"""Drift provenance must distinguish tested and skipped columns."""

import numpy as np
import pandas as pd

from buildml import Session
from buildml.eda.analyzers.drift import analyze_drift


def test_drift_reports_only_tested_columns_and_cap_reasons():
    frame = pd.DataFrame({
        "valid": np.arange(20.0),
        "sparse": [1.0] + [np.nan] * 19,
        **{f"category_{i}": ["a", "b"] * 10 for i in range(31)},
    })
    session = Session.ingest(frame).split(test_size=0.5, random_state=0)
    result = analyze_drift(session.dataset, session.split_plan)
    assert "valid" in result["feature_columns_analyzed"]
    assert "sparse" not in result["feature_columns_analyzed"]
    assert "category_30" not in result["feature_columns_analyzed"]
    assert "five finite" in result["skipped_columns"]["sparse"]
    assert "cap" in result["skipped_columns"]["category_30"]
    assert len(result["feature_columns_analyzed"]) == 31
