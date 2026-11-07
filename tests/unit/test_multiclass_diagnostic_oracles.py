"""Diagnostic class summaries must match independent probability and error counts."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.core.errors import ValidationError


def _session(labels):
    rng = np.random.default_rng(810)
    points = rng.normal(size=(180, 2))
    codes = np.where(points[:, 0] < -0.6, 0, np.where(points[:, 0] > 0.6, 2, 1))
    frame = pd.DataFrame({"x": points[:, 0], "z": points[:, 1],
                          "y": np.asarray(labels)[codes], "site": ["east", "west", "south"] * 60})
    return (Session.ingest(frame)
            .set_roles({"x": "feature", "z": "feature", "y": "target", "site": "ignore"})
            .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=8)
            .fit(LogisticRegression(max_iter=500), task="classification"))


@pytest.mark.parametrize("labels", [["low", "middle", "high"], [2, 5, 9]])
def test_per_class_brier_matches_fitted_probability_column(labels):
    session = _session(labels)
    part = session.partition("validation")
    model = session.fit_result.estimator
    probabilities = model.predict_proba(part[["x", "z"]])
    report = session.calibration(partition="validation")
    assert report.payload["n_classes"] == 3
    rows = report.payload["per_class_brier"]
    assert len(rows) == 3
    for index, label in enumerate(model.classes_):
        row = next(item for item in rows if item["class"] == str(label))
        truth = (part.y.to_numpy() == label).astype(float)
        expected = np.square(truth - probabilities[:, index]).mean()
        assert row["brier_score"] == pytest.approx(expected)
    with pytest.raises(ValidationError, match="binary"):
        session.tune_threshold(partition="validation")


@pytest.mark.parametrize("labels", [["low", "middle", "high"], [2, 5, 9]])
def test_multiclass_segment_counts_match_original_classes_and_rows(labels, tmp_path):
    session = _session(labels)
    part = session.partition("test")
    predictions = session.fit_result.estimator.predict(part[["x", "z"]])
    report = session.error_slices(by="site", partition="test", min_segment_n=1,
                                  export_html=tmp_path / "segments.html")
    assert report.payload["binary_positive_class"] is None
    assert sum(row["n"] for row in report.payload["segments"]) == len(part)
    for row in report.payload["segments"]:
        site = row["segment"].split("=", 1)[1]
        mask = (part.site == site).to_numpy()
        correct = int((predictions[mask] == part.y.to_numpy()[mask]).sum())
        assert row["n_correct"] == correct
        assert row["n_incorrect"] == int(mask.sum()) - correct
        assert row["accuracy"] == pytest.approx(correct / mask.sum())
        assert "precision" not in row
    assert (tmp_path / "segments.html").exists()
