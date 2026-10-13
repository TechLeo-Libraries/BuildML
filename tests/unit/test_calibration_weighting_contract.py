"""Tied predicted probabilities must not give small bins equal ECE weight."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from buildml import Session


def test_tied_score_ece_weights_actual_group_sizes_and_preserves_brier():
    frame = pd.DataFrame({"x": [0] * 140 + [1] * 40 + [2] * 20,
                          "y": ([0] * 110 + [1] * 30 + [0] * 15 + [1] * 25
                                + [0] * 2 + [1] * 18)})
    session = (Session.ingest(frame).set_roles({"x": "feature", "y": "target"})
               .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=3)
               .fit(LogisticRegression(), task="classification"))
    report = session.calibration(partition="validation")
    part = session.partition("validation")
    probabilities = session.fit_result.estimator.predict_proba(part[["x"]])[:, 1]
    actual = part.y.to_numpy()
    expected_error = 0.0
    group_counts = []
    for score in sorted(set(probabilities)):
        mask = probabilities == score
        group_counts.append(int(mask.sum()))
        expected_error += mask.sum() * abs(actual[mask].mean() - score) / len(part)
    assert len(report.payload["calibration_curve"]["fraction_positives"]) == 3
    assert report.payload["bin_counts"] == group_counts
    assert report.payload["ece"] == pytest.approx(expected_error)
    assert report.payload["brier_score"] == pytest.approx(np.square(actual - probabilities).mean())
    assert sum(report.payload["bin_counts"]) == report.payload["n_rows"]
    curve = report.payload["calibration_curve"]
    unweighted = np.abs(np.asarray(curve["fraction_positives"]) - curve["mean_predicted_value"]).mean()
    assert abs(expected_error - unweighted) > 0.001
    assert report.payload["ece_method"] == "sample-weighted mean absolute bin gap"

