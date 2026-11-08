"""MAPIE point scoring stays available when uncertainty cannot be calibrated."""

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError


@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("method", ["split", "cv_plus", "jackknife_plus"])
@pytest.mark.parametrize("alpha", [0.01, 0.2])
def test_mapie_points_and_uncertainty_are_scored_separately(task, method, alpha):
    pytest.importorskip("mapie")
    rng = np.random.default_rng(187)
    x = rng.normal(size=(40, 2))
    y = 2 * x[:, 0] + rng.normal(size=len(x)) / 3
    labels = np.where(y > 0, "positive", "negative") if task == "classification" else y
    session = (
        Session.ingest(pd.DataFrame({"x": x[:, 0], "z": x[:, 1], "y": labels}))
        .set_roles({"x": "feature", "z": "feature", "y": "target"})
        .inject_split(train_indices=list(range(32)), test_indices=list(range(32, 40)))
    )
    session.probabilistic.fit(backend="mapie", estimator=method, task=task, alpha=alpha)
    plan = session.probabilistic.plan
    native_points = plan.estimator_.estimator.predict(x[32:])
    prediction = session.probabilistic.predict(partition="test", return_proba=True)
    evaluation = session.probabilistic.evaluate(partition="test")
    assert prediction.n_rows == evaluation.n_rows == 8
    if task == "regression":
        np.testing.assert_allclose(prediction.predictions, native_points)
        assert evaluation.metrics["mae"] == pytest.approx(np.mean(np.abs(y[32:] - native_points)))
        metric = "interval_coverage"
    else:
        points = np.array(["negative", "positive"])[native_points]
        np.testing.assert_array_equal(prediction.predictions, points)
        probabilities = np.asarray(prediction.probabilities)
        assert probabilities.shape == (8, 2)
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(8))
        assert evaluation.metrics["accuracy"] == pytest.approx(np.mean(labels[32:] == points))
        metric = "set_coverage"

    if alpha == 0.01:
        assert metric not in evaluation.metrics
        assert any("metrics skipped: MAPIE" in warning for warning in evaluation.warnings)
        with pytest.raises(ValidationError, match="MAPIE.*samples"):
            session.probabilistic.predict_interval(partition="test")
    else:
        interval = session.probabilistic.predict_interval(partition="test")
        np.testing.assert_array_equal(interval.point, prediction.predictions)
        if task == "regression":
            hits = (y[32:] >= np.asarray(interval.lower)) & (y[32:] <= np.asarray(interval.upper))
        else:
            hits = [
                label in members
                for label, members in zip(labels[32:], interval.prediction_sets, strict=True)
            ]
        assert evaluation.metrics[metric] == pytest.approx(np.mean(hits))
