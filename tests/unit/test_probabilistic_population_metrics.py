"""Uncertainty scores must retain the same labeled population and calibrated sets."""
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.probabilistic.evaluate import evaluate_probabilistic


@pytest.mark.parametrize("backend", ["native", "mapie"])
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_null_target_filter_retains_original_intervals_and_prediction_sets(backend, task):
    if backend == "mapie":
        pytest.importorskip("mapie")
    rng = np.random.default_rng(714)
    x = rng.normal(size=(180, 2))
    y = 2 * x[:, 0] + .2 * x[:, 1] + rng.normal(scale=.4, size=180)
    if task == "classification":
        y = np.where(y > 0, "malignant", "benign")
    frame = pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "y": y})
    roles = {"a": "feature", "b": "feature", "y": "target"}
    s = Session.ingest(frame).set_roles(roles).split(test_size=.2, validation_size=.2, random_state=4)
    estimator = "split" if backend == "mapie" else (
        "gaussian_nb" if task == "classification" else "bayesian_ridge")
    s.probabilistic.fit(backend=backend, estimator=estimator, task=task, conformal=True, alpha=.1)
    original = s.probabilistic.predict_interval(partition="test")
    test = s.partition("test")
    keep = np.ones(len(test), dtype=bool)
    keep[::3] = False
    changed = frame.copy()
    changed.loc[test.index[~keep], "y"] = np.nan
    scored = Session.ingest(changed).set_roles(roles)
    result = evaluate_probabilistic(scored.dataset, s.probabilistic_plan, s.split_plan, partition="test")
    assert result.n_rows == int(keep.sum())
    assert any("null targets" in warning for warning in result.warnings)
    observed = test.y.to_numpy()[keep]
    if task == "regression":
        lo, hi = np.asarray(original.lower)[keep], np.asarray(original.upper)[keep]
        assert result.interval_coverage == pytest.approx(np.mean((observed >= lo) & (observed <= hi)))
        assert result.mean_interval_width == pytest.approx(np.mean(hi - lo))
    else:
        sets = [row for row, included in zip(original.prediction_sets, keep, strict=True) if included]
        assert result.metrics["set_coverage"] == pytest.approx(np.mean(
            [label in row for label, row in zip(observed, sets, strict=True)]))
        assert result.metrics["mean_set_size"] == pytest.approx(np.mean([len(row) for row in sets]))
    changed.loc[test.index, "y"] = np.nan
    unlabeled = Session.ingest(changed).set_roles(roles)
    empty = evaluate_probabilistic(unlabeled.dataset, s.probabilistic_plan, s.split_plan, partition="test")
    assert empty.n_rows == 0
    assert empty.metrics == {}
    assert any("No labeled" in warning for warning in empty.warnings)
    plain = s.probabilistic.predict(partition="test", return_std=False, return_proba=False)
    assert len(plain.predictions) == len(test)
    assert plain.std is None and plain.probabilities is None
