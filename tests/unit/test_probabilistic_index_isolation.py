"""Split positions cannot become DataFrame labels during fitting or calibration."""

from __future__ import annotations

from importlib.util import find_spec

import numpy as np
import pandas as pd
import pytest

from buildml import Session


def _frame(task):
    rng = np.random.default_rng(932)
    x = rng.normal(size=(180, 2))
    y = 1.7 * x[:, 0] - 0.3 * x[:, 1] + rng.normal(0, 0.25, len(x))
    if task == "classification":
        y = np.where(y > 0, "accepted", "rejected")
    return pd.DataFrame({"x": x[:, 0], "z": x[:, 1], "y": y})


def _fit(frame, task, conformal, backend="native"):
    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "z": "feature", "y": "target"})
        .split(test_size=0.25, validation_size=0.2, random_state=6)
    )
    estimator = "bayesian_ridge" if task == "regression" else "gaussian_nb"
    kwargs = {}
    if backend == "mapie":
        estimator = "split"
    elif backend == "ngboost":
        estimator = "ngboost_regressor"
        kwargs = {"n_estimators": 5, "learning_rate": 0.1}
    session.probabilistic.fit(
        backend=backend,
        task=task,
        conformal=conformal,
        estimator=estimator,
        random_state=25,
        alpha=0.2,
        **kwargs,
    )
    return session


@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("conformal", [False, True])
@pytest.mark.parametrize("index_kind", ["offset", "permuted", "duplicates"])
def test_index_labels_cannot_change_fit_or_calibration_membership(
    task, conformal, index_kind, backend="native"
):
    frame = _frame(task)
    baseline = _fit(frame, task, conformal, backend)
    renamed = frame.copy()
    if index_kind == "offset":
        renamed.index = np.arange(8000, 8000 + len(frame))
    elif index_kind == "permuted":
        renamed.index = np.random.default_rng(71).permutation(len(frame))
    else:
        renamed.index = np.arange(len(frame)) // 2
    actual = _fit(renamed, task, conformal, backend)
    expected_plan, actual_plan = baseline.probabilistic_plan, actual.probabilistic_plan
    assert actual_plan.conformal_fit_indices_ == expected_plan.conformal_fit_indices_
    assert actual_plan.conformal_calib_indices_ == expected_plan.conformal_calib_indices_
    fit_positions = set(actual_plan.conformal_fit_indices_)
    calibration_positions = set(actual_plan.conformal_calib_indices_)
    assert not fit_positions & calibration_positions
    assert fit_positions | calibration_positions == set(actual.split_plan.train_indices)

    def assert_same(reference, candidate):
        left = reference.probabilistic.predict(partition="test").predictions
        right = candidate.probabilistic.predict(partition="test").predictions
        if task == "regression":
            np.testing.assert_allclose(left, right, rtol=0, atol=1e-12)
        else:
            np.testing.assert_array_equal(left, right)
        q_left = reference.probabilistic_plan.conformal_quantile_
        q_right = candidate.probabilistic_plan.conformal_quantile_
        assert q_right == (pytest.approx(q_left) if q_left is not None else None)
        if task == "regression" and conformal:
            expected = reference.probabilistic.predict_interval(partition="test")
            observed = candidate.probabilistic.predict_interval(partition="test")
            np.testing.assert_allclose(expected.lower, observed.lower, rtol=0, atol=1e-12)
            np.testing.assert_allclose(expected.upper, observed.upper, rtol=0, atol=1e-12)

    assert_same(baseline, actual)
    changed = renamed.copy()
    holdout = [*actual.split_plan.test_indices, *actual.split_plan.validation_indices]
    column = changed.columns.get_loc("y")
    if task == "regression":
        changed.iloc[holdout, column] += 100_000
    else:
        changed.iloc[holdout, column] = np.where(
            changed.iloc[holdout].y == "accepted", "rejected", "accepted"
        )
    perturbed = _fit(changed, task, conformal, backend)
    assert_same(actual, perturbed)


@pytest.mark.parametrize("backend", ["mapie", "ngboost"])
@pytest.mark.parametrize("index_kind", ["offset", "permuted", "duplicates"])
def test_installed_industry_backend_preserves_positional_calibration(backend, index_kind):
    if find_spec(backend) is None:
        pytest.skip(f"{backend} is not installed")
    test_index_labels_cannot_change_fit_or_calibration_membership(
        "regression", True, index_kind, backend=backend
    )
