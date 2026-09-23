"""Passive-aggressive aliases retain their PA-I update across sklearn versions."""

import warnings

import numpy as np
import pytest

from buildml.online.adapters.sklearn import build_sklearn_estimator


@pytest.mark.parametrize("task", ["classifier", "regressor"])
def test_passive_aggressive_first_update(task: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        estimator = build_sklearn_estimator(f"passive_aggressive_{task}", 0)
        if task == "classifier":
            estimator.partial_fit([[1.0, 2.0]], [1], classes=[0, 1])
            step = 1.0 / 5.0
        else:
            estimator.partial_fit([[1.0, 2.0]], [2.0])
            step = (2.0 - 0.1) / 5.0
    # PA-I uses min(C, hinge loss / squared feature norm), C=1.
    np.testing.assert_allclose(np.ravel(estimator.coef_), [step, 2 * step])
    np.testing.assert_allclose(estimator.intercept_, [step])
