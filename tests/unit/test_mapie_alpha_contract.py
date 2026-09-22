"""MAPIE coverage must match the intervals or sets actually produced."""

from types import SimpleNamespace

import numpy as np
import pytest

from buildml.core.errors import ValidationError
from buildml.probabilistic.adapters.mapie import (
    MapieWrapper,
    mapie_predict_interval,
    mapie_predict_sets,
)
from buildml.probabilistic.evaluate import evaluate_probabilistic


class ModernRegression:
    def predict_interval(self, x):
        return np.ones(len(x)), np.tile([[0., 2.]], (len(x), 1))


class ModernClassification:
    classes_ = np.array([0, 1])

    def predict_set(self, x):
        return np.zeros(len(x)), np.tile([[True, False]], (len(x), 1))


class Legacy:
    def __init__(self, task):
        self.task = task
        self.received_alpha = None

    def predict(self, x, *, alpha):
        self.received_alpha = alpha
        values = np.tile(np.array([[[0.], [2.]]]), (len(x), 1, 1))
        return np.zeros(len(x)), values if self.task == "regression" else values.astype(bool)


def call_prediction(estimator, task, alpha):
    if task == "regression":
        return mapie_predict_interval(estimator, np.ones((2, 1)), task=task, alpha=alpha)
    return mapie_predict_sets(estimator, np.ones((2, 1)), task=task, alpha=alpha)


@pytest.mark.parametrize("task,model", [
    ("regression", ModernRegression), ("classification", ModernClassification),
])
def test_modern_fitted_alpha_enforced(task, model):
    wrapper = MapieWrapper(task, "split", .1, model(), api="modern")
    assert call_prediction(wrapper, task, .1)
    with pytest.raises(ValidationError, match="differs from calibrated alpha"):
        call_prediction(wrapper, task, .2)


@pytest.mark.parametrize("task,model", [
    ("regression", ModernRegression), ("classification", ModernClassification),
])
def test_raw_modern_cannot_invent_fitted_alpha(task, model):
    with pytest.raises(ValidationError, match="Raw modern MAPIE"):
        call_prediction(model(), task, .2)


@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("wrapped", [True, False])
def test_legacy_prediction_receives_changed_alpha(task, wrapped):
    model = Legacy(task)
    estimator = MapieWrapper(task, "split", .1, model, api="legacy") if wrapped else model
    call_prediction(estimator, task, .2)
    np.testing.assert_array_equal(model.received_alpha, [.2])


@pytest.mark.parametrize("task,model", [
    ("regression", ModernRegression), ("classification", ModernClassification),
])
def test_evaluation_rejects_changed_modern_alpha_before_scoring(task, model):
    # No dataset is needed: invalid coverage must fail before broad scoring
    # exception handling can turn the failure into a partial evaluation result.
    plan = SimpleNamespace(backend="mapie", task=task, alpha=.1, interval_method="mapie",
                           estimator_=MapieWrapper(task, "split", .1, model(), api="modern"))
    with pytest.raises(ValidationError, match="differs from calibrated alpha"):
        evaluate_probabilistic(None, plan, None, partition="all", alpha=.2)
