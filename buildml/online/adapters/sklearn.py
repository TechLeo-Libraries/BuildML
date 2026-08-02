"""Sklearn partial_fit online estimator adapters."""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import (
    PassiveAggressiveClassifier,
    PassiveAggressiveRegressor,
    Perceptron,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

from buildml.core.errors import ValidationError
from buildml.online.types import OnlineTask

_CLASSIFIERS = {
    "sgd_classifier",
    "passive_aggressive_classifier",
    "perceptron",
    "multinomial_nb",
    "bernoulli_nb",
}
_REGRESSORS = {
    "sgd_regressor",
    "passive_aggressive_regressor",
}


def resolve_sklearn_task(estimator: str, task: OnlineTask | None) -> OnlineTask:
    if estimator in _CLASSIFIERS:
        if task == "regression":
            raise ValidationError(
                f"Estimator {estimator!r} is a classifier; task cannot be 'regression'."
            )
        return "classification"
    if estimator in _REGRESSORS:
        if task == "classification":
            raise ValidationError(
                f"Estimator {estimator!r} is a regressor; task cannot be "
                "'classification'."
            )
        return "regression"
    raise ValidationError(
        f"Unknown sklearn online estimator={estimator!r}. "
        f"Supported: {sorted(_CLASSIFIERS | _REGRESSORS)}"
    )


def build_sklearn_estimator(name: str, random_state: int | None) -> Any:
    if name == "sgd_classifier":
        return SGDClassifier(loss="log_loss", random_state=random_state)
    if name == "sgd_regressor":
        return SGDRegressor(random_state=random_state)
    if name == "passive_aggressive_classifier":
        return PassiveAggressiveClassifier(random_state=random_state)
    if name == "passive_aggressive_regressor":
        return PassiveAggressiveRegressor(random_state=random_state)
    if name == "perceptron":
        return Perceptron(random_state=random_state)
    if name == "multinomial_nb":
        return MultinomialNB()
    if name == "bernoulli_nb":
        return BernoulliNB()
    raise ValidationError(f"Unsupported sklearn online estimator '{name}'")
