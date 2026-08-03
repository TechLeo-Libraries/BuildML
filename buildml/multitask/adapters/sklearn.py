"""Sklearn MultiOutput / Chain multi-task adapters."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multioutput import (
    ClassifierChain,
    MultiOutputClassifier,
    MultiOutputRegressor,
    RegressorChain,
)

from buildml.core.errors import ValidationError

_CLS_BASES = {
    "logistic_regression": lambda rs: LogisticRegression(
        max_iter=500, random_state=rs
    ),
    "hist_gradient_boosting": lambda rs: HistGradientBoostingClassifier(
        random_state=rs
    ),
}
_REG_BASES = {
    "ridge": lambda rs: Ridge(random_state=rs),
    "hist_gradient_boosting_regressor": lambda rs: HistGradientBoostingRegressor(
        random_state=rs
    ),
}


def build_sklearn_estimator(
    *,
    method: str,
    task: str,
    base_estimator: str,
    random_state: int | None,
    order_indices: list[int] | None = None,
) -> Any:
    """Build a sklearn MultiOutput or Chain estimator for multi-task learning.

    Selects ``MultiOutputClassifier``/``Regressor`` or chain variants based on
    ``method`` and wraps the requested ``base_estimator`` factory.

    Parameters
    ----------
    method:
        ``multi_output``, ``classifier_chain``, or ``regressor_chain``.
    task:
        ``classification`` or ``regression`` (not ``mixed``).
    base_estimator:
        Base learner name (e.g. ``logistic_regression``, ``ridge``).
    random_state:
        Seed for base estimators and chain shuffling.
    order_indices:
        Optional target order indices for chain methods.

    Returns
    -------
    estimator
        Sklearn multi-output or chain wrapper ready for fitting.

    Raises
    ------
    ValidationError
        When ``method``, ``task``, or ``base_estimator`` pairing is invalid.
    """
    base_key = str(base_estimator).lower().replace("-", "_")
    if task == "classification":
        if base_key not in _CLS_BASES:
            if base_key in _REG_BASES:
                raise ValidationError(
                    f"base_estimator={base_estimator!r} is a regressor; "
                    "classification multi-task needs "
                    f"{sorted(_CLS_BASES)}."
                )
            raise ValidationError(
                f"Unknown classification base_estimator={base_estimator!r}. "
                f"Supported: {sorted(_CLS_BASES)}"
            )
        base = _CLS_BASES[base_key](random_state)
        if method == "multi_output":
            return MultiOutputClassifier(base)
        if method == "classifier_chain":
            return ClassifierChain(base, order=order_indices, random_state=random_state)
        raise ValidationError(
            f"method={method!r} is not valid for sklearn classification multi-task."
        )

    if task == "regression":
        if base_key not in _REG_BASES:
            if base_key in _CLS_BASES:
                raise ValidationError(
                    f"base_estimator={base_estimator!r} is a classifier; "
                    f"regression multi-task needs {sorted(_REG_BASES)}."
                )
            raise ValidationError(
                f"Unknown regression base_estimator={base_estimator!r}. "
                f"Supported: {sorted(_REG_BASES)}"
            )
        base = _REG_BASES[base_key](random_state)
        if method == "multi_output":
            return MultiOutputRegressor(base)
        if method == "regressor_chain":
            return RegressorChain(base, order=order_indices, random_state=random_state)
        raise ValidationError(
            f"method={method!r} is not valid for sklearn regression multi-task."
        )

    raise ValidationError(
        f"Sklearn backend requires task='classification' or 'regression' "
        f"(got {task!r}). Mixed targets need backend='torch'."
    )
