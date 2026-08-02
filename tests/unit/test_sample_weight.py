"""ColumnRole.WEIGHT must affect classical fit / evaluate / CV paths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole


def _weighted_regression_frame(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    # True slope near 2 for high-weight rows; noise for low-weight rows.
    y = 2.0 * x + rng.normal(scale=0.05, size=n)
    # Poison a subset of rows that receive near-zero weight.
    poison = np.arange(n) % 5 == 0
    y = y.copy()
    y[poison] = -10.0 * x[poison]
    w = np.where(poison, 0.01, 1.0)
    return pd.DataFrame({"x": x, "y": y, "w": w})


def test_weight_role_excluded_from_auto_features_and_affects_fit() -> None:
    frame = _weighted_regression_frame()
    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "y": "target", "w": "weight"})
        .split(test_size=0.25, random_state=0)
    )
    assert session.dataset.role_columns(ColumnRole.WEIGHT) == ["w"]
    session.fit(LinearRegression())
    assert session.fit_result is not None
    assert session.fit_result.weight_column == "w"
    assert "w" not in session.fit_result.feature_columns
    coef = float(session.fit_result.estimator.coef_[0])
    # Weighted fit should recover the true slope (~2), not the poisoned slope.
    assert coef > 1.5

    unweighted = (
        Session.ingest(frame.drop(columns=["w"]))
        .set_roles({"x": "feature", "y": "target"})
        .split(test_size=0.25, random_state=0)
        .fit(LinearRegression())
    )
    unweighted_coef = float(unweighted.fit_result.estimator.coef_[0])  # type: ignore[union-attr]
    assert abs(coef - 2.0) < abs(unweighted_coef - 2.0)


def test_weight_unsupported_estimator_raises() -> None:
    from sklearn.base import BaseEstimator, RegressorMixin

    class NoWeightEstimator(BaseEstimator, RegressorMixin):
        def fit(self, x, y):  # noqa: ANN001
            self.coef_ = np.asarray([0.0])
            return self

        def predict(self, x):  # noqa: ANN001
            return np.zeros(len(x))

    frame = _weighted_regression_frame(40)
    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "y": "target", "w": "weight"})
        .split(test_size=0.25, random_state=1)
    )
    with pytest.raises(ValidationError, match="does not accept sample_weight"):
        session.fit(NoWeightEstimator())


def test_negative_weights_rejected() -> None:
    frame = _weighted_regression_frame(40)
    frame["w"] = -1.0
    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "y": "target", "w": "weight"})
        .split(test_size=0.25, random_state=2)
    )
    with pytest.raises(ValidationError, match="non-negative"):
        session.fit(LinearRegression())


def test_cv_score_uses_sample_weights() -> None:
    frame = _weighted_regression_frame(100)
    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "y": "target", "w": "weight"})
        .split(test_size=0.2, random_state=3)
    )
    result = session.cv_score(LinearRegression(), task="regression", cv=3)
    assert "r2" in result.mean_metrics
    assert result.mean_metrics["r2"] > 0.5


def test_evaluate_records_weight_usage() -> None:
    rng = np.random.default_rng(4)
    n = 60
    frame = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "y": rng.integers(0, 2, size=n),
            "w": rng.uniform(0.5, 2.0, size=n),
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"x1": "feature", "x2": "feature", "y": "target", "w": "weight"})
        .split(test_size=0.3, stratify=True, random_state=4)
        .fit(LogisticRegression(max_iter=400))
    )
    metrics = session.evaluate(partition="test")
    assert metrics.diagnostics.get("sample_weight_column") == "w"
    assert "f1_weighted" in metrics.metrics
