"""Fitted models and exported plot boards carry finite, actionable evidence."""

import json
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from buildml import Session


@pytest.fixture(scope="module", autouse=True)
def viz():
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")


def _fit_model(train_y, test_y, estimator, task):
    train_x = np.arange(len(train_y), dtype=float)
    test_x = np.arange(len(test_y), dtype=float)
    data = pd.DataFrame({"x": np.r_[train_x, test_x], "y": np.r_[train_y, test_y]})
    split = Session.ingest(data).set_roles({"x": "feature", "y": "target"})
    split.inject_split(
        train_indices=list(range(len(train_y))),
        test_indices=list(range(len(train_y), len(data))),
    )
    return split.fit(estimator, task=task)


@pytest.mark.parametrize(
    "mode", ("constant", "perfect", "constant_prediction", "short", "increasing", "decreasing")
)
def test_regression_board_exports_finite_residual_evidence(tmp_path, mode):
    n = 7 if mode == "short" else 12
    x = np.arange(n, dtype=float)
    prediction = x
    estimator = DecisionTreeRegressor(random_state=9)
    if mode == "constant_prediction":
        prediction = np.zeros(n)
        estimator = DummyRegressor(strategy="constant", constant=0)
    error = {
        "constant": np.where(x % 2 == 0, 2.0, -2.0),
        "perfect": np.zeros(n),
        "constant_prediction": x + 1.0,
        "short": x + 1.0,
        "increasing": x + 1.0,
        "decreasing": n - x,
    }[mode]
    session = _fit_model(x, prediction + error, estimator, "regression")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        board = session.eval_plots(
            include_learning_curve=False,
            include_importance=False,
            export_html=tmp_path / "board.html",
        )
    assert not [item for item in captured if item.category is RuntimeWarning]
    metrics = board.metrics
    assert metrics["rmse"] == pytest.approx(np.sqrt(np.mean(error**2)))
    assert metrics["residual_bias"] == pytest.approx(error.mean())
    if mode in {"increasing", "decreasing"}:
        centered_prediction = prediction - prediction.mean()
        centered_error = np.abs(error) - np.abs(error).mean()
        expected = (centered_prediction @ centered_error) / (
            np.linalg.norm(centered_prediction) * np.linalg.norm(centered_error)
        )
        assert metrics["heteroscedasticity_correlation"] == pytest.approx(expected)
        assert "review-residual-spread" in {item.key for item in board.recommendation_details}
        assert "Inspect how error magnitude varies" in (tmp_path / "board.html").read_text(
            encoding="utf-8"
        )
    else:
        assert metrics.get("heteroscedasticity_correlation") is None
    if abs(error.mean()) > 0.25 * (metrics["rmse"] + 1e-12):
        assert "review-residual-bias" in {item.key for item in board.recommendation_details}
    for recommendation in board.recommendation_details:
        assert set(recommendation.based_on) <= {item.key for item in board.findings}
    assert set(board.figures) == {"residuals_scatter", "residuals_hist", "predicted_vs_actual"}
    assert json.dumps(board.to_dict(), allow_nan=False)


class _BrokenProbabilityTree(DecisionTreeClassifier):
    def __init__(self, failure):
        super().__init__(random_state=11)
        self.failure = failure

    def predict_proba(self, X, check_input=True):
        result = super().predict_proba(X, check_input=check_input)
        if self.failure == "raises":
            raise ValueError("score source unavailable")
        if self.failure == "short_rows":
            return result[:-1]
        if self.failure == "short_columns":
            return result[:, :1]
        result[0, 1] = {"nan": np.nan, "infinity": np.inf, "out_of_range": 1.1}[self.failure]
        return result


@pytest.mark.parametrize(
    "failure", ("raises", "short_rows", "short_columns", "nan", "infinity", "out_of_range")
)
def test_binary_probability_failures_preserve_confusion_board(tmp_path, failure):
    target = np.array([2] * 6 + [10] * 6)
    session = _fit_model(target, target, _BrokenProbabilityTree(failure), "classification")
    board = session.eval_plots(
        include_learning_curve=False, include_importance=False, export_html=tmp_path / "board.html"
    )
    assert set(board.figures) == {"confusion_matrix"}
    assert board.metrics["accuracy_proxy"] == 1.0
    assert board.metrics["positive_class"] == "10"
    assert "roc_auc" not in board.metrics
    skipped = {item["panel"]: item["reason"] for item in board.skipped}
    reason = {
        "raises": "score source unavailable",
        "short_rows": "probability matrix shape",
        "short_columns": "probability matrix shape",
        "nan": "finite values in [0, 1]",
        "infinity": "finite values in [0, 1]",
        "out_of_range": "finite values in [0, 1]",
    }[failure]
    for panel in ("roc_curve", "pr_curve", "calibration", "threshold_tradeoff"):
        assert reason in skipped[panel]
    assert reason in (tmp_path / "board.html").read_text(encoding="utf-8")
    assert json.dumps(board.to_dict(), allow_nan=False)


@pytest.mark.parametrize("classes", (2, 3))
def test_fitted_classifier_exports_only_supported_panels(tmp_path, classes):
    target = np.repeat([2, 10, 22][:classes], 6)
    session = _fit_model(target, target, DecisionTreeClassifier(random_state=11), "classification")
    board = session.eval_plots(
        include_learning_curve=False, include_importance=False, export_html=tmp_path / "board.html"
    )
    assert board.metrics["accuracy_proxy"] == 1.0 and board.metrics["n_labels"] == classes
    assert "confusion_matrix" in board.figures
    if classes == 2:
        assert board.metrics["positive_class"] == "10"
        assert board.metrics["roc_auc"] == board.metrics["average_precision"] == 1.0
        assert {"roc_curve", "pr_curve", "calibration", "threshold_tradeoff"} <= set(board.figures)
    else:
        assert set(board.figures) == {"confusion_matrix"}
        assert "roc_auc" not in board.metrics
        skipped = {item["panel"]: item["reason"] for item in board.skipped}
        assert "multiclass" in skipped["roc_curve"] and "binary" in skipped["threshold_tradeoff"]
    assert json.dumps(board.to_dict(), allow_nan=False)
