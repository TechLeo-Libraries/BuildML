"""Actual EDA reports must preserve denominators and teach runnable workflows."""

import contextlib
import io

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.dashboard.gates import build_gates_payload


def _report(frame, feature="x", target="y"):
    return Session.ingest(frame).set_roles({feature: "feature", target: "target"}).eda(
        include_plots=False, show=False
    ).to_dict()


def _rows(report):
    return {row["id"]: row for row in build_gates_payload(report)["rows"]}


@pytest.fixture(scope="module", params=[(60, 20, 40), (40, 40, 40), (5, 15, 100)])
def classification_report(request):
    first, second, missing = request.param
    labels = ["malignant"] * first + ["benign"] * second + [None] * missing
    frame = pd.DataFrame({"x": np.random.default_rng(11).normal(size=len(labels)), "y": labels})
    return frame, _report(frame)


def test_majority_gate_and_calculation_use_observed_target_population(classification_report):
    frame, report = classification_report
    counts = frame.y.dropna().value_counts()
    expected = counts.max() / counts.sum()
    row = _rows(report)["04.2"]
    calc = row["teaching"]["calculation"]
    assert calc["inputs"]["majority_share"] == pytest.approx(expected)
    assert f"{expected:.1%}" in row["evidence"]
    assert "labeled" in row["evidence"].lower()
    assert "held-out" in calc["reading"].lower()


def test_class_share_teaching_is_a_probability_distribution(classification_report):
    frame, report = classification_report
    expected = (frame.y.dropna().value_counts() / frame.y.notna().sum()).to_dict()
    row = _rows(report)["04.3"]
    shares = row["teaching"]["calculation"]["inputs"]["shares"]
    assert shares == pytest.approx(expected)
    assert sum(shares.values()) == pytest.approx(1)
    assert "accuracy is uninformative" not in row["evidence"].lower()


def test_threshold_teaching_does_not_invent_positive_label_or_balanced_prior(classification_report):
    _, report = classification_report
    row = _rows(report)["04.4"]
    teaching = row["teaching"]
    assert "Positive rate" not in row["evidence"]
    assert "positive" in row["evidence"].lower() and "label" in row["evidence"].lower()
    assert "positive_rate" not in teaching["calculation"]["inputs"]
    assert "balanced base rate" not in teaching["why_before_modeling"].lower()
    assert "calibrated" in teaching["why_before_modeling"].lower()
    # For posterior probabilities, equal-cost Bayes actions use 0.5 at any prior.
    posterior = np.array([0.01, 0.2, 0.49, 0.51, 0.8, 0.99])
    cost_predict_zero, cost_predict_one = posterior, 1 - posterior
    np.testing.assert_array_equal(posterior >= 0.5, cost_predict_one <= cost_predict_zero)


@pytest.mark.parametrize("offset", [0.0, 17.5])
def test_actual_regression_report_carries_observed_median(offset):
    x = np.random.default_rng(13).normal(size=80)
    y = 3.7 * x + offset
    y[::5] = np.nan
    report = _report(pd.DataFrame({"x": x, "y": y}))
    expected = np.nanmedian(y)
    assert report["target"]["summary"]["median"] == pytest.approx(expected)
    calc = _rows(report)["04.2"]["teaching"]["calculation"]
    assert calc["inputs"]["median"] == pytest.approx(expected)
    assert "observed" in calc["label"].lower()
    assert "y_train" not in calc["formula"]


def test_legacy_regression_report_does_not_invent_zero_median():
    rows = _rows({"overview": {"n_rows": 100, "analysis_rows": 100},
                  "target": {"column": "y", "summary": {"type": "regression_target", "mean": 7.0}}})
    assert "unavailable" in rows["04.2"]["evidence"].lower()
    assert "unavailable" in rows["04.2"]["teaching"]["calculation"]["result"].lower()


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("unusual_names", [False, True])
@pytest.mark.parametrize("categorical_feature", [False, True])
def test_generated_gate_modeling_example_runs_against_real_session(task, unusual_names, categorical_feature):
    x = np.random.default_rng(17).normal(size=100)
    y = (x > 0).astype(float) if task == "classification" else 2.3 * x + 0.25
    y[::11] = np.nan
    feature, target = ('x" field', "target's label") if unusual_names else ("x", "y")
    values = np.where(x > 0, "high", "low") if categorical_feature else x
    frame = pd.DataFrame({feature: values, target: y})
    rows = _rows(_report(frame, feature, target))
    code = rows["04.2"]["teaching"]["worked_example"]["code"]
    namespace = {"frame": frame.copy(deep=True)}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(code, "<generated EDA gate example>", "exec"), namespace)
    session, evaluation = namespace["session"], namespace["evaluation"]
    assert target not in namespace["features"]
    assert evaluation.n_rows == len(session.partition("test"))
    assert not session.dataset.frame[target].isna().any()
    assert all(np.isfinite(value) for value in evaluation.metrics.values())
