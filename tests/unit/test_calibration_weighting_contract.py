"""Calibration reports and figures describe the requested evaluation population."""

import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import LinearSVC

from buildml import Session
from buildml.core.errors import MissingExtraError, ValidationError
from buildml.model.diagnostics import calibration_report
from buildml.model.plot_boards import build_eval_plot_board


def test_tied_score_ece_weights_actual_group_sizes_and_preserves_brier():
    frame = pd.DataFrame({"x": [0] * 140 + [1] * 40 + [2] * 20,
                          "y": ([0] * 110 + [1] * 30 + [0] * 15 + [1] * 25
                                + [0] * 2 + [1] * 18)})
    session = (Session.ingest(frame).set_roles({"x": "feature", "y": "target"})
               .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=3)
               .fit(LogisticRegression(), task="classification"))
    report = session.calibration(partition="validation")
    part = session.partition("validation")
    probabilities = session.fit_result.estimator.predict_proba(part[["x"]])[:, 1]
    actual = part.y.to_numpy()
    expected_error = 0.0
    group_counts = []
    for score in sorted(set(probabilities)):
        mask = probabilities == score
        group_counts.append(int(mask.sum()))
        expected_error += mask.sum() * abs(actual[mask].mean() - score) / len(part)
    assert len(report.payload["calibration_curve"]["fraction_positives"]) == 3
    assert report.payload["bin_counts"] == group_counts
    assert report.payload["ece"] == pytest.approx(expected_error)
    assert report.payload["brier_score"] == pytest.approx(np.square(actual - probabilities).mean())
    assert sum(report.payload["bin_counts"]) == report.payload["n_rows"]
    curve = report.payload["calibration_curve"]
    unweighted = np.abs(np.asarray(curve["fraction_positives"]) - curve["mean_predicted_value"]).mean()
    assert abs(expected_error - unweighted) > 0.001
    assert report.payload["ece_method"] == "sample-weighted mean absolute bin gap"


def _weighted_classifier(classes):
    rows = np.arange(120)
    x = np.tile([-2.0, -1.0, 1.0, 2.0], 30)
    if classes == 2:
        target = ((x > 0) ^ (rows % 7 == 0)).astype(int)
        weights = np.where(target != (x > 0), 20.0, 1.0)
    else:
        target = np.where(rows % 7 == 0, 2, np.searchsorted([-1.5, 1.5], x))
        weights = np.where(target == 2, 20.0, 1.0)
    weights[rows < 80] = 1
    weights[(rows >= 80) & (x == 2)] = 0
    labels = np.asarray(["red", "blue", "gold"])[target]
    frame = pd.DataFrame({"x": x, "y": labels, "importance": weights}, index=rows // 3)
    return (
        Session.ingest(frame)
        .set_roles({"x": "feature", "y": "target", "importance": "weight"})
        .inject_split(train_indices=list(range(80)), validation_indices=[], test_indices=list(range(80, 120)))
        .fit(LogisticRegression(), task="classification")
    )


def test_binary_report_and_export_use_the_same_weighted_bins(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    from matplotlib.figure import Figure

    session = _weighted_classifier(2)
    holdout = session.partition("test")
    fit = session.fit_result
    label = fit.estimator.classes_[1]
    keep = holdout.importance.to_numpy() > 0
    y = (holdout.y.to_numpy() == label)[keep].astype(float)
    p = fit.estimator.predict_proba(holdout[["x"]])[keep, 1]
    w = holdout.importance.to_numpy()[keep]
    # Two empirical quantile bins split at the median of the positive-weight rows.
    median = np.median(p)
    membership = (p <= median, p > median)
    mass = [w[mask].sum() for mask in membership]
    counts = [int(mask.sum()) for mask in membership]
    actual = [np.average(y[mask], weights=w[mask]) for mask in membership]
    predicted = [np.average(p[mask], weights=w[mask]) for mask in membership]
    expected_ece = sum(abs(np.sum(w[mask] * (y[mask] - p[mask]))) for mask in membership) / w.sum()

    saved_curve = []
    savefig = Figure.savefig

    def capture(figure, filename, *args, **kwargs):
        if str(filename).endswith("calibration.png"):
            model = figure.axes[0].lines[1]
            saved_curve.append((model.get_xdata(), model.get_ydata(), model.get_label()))
        return savefig(figure, filename, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    folder = tmp_path / "figures"
    html = tmp_path / "report.html"
    report = calibration_report(
        session.dataset, session.split_plan, fit, partition="test",
        n_bins=2, export_figures=folder, export_html=html,
    )
    payload = report.payload
    assert payload["n_rows"] == len(holdout)
    assert payload["sample_weight_column"] == "importance"
    assert payload["bin_counts"] == counts
    assert payload["bin_weights"] == pytest.approx(mass)
    assert payload["prevalence"] == pytest.approx(np.average(y, weights=w))
    assert payload["brier_score"] == pytest.approx(np.average((y - p) ** 2, weights=w))
    assert payload["ece"] == pytest.approx(expected_ece)
    assert payload["calibration_curve"]["fraction_positives"] == pytest.approx(actual)
    assert payload["calibration_curve"]["mean_predicted_value"] == pytest.approx(predicted)
    assert len(saved_curve) == 1
    curve_x, curve_y, curve_label = saved_curve[0]
    np.testing.assert_allclose(curve_x, predicted)
    np.testing.assert_allclose(curve_y, actual)
    assert f"{expected_ece:.3f}" in curve_label
    assert html.is_file() and (folder / "calibration.png").is_file()
    assert "data:image/png;base64," in html.read_text(encoding="utf-8")

    board = build_eval_plot_board(
        session.dataset, session.split_plan, fit, partition="test", calibration_n_bins=2,
        include_learning_curve=False, include_importance=False,
    )
    assert board.metrics["ece"] == pytest.approx(expected_ece)
    assert board.metrics["calibration_bin_counts"] == counts
    assert board.metrics["calibration_bin_weights"] == pytest.approx(mass)
    assert board.metrics["calibration_sample_weight_column"] == "importance"


def test_multiclass_calibration_uses_declared_weights_in_provider_class_order():
    session = _weighted_classifier(3)
    holdout = session.partition("test")
    estimator = session.fit_result.estimator
    probabilities = estimator.predict_proba(holdout[["x"]])
    expected = {
        label: np.average((holdout.y.eq(label) - probabilities[:, index]) ** 2, weights=holdout.importance)
        for index, label in enumerate(estimator.classes_)
    }
    report = session.calibration(partition="test")
    assert report.payload["sample_weight_column"] == "importance"
    assert {row["class"]: row["brier_score"] for row in report.payload["per_class_brier"]} == pytest.approx(expected)


@pytest.mark.parametrize(("fault", "message"), [("shape", "one probability"), ("nonfinite", "finite probabilities")])
def test_calibration_rejects_malformed_probability_output(fault, message, monkeypatch):
    session = _weighted_classifier(2)
    fitted = session.fit_result.estimator
    predict = fitted.predict_proba

    def invalid(frame):
        probabilities = predict(frame)
        if fault == "shape":
            return probabilities[:, :1]
        probabilities[0, 1] = np.nan
        return probabilities

    monkeypatch.setattr(fitted, "predict_proba", invalid)
    with pytest.raises(ValidationError, match=message):
        session.calibration(partition="test")


@pytest.mark.parametrize(
    ("estimator", "task", "split", "message"),
    [
        (Ridge(), "regression", True, "requires a classification model"),
        (LinearSVC(), "classification", True, "does not support predict_proba"),
        (LogisticRegression(), "classification", False, "Split required"),
    ],
)
def test_calibration_requires_a_compatible_fitted_model_and_split(estimator, task, split, message):
    x = np.arange(30, dtype=float)
    session = Session.ingest(pd.DataFrame({"x": x, "y": x % 2})).set_roles({"x": "feature", "y": "target"})
    session.inject_split(train_indices=list(range(24)), validation_indices=[], test_indices=list(range(24, 30)))
    session.fit(estimator, task=task)
    with pytest.raises(ValidationError, match=message):
        calibration_report(
            session.dataset, session.split_plan if split else None, session.fit_result, partition="test",
        )


def test_weighted_calibration_can_export_html_without_plotting_dependencies(tmp_path, monkeypatch):
    session = _weighted_classifier(2)
    expected = session.calibration(partition="test").payload
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    destination = tmp_path / "calibration.html"
    report = session.calibration(partition="test", export_html=destination)
    assert report.payload == expected
    html = destination.read_text(encoding="utf-8")
    assert "Brier" in html and "Interpretation" in html and "data:image/png;base64," not in html
    with pytest.raises(MissingExtraError, match="viz"):
        session.calibration(partition="test", export_figures=tmp_path / "figures")

