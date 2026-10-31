"""Evaluation figures retain fitted class meaning and numerical calibration."""

from importlib.util import find_spec
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier

from buildml import Session
from buildml.core.errors import ValidationError

pytestmark = pytest.mark.skipif(
    find_spec("matplotlib") is None or find_spec("seaborn") is None,
    reason="Plot-board consumers require matplotlib and seaborn (buildml[viz]).",
)

LABELS = [(2, 10), (0, 1), (False, True), (" no ", "yes")]


def _session(labels=(2, 10), learner="tree", population="mixed"):
    # Four genuinely fitted leaves with tied, unequal positive probabilities.
    train_x, train_y = [], []
    for group, positives in enumerate([1, 3, 7, 9]):
        train_x.extend([group] * 10)
        train_y.extend([labels[0]] * (10 - positives) + [labels[1]] * positives)
    test_x = np.repeat([0, 1, 2, 3], [3, 7, 11, 19])
    positive = np.asarray([i % 5 < 2 for i in range(40)])
    if population != "mixed":
        positive[:] = population == "positive"
    test_y = [labels[int(flag)] for flag in positive]
    frame = pd.DataFrame({"x": train_x + test_x.tolist(), "y": train_y + test_y})
    frame.index = np.arange(len(frame)) % 11
    session = Session.ingest(frame).set_roles({"x": "feature", "y": "target"})
    session.inject_split(train_indices=list(range(40)), validation_indices=[40, 41],
                         test_indices=list(range(42, 80)))
    estimator = (DecisionTreeClassifier(max_depth=2, random_state=37) if learner == "tree"
                 else Perceptron(max_iter=400, tol=None, random_state=37))
    return session.fit(estimator, task="classification")


def _replay(session, tmp_path):
    path = session.save_pipeline(tmp_path / "pipeline", evaluate_partition=None)
    restored = Session(dataset=session.dataset, split_plan=session.split_plan)
    return restored.load_pipeline(path, trusted=True)


def _calibration_oracle(truth, scores):
    bins = min(10, max(3, len(truth) // 5))
    edges = np.quantile(scores, np.linspace(0, 1, bins + 1))
    groups = {}
    for outcome, score in zip(truth, scores):
        # A score on an edge belongs to the bin to its left.
        key = sum(score > edge for edge in edges[1:-1])
        groups.setdefault(key, []).append((outcome, score))
    counts, actual, predicted = [], [], []
    error = 0.
    for key in sorted(groups):
        rows = groups[key]
        observed = sum(row[0] for row in rows) / len(rows)
        mean = sum(row[1] for row in rows) / len(rows)
        error += len(rows) * abs(observed - mean)
        counts.append(len(rows))
        actual.append(observed)
        predicted.append(mean)
    return error / len(truth), counts, actual, predicted


@pytest.mark.parametrize("labels", LABELS)
@pytest.mark.parametrize("learner", ["tree", "perceptron"])
@pytest.mark.parametrize("population", ["mixed", "positive", "negative"])
@pytest.mark.parametrize("replay", [False, True])
def test_board_class_meaning_and_calibration(labels, learner, population, replay, tmp_path):
    session = _session(labels, learner, population)
    if replay:
        session = _replay(session, tmp_path)
    model = session.fit_result.estimator
    before = joblib.hash(session.fit_result)
    part = session.partition("test")
    truth, predicted = part["y"].to_numpy(), model.predict(part[["x"]])
    classes = list(model.classes_)
    positive = classes[1]
    expected = np.zeros((2, 2), dtype=int)
    for actual, guess in zip(truth, predicted):
        expected[int(actual == positive), int(guess == positive)] += 1
    board = session.eval_plots(include_learning_curve=False, include_importance=False)
    assert board.metrics["false_negatives"] == expected[1, 0]
    assert board.metrics["false_positives"] == expected[0, 1]
    axis = board.figures["confusion_matrix"].axes[0]
    assert [tick.get_text() for tick in axis.get_xticklabels()] == list(map(str, classes))
    np.testing.assert_array_equal(np.asarray(axis.collections[0].get_array()).reshape(2, 2), expected)
    assert board.metrics["accuracy_proxy"] == pytest.approx(np.trace(expected) / len(truth))
    skipped = {item["panel"]: item["reason"] for item in board.skipped}
    if learner == "tree":
        assert board.metrics["positive_class"] == str(positive)
        scores = model.predict_proba(part[["x"]])[:, 1]
        outcomes = np.asarray(truth == positive, dtype=int)
        ece, counts, actual, means = _calibration_oracle(outcomes, scores)
        assert board.metrics["ece"] == pytest.approx(ece, abs=1e-12)
        assert board.metrics["calibration_bin_counts"] == counts
        assert board.metrics["ece_method"] == "sample-weighted mean absolute bin gap"
        calibration = session.calibration(partition="test")
        assert calibration.payload["ece"] == pytest.approx(ece, abs=1e-12)
        line = board.figures["calibration"].axes[0].lines[1]
        np.testing.assert_allclose(line.get_xdata(), means)
        np.testing.assert_allclose(line.get_ydata(), actual)
        if population == "mixed":
            pos, neg = scores[outcomes == 1], scores[outcomes == 0]
            auc = sum(float(a > b) + .5 * float(a == b) for a in pos for b in neg) / (len(pos) * len(neg))
            assert board.metrics["roc_auc"] == pytest.approx(auc)
        else:
            assert "roc_auc" not in board.metrics and "average_precision" not in board.metrics
            assert "both fitted classes" in skipped["roc_curve"]
            assert "both fitted classes" in skipped["pr_curve"]
    else:
        assert "predict_proba" in skipped["roc_curve"]
        assert "ece" not in board.metrics
    assert joblib.hash(session.fit_result) == before


@pytest.mark.parametrize("invalid", [None, 99, "10"])
@pytest.mark.parametrize("replay", [False, True])
def test_invalid_board_targets_preserve_previous_report(invalid, replay, tmp_path):
    session = _session()
    if replay:
        session = _replay(session, tmp_path)
    previous = session.eval_plots(include_learning_curve=False, include_importance=False)
    frame = session.to_pandas().copy()
    frame["y"] = frame["y"].astype(object)
    frame.iloc[session.split_plan.test_indices[0], 1] = invalid
    broken = Session.ingest(frame).set_roles(session.dataset.roles)
    broken._split_plan = session.split_plan
    broken._fit_result = session.fit_result
    broken._last_plot_board = previous
    before = joblib.hash(broken.history)
    with pytest.raises(ValidationError, match="missing class labels|absent from the fitted classifier"):
        broken.eval_plots(include_learning_curve=False, include_importance=False,
                          export_figures=tmp_path / "bad-figures", export_html=tmp_path / "bad.html")
    assert broken.last_plot_board is previous
    assert joblib.hash(broken.history) == before
    assert not (tmp_path / "bad-figures").exists() and not (tmp_path / "bad.html").exists()


def test_numeric_board_exports_real_figures_and_matching_metrics(tmp_path):
    session = _session()
    before = joblib.hash(session.fit_result)
    result = session.evaluate(include_plots=True, export_figures=tmp_path / "figures",
                              export_html=tmp_path / "report.html")
    board = session.last_plot_board
    assert len(result.diagnostics["plot_board"]["figure_paths"]) >= 5
    for path in board.figure_paths.values():
        assert Path(path).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "data:image/png;base64," in html and "Exact metrics and data" in html
    assert "<td>positive_class</td><td>10</td>" in html
    assert joblib.hash(session.fit_result) == before
