"""Exported learning curves retain measured CV scores and the requested fit budget."""

import base64
import inspect
import json
import re
import sys
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.model.diagnostics import learning_curve_report


def _session(task="regression"):
    rng = np.random.default_rng(18)
    x = np.linspace(-3, 3, 100)
    z = rng.normal(size=len(x))
    y = np.sin(2 * x) + x * x + 0.7 * z
    if task == "classification":
        y = np.where(y > np.median(y), 30, 10)
    frame = pd.DataFrame({"x": x, "z": z, "y": y})
    return (
        Session.ingest(frame)
        .set_roles({"y": "target"})
        .split(test_size=0.2, random_state=7, stratify=task == "classification")
    )


def _estimator(task):
    cls = DecisionTreeClassifier if task == "classification" else DecisionTreeRegressor
    return cls(splitter="random", max_depth=2, random_state=None)


def _call(session, estimator, api, **kwargs):
    if api == "session":
        return session.classical.learning_curve(estimator, task="auto", **kwargs)
    return learning_curve_report(
        session.dataset, session.split_plan, estimator, task="auto", **kwargs
    )


def _score(task, truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if task == "regression":
        return 1 - float(
            np.square(truth - prediction).sum() / np.square(truth - truth.mean()).sum()
        )
    total = 0.0
    for label in np.union1d(truth, prediction):
        tp = int(((truth == label) & (prediction == label)).sum())
        fp = int(((truth != label) & (prediction == label)).sum())
        fn = int(((truth == label) & (prediction != label)).sum())
        denominator = 2 * tp + fp + fn
        total += (0 if not denominator else 2 * tp / denominator) * int((truth == label).sum())
    return total / len(truth)


@contextmanager
def _observe(session, task, cv, figures=False):
    """Read real estimator fit returns and real saved figures without replacing either."""
    train = session.dataset.frame.iloc[list(session.split_plan.train_indices)]
    splitter = StratifiedKFold(cv) if task == "classification" else KFold(cv)
    folds = list(splitter.split(train[["x", "z"]], train.y))
    code = inspect.unwrap(type(_estimator(task)).fit).__code__
    records, plots = [], {}
    if figures:
        from matplotlib.figure import Figure

        save_code = Figure.savefig.__code__
    else:
        save_code = None

    def observe(frame, event, arg):
        if event == "return" and frame.f_code is code and arg is not None:
            values = frame.f_locals["X"]
            if not figures:
                records.append({"n": len(values)})
                return
            target = frame.f_locals["y"]
            fitted = frame.f_locals["self"]
            assert set(values.index) <= set(train.index)
            fold = (len(records) // 5) % cv
            valid = train.iloc[folds[fold][1]]
            assert set(values.index).isdisjoint(valid.index)
            records.append(
                {
                    "n": len(values),
                    "train": _score(task, target, fitted.predict(values)),
                    "valid": _score(task, valid.y, fitted.predict(valid[["x", "z"]])),
                }
            )
        if event == "call" and frame.f_code is save_code:
            for axes in frame.f_locals["self"].axes:
                for line in axes.lines:
                    if line.get_label() in {"Train", "CV validation"}:
                        plots[line.get_label()] = (
                            np.asarray(line.get_xdata()).copy(),
                            np.asarray(line.get_ydata()).copy(),
                        )
                plots["bands"] = [
                    collection.get_paths()[0].vertices.copy()
                    for collection in axes.collections
                    if collection.get_paths()
                ]

    prior = sys.getprofile()
    sys.setprofile(observe)
    try:
        yield records, plots
    finally:
        sys.setprofile(prior)


@pytest.mark.parametrize("api", ["direct", "session"])
@pytest.mark.parametrize("value", [True, np.bool_(True), 2.5, "3", np.nan, np.inf, 0, -1, 1, 100.5])
def test_invalid_curve_fold_budget_refuses_before_fit_or_export(api, value, tmp_path):
    session = _session()
    estimator = _estimator("regression")
    history = list(session.history)
    # Observe an ordinary three-fold population; invalid requests must start no fit.
    with _observe(session, "regression", 3) as (records, _):
        with pytest.raises(ValidationError, match="cv"):
            _call(
                session,
                estimator,
                api,
                cv=value,
                export_figures=tmp_path / "figures",
                export_html=tmp_path / "refused.html",
            )
    assert records == []
    assert not hasattr(estimator, "tree_")
    assert session.history == history
    assert not (tmp_path / "figures").exists()
    assert not (tmp_path / "refused.html").exists()


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("api", ["direct", "session"])
@pytest.mark.parametrize("cv", [2, 3, np.int64(4)])
@pytest.mark.parametrize("seed", [91, 107])
def test_exported_stochastic_curve_matches_measured_scores_and_fit_budget(
    task, api, cv, seed, tmp_path, capsys
):
    pytest.importorskip("matplotlib", reason="real learning-curve PNG/HTML consumer requires viz")
    pytest.importorskip("seaborn", reason="existing learning-curve viz preflight requires seaborn")
    import matplotlib.pyplot as plt

    session = _session(task)
    estimator = _estimator(task)
    original = session.dataset.frame.copy(deep=True)
    previous_figures = set(plt.get_fignums())
    try:
        np.random.seed(seed)
        with _observe(session, task, int(cv), figures=True) as (records, plotted):
            report = _call(
                session,
                estimator,
                api,
                cv=cv,
                export_figures=tmp_path / "figures",
                export_html=tmp_path / "curve.html",
            )
        measured = records[: 5 * int(cv)]
        assert len(measured) == 5 * int(cv)
        sizes = [r["n"] for r in measured[:5]]
        np.testing.assert_array_equal(report.payload["train_sizes"], sizes)
        statistics = {}
        for split, payload_name in (("train", "train"), ("valid", "valid")):
            values = np.asarray([r[split] for r in measured]).reshape(int(cv), 5)
            mean, std = values.mean(axis=0), values.std(axis=0)
            np.testing.assert_allclose(
                report.payload[f"{payload_name}_scores_mean"], mean, rtol=1e-11, atol=1e-12
            )
            np.testing.assert_allclose(
                report.payload[f"{payload_name}_scores_std"], std, rtol=1e-11, atol=1e-12
            )
            statistics[split] = (mean, std)
        assert report.payload["final_gap"] == pytest.approx(
            statistics["train"][0][-1] - statistics["valid"][0][-1]
        )
        assert report.payload["valid_score_gain"] == pytest.approx(
            statistics["valid"][0][-1] - statistics["valid"][0][0]
        )
        assert report.payload["cv_folds"] == cv
        assert report.payload["scoring"] == ("f1_weighted" if task == "classification" else "r2")
        assert not hasattr(estimator, "tree_")
        pd.testing.assert_frame_equal(session.dataset.frame, original)
        png = tmp_path / "figures" / "learning_curve.png"
        raw_png = png.read_bytes()
        assert raw_png.startswith(b"\x89PNG\r\n\x1a\n")
        html = (tmp_path / "curve.html").read_text(encoding="utf-8")
        images = re.findall(r'data:image/png;base64,([^"\s]+)', html)
        assert any(base64.b64decode(encoded) == raw_png for encoded in images)
        report.show()
        output = capsys.readouterr().out
        assert "Diagnostic" in output and "learning_curve" in output
        assert "Figures:" in output and "HTML:" in output
        if api == "session":
            assert session.history[-1]["operation_id"] == "learning_curve"
        # Emit measured evidence before the export consistency/resource checks.
        print(
            "CURVE_NUMERICAL",
            json.dumps(
                {"task": task, "api": api, "cv": int(cv), "seed": seed, "payload": report.payload}
            ),
        )
        for label, split in (("Train", "train"), ("CV validation", "valid")):
            np.testing.assert_array_equal(plotted[label][0], sizes)
            np.testing.assert_allclose(
                plotted[label][1], statistics[split][0], rtol=1e-11, atol=1e-12
            )
        assert len(plotted["bands"]) == 2
        for vertices, split in zip(plotted["bands"], ("train", "valid"), strict=True):
            mean, std = statistics[split]
            for i, size in enumerate(sizes):
                ys = vertices[vertices[:, 0] == size, 1]
                np.testing.assert_allclose(
                    [ys.min(), ys.max()],
                    [mean[i] - std[i], mean[i] + std[i]],
                    rtol=1e-11,
                    atol=1e-12,
                )
        assert len(records) == 5 * int(cv), (
            "Export must reuse measured scores instead of fitting again"
        )
    finally:
        for number in set(plt.get_fignums()) - previous_figures:
            plt.close(number)
