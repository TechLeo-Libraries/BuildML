"""Execute AI proposals against real models and check numerical outcomes."""

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.executor import execute_tool, propose_tool_execution
from buildml.ai.tools import build_default_registry


def _run(session, name, **arguments):
    registry = build_default_registry()
    proposal = propose_tool_execution(name, arguments, registry)
    result = execute_tool(session, proposal, confirmed=True, registry=registry)
    assert result.executed, result.error
    return result.result


def _classification():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(120, 2))
    frame = pd.DataFrame(x, columns=["x", "z"])
    frame["y"] = np.where(x[:, 0] > 0, "positive", "negative")
    return (Session.ingest(frame)
            .set_roles({"x": "feature", "z": "feature", "y": "target"})
            .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=9))


@pytest.mark.parametrize("method", ["fit_voting", "fit_stacking", "fit_blending"])
def test_ai_ensemble_workflows_keep_label_semantics_and_test_boundary(method, tmp_path):
    session = _classification()
    before = session.dataset.frame.copy(deep=True)
    _run(session, method, estimators={"linear": "LogisticRegression", "tree": "DecisionTreeClassifier"},
         task="classification")
    scores = _run(session, "evaluate_ensemble", partition="test")
    assert scores.metrics["accuracy"] >= 0.8
    pd.testing.assert_frame_equal(session.dataset.frame, before)
    bundle = tmp_path / method
    saved = _run(session, "save_ensemble_bundle", path=str(bundle))
    assert saved["path"] == str(bundle)
    assert bundle.is_dir()


@pytest.mark.parametrize("estimator", ["sgd_classifier", "passive_aggressive_classifier"])
def test_ai_online_updates_consume_training_rows_without_touching_holdout(estimator, tmp_path):
    session = _classification()
    before = session.dataset.frame.copy(deep=True)
    _run(session, "fit_online", estimator=estimator, chunk_size=12, n_init=24)
    _run(session, "partial_fit_online", n_rows=12)
    assert session.online_plan.n_seen_rows == 36
    scores = _run(session, "evaluate_online", partition="test")
    assert scores.metrics["accuracy"] >= 0.7
    pd.testing.assert_frame_equal(session.dataset.frame, before)
    _run(session, "save_online_bundle", path=str(tmp_path / estimator))


@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
def test_ai_cbr_predictions_preserve_original_string_classes(metric, tmp_path):
    session = _classification()
    before = session.dataset.frame.copy(deep=True)
    _run(session, "fit_cbr", task="classification", k=3, metric=metric, standardize=True)
    result = _run(session, "predict_cbr", partition="test", k=3, return_traces=True)
    assert set(result.predictions) <= {"positive", "negative"}
    scores = _run(session, "evaluate_cbr", partition="test", k=3)
    assert scores.metrics["accuracy"] >= 0.8
    pd.testing.assert_frame_equal(session.dataset.frame, before)
    _run(session, "save_cbr_bundle", path=str(tmp_path / metric))


@pytest.mark.parametrize("conformal", [False, True])
def test_ai_uncertainty_intervals_are_finite_ordered_and_follow_alpha(conformal, tmp_path):
    rng = np.random.default_rng(41)
    x = rng.normal(size=160)
    frame = pd.DataFrame({"x": x, "y": 2 * x + rng.normal(scale=0.2, size=len(x))})
    session = (Session.ingest(frame).set_roles({"x": "feature", "y": "target"})
               .split(test_size=0.2, validation_size=0.2, random_state=8))
    before = session.dataset.frame.copy(deep=True)
    _run(session, "fit_probabilistic", estimator="bayesian_ridge", task="regression",
         conformal=conformal, alpha=0.1)
    if conformal:
        registry = build_default_registry()
        proposal = propose_tool_execution("predict_interval", {"partition": "test", "alpha": 0.2}, registry)
        refused = execute_tool(session, proposal, confirmed=True, registry=registry)
        assert not refused.executed
        assert "calibrated alpha" in refused.error
    narrow = _run(session, "predict_interval", partition="test", alpha=0.1 if conformal else 0.2)
    wide = _run(session, "predict_interval", partition="test", alpha=0.1 if conformal else 0.05)
    assert np.isfinite(narrow.lower).all() and np.isfinite(wide.upper).all()
    assert np.all(np.asarray(narrow.lower) <= np.asarray(narrow.upper))
    assert np.all(np.asarray(wide.lower) <= np.asarray(narrow.lower))
    assert np.all(np.asarray(wide.upper) >= np.asarray(narrow.upper))
    point = _run(session, "predict_probabilistic", partition="test")
    assert len(point.predictions) == len(session.split_plan.test_indices)
    scores = _run(session, "evaluate_probabilistic", partition="test")
    assert scores.metrics["r2"] > 0.9
    pd.testing.assert_frame_equal(session.dataset.frame, before)
    _run(session, "save_probabilistic_bundle", path=str(tmp_path / str(conformal)))


