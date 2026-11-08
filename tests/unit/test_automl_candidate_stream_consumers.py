"""Real sampler/fit observations verify bounded pending candidates and numerical stability."""

import importlib.util
import inspect
import json
import sys
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, ParameterSampler
from test_automl_budget_input_consumers import assert_preserved, restored, snapshot
from test_automl_deadline_consumers import SECONDS, DelayedCV
from test_automl_refit_consumers import options, population, predict, score

from buildml.core.errors import MissingExtraError
from buildml.preprocess import PreprocessRecipe


@contextmanager
def observed_work():
    """Observe actual constructors/fits without replacing their implementations."""
    sampler_code = inspect.unwrap(ParameterSampler.__init__).__code__
    fit_code = inspect.unwrap(Ridge.fit).__code__
    previous = sys.getprofile()
    events = []

    def observe(frame, event, arg):
        if previous is not None:
            previous(frame, event, arg)
        if event != "call":
            return
        if frame.f_code is sampler_code:
            events.append(("sample",))
        elif frame.f_code is fit_code:
            events.append(("fit", float(frame.f_locals["self"].alpha)))

    sys.setprofile(observe)
    try:
        yield events
    finally:
        sys.setprofile(previous)


def reference_prediction(fitting, scoring, alpha, weighted, scaled):
    fitting = fitting.copy()
    scoring = scoring.copy()
    if scaled:
        mean = fitting[["x", "z"]].mean()
        scale = fitting[["x", "z"]].std(ddof=0)
        fitting[["x", "z"]] = (fitting[["x", "z"]] - mean) / scale
        scoring[["x", "z"]] = (scoring[["x", "z"]] - mean) / scale
    return predict(fitting, scoring, alpha, weighted)


def check_numerical_bundle(s, result, selection, weighted, scaled, tmp_path):
    train, validation, test = s.partition("train"), s.partition("validation"), s.partition("test")
    for trial in result.trials:
        pairs = (
            [(train, validation)]
            if selection == "validation"
            else [
                (train.iloc[a], train.iloc[b])
                for a, b in KFold(2, shuffle=True, random_state=42).split(train)
            ]
        )
        values = [
            score(
                evaluate,
                reference_prediction(fit, evaluate, trial.params["alpha"], weighted, scaled),
                "mse",
                weighted,
            )
            for fit, evaluate in pairs
        ]
        assert trial.mean_score == pytest.approx(np.mean(values), abs=1e-10)
    winner = min(result.trials, key=lambda t: t.mean_score)
    assert result.best_params == winner.params
    expected = reference_prediction(train, test, winner.params["alpha"], weighted, scaled)
    actual = np.asarray(s.predict(partition="test"))
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    saved = s.automl.save_bundle(tmp_path / "streamed")
    loaded = population(weighted)
    loaded.automl.load_bundle(saved, trusted=True)
    np.testing.assert_allclose(loaded.predict(partition="test"), expected, atol=1e-10)
    assert loaded.automl.evaluate().metrics["mse"] == pytest.approx(
        score(test, expected, "mse", weighted)
    )
    json.dumps(result.to_dict(), allow_nan=False)
    return actual


def signature(label, result=None, predictions=None):
    # Paired local receipts compare public choices/scores/predictions across
    # archived and corrected sources; no resource observation enters this record.
    value = {"label": label, "status": "missing" if result is None else "fitted"}
    if result is not None:
        value.update(
            best=result.best_params,
            trials=[{"params": t.params, "score": t.mean_score} for t in result.trials],
            predictions=predictions.tolist(),
        )
    print("CANDIDATE_SIGNATURE " + json.dumps(value, sort_keys=True))


def assert_resource_order(events, method, selection, n_trials):
    samplers = [i for i, event in enumerate(events) if event[0] == "sample"]
    fits = [i for i, event in enumerate(events) if event[0] == "fit"]
    assert len(fits) == n_trials * (2 if selection == "cv" else 1) + 1
    if method == "randomized":
        assert len(samplers) == n_trials
        for left, right in zip(samplers, samplers[1:] + [len(events)]):
            assert any(left < fit < right for fit in fits), (
                "A real fit must consume each sampled candidate before another is prepared."
            )
    else:
        assert samplers == [], "This search method must not prepare an unused random queue."


@pytest.fixture(scope="module")
def prior_bundle(tmp_path_factory):
    s = population(False)
    s.automl.run(**options("validation", "mse"))
    return s.automl.save_bundle(tmp_path_factory.mktemp("stream-prior") / "winner")


def consumer(method, selection, weighted, scaled, tmp_path, prior_bundle, backend="native"):
    opts = options(selection, "mse")
    opts.update(
        backend=backend,
        method=method,
        preprocess=PreprocessRecipe(impute=None, scale="standard" if scaled else None),
    )
    resolved_method = "optuna" if backend == "optuna" else method
    label = f"{backend}/{method}/{selection}/{weighted}/{scaled}"
    if resolved_method == "optuna" and importlib.util.find_spec("optuna") is None:
        s = restored(prior_bundle)
        before = snapshot(s)
        with observed_work() as events:
            with pytest.raises(MissingExtraError):
                s.automl.run(**opts)
        assert_preserved(s, before)
        signature(label)
        assert events == [], "Missing dependencies must refuse before unused sampling or fits."
        return
    s = population(weighted)
    original = s.dataset.to_pandas().copy(deep=True)
    with observed_work() as events:
        result = s.automl.run(**opts)
    assert 0 < len(result.trials) <= 3
    predictions = check_numerical_bundle(s, result, selection, weighted, scaled, tmp_path)
    signature(label, result, predictions)
    pd.testing.assert_frame_equal(s.dataset.to_pandas(), original)
    assert_resource_order(events, resolved_method, selection, len(result.trials))


@pytest.mark.parametrize("method", ["grid", "randomized", "evolutionary", "optuna"])
@pytest.mark.parametrize("selection", ["cv", "validation"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("scaled", [False, True])
def test_real_searches_preserve_numerical_winners_without_pending_random_queues(
    method, selection, weighted, scaled, tmp_path, prior_bundle
):
    consumer(method, selection, weighted, scaled, tmp_path, prior_bundle)


@pytest.mark.parametrize("selection", ["cv", "validation"])
def test_optuna_backend_override_does_not_prepare_a_native_random_queue(
    selection, tmp_path, prior_bundle
):
    consumer("grid", selection, False, False, tmp_path, prior_bundle, backend="optuna")


@pytest.mark.parametrize("weighted", [False, True])
def test_expired_random_search_does_not_sample_the_next_candidate(weighted, tmp_path):
    s = population(weighted)
    cv = DelayedCV()
    opts = options("cv", "mse")
    opts.update(method="randomized", cv=cv)
    with observed_work() as events:
        result = s.automl.run(**opts, time_budget=SECONDS)
    assert len(cv.calls) == len(result.trials) == 1
    predictions = check_numerical_bundle(s, result, "cv", weighted, False, tmp_path)
    signature(f"deadline/randomized/{weighted}", result, predictions)
    assert_resource_order(events, "randomized", "cv", 1)
