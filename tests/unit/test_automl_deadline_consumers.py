"""Real CV consumers verify AutoML deadline stops and retained numerical evidence."""

import copy
import json
import time

import joblib
import numpy as np
import pytest
from sklearn.model_selection import KFold
from test_automl_budget_input_consumers import assert_preserved, restored, snapshot
from test_automl_refit_consumers import cv_scores, options, population, predict, ridge_fits, score

from buildml.automl import AutoMLBudget
from buildml.core.errors import ValidationError

SECONDS = 5.0
METHODS = ["grid", "randomized", "evolutionary"]


class DelayedCV:
    """A caller splitter with a real blocking wait on one requested invocation."""

    def __init__(self, delay_on=1):
        self.delay_on = delay_on
        self.calls = []

    def get_n_splits(self, X=None, y=None, groups=None):
        return 2

    def split(self, X, y=None, groups=None):
        self.calls.append(time.monotonic())
        if len(self.calls) == self.delay_on:
            time.sleep(SECONDS + 0.1)
        yield from KFold(2, shuffle=True, random_state=42).split(X, y)


@pytest.fixture(scope="module")
def prior_bundle(tmp_path_factory):
    s = population(False)
    s.automl.run(**options("validation", "mse"))
    return s.automl.save_bundle(tmp_path_factory.mktemp("deadline-prior") / "winner")


@pytest.mark.parametrize("name", ["time_budget", "max_time_seconds"])
@pytest.mark.parametrize(
    "bad",
    [np.nan, np.inf, -np.inf, 0.0, -1.0, True, np.bool_(False), "1", 1 + 2j, object(), 10**400],
    ids=[
        "nan",
        "infinite",
        "negative-infinite",
        "zero",
        "negative",
        "bool",
        "numpy-bool",
        "text",
        "complex",
        "object",
        "overflow",
    ],
)
def test_invalid_effective_durations_refuse_and_preserve_loaded_evidence(name, bad, prior_bundle):
    s = restored(prior_bundle)
    budget = AutoMLBudget(max_trials=3)
    kwargs = {}
    if name == "time_budget":
        kwargs[name] = bad
    else:
        budget.max_time_seconds = bad
    fingerprint = joblib.hash(budget)
    before = snapshot(s)
    with pytest.raises(ValidationError, match=name):
        s.automl.run(**options("cv", "mse"), budget=budget, **kwargs)
    assert joblib.hash(budget) == fingerprint
    assert_preserved(s, before)


@pytest.mark.parametrize("name", ["time_budget", "max_time_seconds"])
@pytest.mark.parametrize("selection", ["cv", "validation", "nested"])
@pytest.mark.parametrize("method", METHODS)
def test_tiny_positive_deadline_starts_no_fit_and_preserves_previous_winner(
    name, selection, method, prior_bundle
):
    s = restored(prior_bundle)
    budget = AutoMLBudget(max_trials=3)
    kwargs = {}
    if name == "time_budget":
        kwargs[name] = 1e-12
    else:
        budget.max_time_seconds = 1e-12
    opts = options(selection, "mse")
    opts["method"] = method
    before = snapshot(s)
    with ridge_fits() as fits:
        with pytest.raises(ValidationError, match="budget|scored"):
            s.automl.run(**opts, budget=budget, **kwargs)
    assert fits == []
    assert_preserved(s, before)


def check_scores(s, result, selection, weighted):
    train = s.partition("train")
    validation = s.partition("validation")
    for trial in result.trials:
        alpha = trial.params["alpha"]
        expected = (
            [score(validation, predict(train, validation, alpha, weighted), "mse", weighted)]
            if selection == "validation"
            else cv_scores(train, alpha, "mse", weighted)
        )
        assert trial.mean_score == pytest.approx(np.mean(expected), abs=1e-10)
    best = min(result.trials, key=lambda t: t.mean_score)
    assert result.best_params == best.params
    expected = predict(train, s.partition("test"), best.params["alpha"], weighted)
    np.testing.assert_allclose(s.predict(partition="test"), expected, atol=1e-10)
    return expected


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("selection", ["cv", "validation"])
@pytest.mark.parametrize("weighted", [False, True])
def test_generous_deadlines_preserve_raw_scores_winners_and_loaded_bundles(
    method, selection, weighted, tmp_path
):
    s = population(weighted)
    budget = AutoMLBudget(max_trials=3, max_time_seconds=np.float32(600))
    fingerprint = joblib.hash(budget)
    opts = options(selection, "mse")
    opts["method"] = method
    result = s.automl.run(**opts, budget=budget)
    assert 0 < len(result.trials) <= 3
    assert joblib.hash(budget) == fingerprint
    assert type(result.config["budget"]["max_time_seconds"]) is float
    expected = check_scores(s, result, selection, weighted)
    path = s.automl.save_bundle(tmp_path / "timed-winner")
    loaded = population(weighted)
    loaded.automl.load_bundle(path, trusted=True)
    np.testing.assert_allclose(loaded.predict(partition="test"), expected, atol=1e-10)
    wanted = score(s.partition("test"), expected, "mse", weighted)
    assert loaded.automl.evaluate().metrics["mse"] == pytest.approx(wanted)
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("method", METHODS)
def test_expiry_during_a_real_cv_trial_retains_the_completed_winner(method):
    s = population(False)
    cv = DelayedCV()
    opts = options("cv", "mse")
    opts.update(method=method, cv=cv)
    result = s.automl.run(**opts, time_budget=SECONDS)
    assert len(cv.calls) == 1 and len(result.trials) == 1
    assert any("time budget exhausted" in v for v in result.warnings)
    check_scores(s, result, "cv", False)


def test_nested_search_refuses_an_incomplete_outer_estimate_after_expiry(prior_bundle):
    s = restored(prior_bundle)
    cv = DelayedCV()
    opts = options("nested", "mse")
    opts["cv"] = cv
    before = snapshot(s)
    with pytest.raises(ValidationError, match="budget.*nested"):
        s.automl.run(**opts, time_budget=SECONDS)
    assert len(cv.calls) == 1
    assert_preserved(s, before)


def test_ensembles_cannot_start_after_the_shared_single_trial_deadline():
    s = population(False)
    cv = DelayedCV(delay_on=4)
    opts = options("cv", "mse")
    opts.update(
        families=("ridge", "lasso"),
        n_trials=6,
        include_ensembles=True,
        cv=cv,
    )
    result = s.automl.run(**opts, time_budget=SECONDS)
    assert len(cv.calls) == 4 and len(result.trials) == 4
    assert {t.family for t in result.trials} == {"ridge", "lasso"}
    assert all(t.kind == "single" for t in result.trials)
    assert any("time budget exhausted" in v for v in result.warnings)


def test_real_optuna_timeout_retains_a_completed_trial():
    pytest.importorskip("optuna")
    s = population(False)
    cv = DelayedCV()
    opts = options("cv", "mse")
    opts.update(method="optuna", cv=cv)
    result = s.automl.run(**opts, time_budget=SECONDS)
    assert len(cv.calls) == 1 and len(result.trials) == 1
    assert any("time budget exhausted" in v for v in result.warnings)
    check_scores(s, result, "cv", False)


def test_valid_override_selects_the_effective_duration_without_mutating_a_bad_stored_cap():
    s = population(False)
    budget = AutoMLBudget(max_trials=2, max_time_seconds=np.nan)
    before = copy.deepcopy(budget)
    result = s.automl.run(**options("validation", "mse"), budget=budget, time_budget=np.int64(600))
    assert joblib.hash(budget) == joblib.hash(before)
    assert result.config["budget"]["max_time_seconds"] == 600.0
    check_scores(s, result, "validation", False)
