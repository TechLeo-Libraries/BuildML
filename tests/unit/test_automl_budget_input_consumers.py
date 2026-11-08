"""AutoML count budgets preserve borrowed inputs, winners, and saved numerical evidence."""

import copy
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from test_automl_refit_consumers import cv_scores, options, population, predict, score

from buildml.automl import AutoMLBudget
from buildml.core.errors import ValidationError

COUNT_FIELDS = [
    "n_trials",
    "max_ensemble_bases",
    "max_trials",
    "max_families",
    "max_recipe_strategies",
    "max_ensemble_trials",
]
INVALID_COUNTS = [
    (name, bad)
    for name in COUNT_FIELDS
    for bad in [True, np.bool_(False), 2.5, "2", np.nan, np.inf, 0, -1]
    if not (name == "max_ensemble_trials" and type(bad) is int and bad == 0)
]
INVALID_COUNTS.append(("max_ensemble_bases", 1))


@pytest.fixture(scope="module")
def prior_bundle(tmp_path_factory):
    s = population(False)
    s.automl.run(**options("validation", "mse"))
    return s.automl.save_bundle(tmp_path_factory.mktemp("budget-prior") / "winner")


def restored(prior_bundle):
    s = population(False)
    s.automl.load_bundle(prior_bundle, trusted=True)
    s.automl.evaluate(partition="test")
    return s


def snapshot(s):
    predicted = np.asarray(s.predict(partition="test")).copy()
    return (
        s.automl_plan,
        s.fit_result,
        s.automl_result,
        joblib.hash((s.automl_plan, s.fit_result)),
        copy.deepcopy(s.history),
        s.dataset.to_pandas().copy(deep=True),
        predicted,
    )


def assert_preserved(s, before):
    plan, fitted, result, fingerprint, history, data, predicted = before
    assert s.automl_plan is plan and s.fit_result is fitted and s.automl_result is result
    assert joblib.hash((s.automl_plan, s.fit_result)) == fingerprint
    pd.testing.assert_frame_equal(s.dataset.to_pandas(), data)
    assert s.history == history
    np.testing.assert_allclose(s.predict(partition="test"), predicted, atol=1e-10)


@pytest.mark.parametrize(
    "name,bad",
    INVALID_COUNTS,
    ids=[f"{name}-{i}" for i, (name, _) in enumerate(INVALID_COUNTS)],
)
def test_bad_resource_counts_refuse_without_replacing_loaded_winner(name, bad, prior_bundle):
    s = restored(prior_bundle)
    opts = options("validation", "mse")
    budget = AutoMLBudget(max_trials=2, max_time_seconds=120.0)
    if name in {"n_trials", "max_ensemble_bases"}:
        opts[name] = bad
    else:
        setattr(budget, name, bad)
    fingerprint = joblib.hash(budget)
    before = snapshot(s)
    with pytest.raises(ValidationError, match=name):
        s.automl.run(**opts, budget=budget)
    assert joblib.hash(budget) == fingerprint
    assert_preserved(s, before)


@pytest.mark.parametrize("budget", [{}, [], "budget"])
def test_structured_budget_requires_the_declared_type_and_preserves_prior_memory(
    budget, prior_bundle
):
    s = restored(prior_bundle)
    before = snapshot(s)
    with pytest.raises(ValidationError, match="AutoMLBudget"):
        s.automl.run(**options("validation", "mse"), budget=budget)
    assert_preserved(s, before)


@pytest.mark.parametrize("selection", ["cv", "nested", "validation"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("scalar", [int, np.int64], ids=["python", "numpy"])
def test_valid_count_caps_and_borrowed_override_keep_raw_scores_and_real_bundles(
    selection, weighted, scalar, tmp_path
):
    s = population(weighted)
    original = s.dataset.to_pandas().copy(deep=True)
    budget = AutoMLBudget(
        max_trials=scalar(2),
        max_families=scalar(1),
        max_recipe_strategies=scalar(1),
        max_ensemble_trials=scalar(0),
        max_time_seconds=120.0,
    )
    fingerprint = joblib.hash(budget)
    opts = options(selection, "mse")
    opts.update(n_trials=scalar(3), max_ensemble_bases=scalar(2), include_ensembles=True)
    result = s.automl.run(**opts, budget=budget, time_budget=600.0)
    assert joblib.hash(budget) == fingerprint
    assert len(result.trials) == 2
    assert result.config["n_trials"] == 2 and type(result.config["n_trials"]) is int
    assert type(result.config["max_ensemble_bases"]) is int
    caps = result.config["budget"]
    for field, value in [
        ("max_trials", 2),
        ("max_families", 1),
        ("max_recipe_strategies", 1),
        ("max_ensemble_trials", 0),
    ]:
        assert caps[field] == value and type(caps[field]) is int
    assert caps["max_time_seconds"] == 600.0
    assert all(t.family == "ridge" and t.kind == "single" for t in result.trials)
    train, validation, test = s.partition("train"), s.partition("validation"), s.partition("test")
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
    assert result.best_score == pytest.approx(best.mean_score)
    expected = predict(train, test, best.params["alpha"], weighted)
    np.testing.assert_allclose(s.predict(partition="test"), expected, atol=1e-10)
    expected_mse = score(test, expected, "mse", weighted)
    assert s.automl.evaluate().metrics["mse"] == pytest.approx(expected_mse)
    json.dumps(result.to_dict(), allow_nan=False)
    saved = s.automl.save_bundle(tmp_path / "count-winner")
    metadata = json.loads((saved / "meta.json").read_text())
    assert metadata["plan"]["config"]["budget"] == caps
    loaded = population(weighted)
    loaded.automl.load_bundle(saved, trusted=True)
    np.testing.assert_allclose(loaded.predict(partition="test"), expected, atol=1e-10)
    assert loaded.automl.evaluate().metrics["mse"] == pytest.approx(expected_mse)
    # A later caller edit must not rewrite recorded search/bundle provenance.
    budget.max_trials = 99
    assert result.config["budget"]["max_trials"] == 2
    assert s.automl_plan.config["budget"]["max_trials"] == 2
    pd.testing.assert_frame_equal(s.dataset.to_pandas(), original)


def test_failure_after_a_valid_override_does_not_mutate_the_borrowed_budget(prior_bundle):
    s = restored(prior_bundle)
    opts = options("validation", "mse")
    opts["families"] = ("missing-family",)
    budget = AutoMLBudget(max_trials=2, max_time_seconds=120.0)
    fingerprint = joblib.hash(budget)
    before = snapshot(s)
    with pytest.raises(ValidationError, match="Unknown AutoML families"):
        s.automl.run(**opts, budget=budget, time_budget=600.0)
    assert joblib.hash(budget) == fingerprint
    assert_preserved(s, before)


@pytest.mark.parametrize("name", ["method", "ensemble_mode", "selection"])
def test_invalid_search_modes_refuse_before_replacing_loaded_evidence(name, prior_bundle):
    s = restored(prior_bundle)
    opts = options("validation", "mse")
    opts[name] = "unknown"
    before = snapshot(s)
    with pytest.raises(ValidationError, match=name):
        s.automl.run(**opts)
    assert_preserved(s, before)
