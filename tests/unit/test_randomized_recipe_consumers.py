"""Randomized recipe search matches independent fold and saved-model calculations."""

import copy
import hashlib

import joblib
import numpy as np
import pandas as pd
import pytest
import test_agent_evolutionary_recipe_numerical_consumers as oracle
from scipy.stats import loguniform
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.pipeline import load_pipeline_bundle, predict_from_pipeline
from buildml.preprocess import PreprocessRecipe

BACKINGS = ("pandas", "pandas-parquet", "polars-lazy", "duckdb-lazy")


def _session(backing, poison, directory, categorical=False):
    original, x, y = oracle._session(poison)
    frame = original.to_pandas().copy(deep=True)
    roles = {"x": "feature", "z": "feature", "q": "feature", "y": "target"}
    if categorical:
        frame["category"] = ["first", "second"] * 48
        roles["category"] = "feature"
    engine = backing.split("-")[0]
    if engine != "pandas":
        pytest.importorskip(engine)
    if backing == "pandas":
        session = Session.ingest(frame)
    else:
        path = directory / "search.parquet"
        frame.to_parquet(path, index=False)
        session = Session.ingest(path, engine=engine, mode="memory" if engine == "pandas" else "lazy")
    session.set_roles(roles).inject_split(train_indices=list(range(64)),
        validation_indices=list(range(64, 80)), test_indices=list(range(80, 96)))
    return session, x, y


def _recipe():
    return PreprocessRecipe(impute="median", scale="standard", outliers="zscore",
        outlier_action="cap", outlier_columns=("z",), select="univariate",
        select_score_func="f_regression", select_columns=("x", "z"))


def _prediction(train, target, probe, threshold, k, alpha):
    train, probe = train[:, :2].copy(), probe[:, :2].copy()
    finite = train[:, 1][np.isfinite(train[:, 1])]
    center, spread = finite.mean(), finite.std(ddof=0)
    lower, upper = center - threshold*spread, center + threshold*spread
    train[:, 1] = np.clip(train[:, 1], lower, upper)
    probe[:, 1] = np.clip(probe[:, 1], lower, upper)
    return oracle._prediction(train, target, probe, "standard", k, alpha)


def _search(session, style, refit):
    recipe = _recipe()
    original = copy.deepcopy(recipe)
    result = session.randomized_search(Ridge(alpha=.4),
        param_distributions={"alpha": loguniform(.1, 5)} if style == "joint" else None,
        recipe_distributions={"zscore_threshold": [.75, 1.5, 3.], "select_k": [1, 2]},
        preprocess=recipe, n_iter=5, cv=3, random_state=42, task="regression",
        ranking_metric="mse", refit=refit)
    assert recipe == original
    return result


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("style", ("joint", "recipe-only"))
@pytest.mark.parametrize("refit", (False, True))
@pytest.mark.parametrize("poison", (False, True))
def test_recipe_trials_ranking_and_saved_refit_match_train_only_oracles(backing, style, refit, poison, tmp_path):
    session, x, y = _session(backing, poison, tmp_path)
    before = session.to_pandas().copy(deep=True)
    split = copy.deepcopy(session.split_plan)
    result = _search(session, style, refit)
    assert len(result.trials) == 5 and session.last_search is result
    for trial in result.trials:
        threshold, k = trial.recipe_knobs["zscore_threshold"], trial.recipe_knobs["select_k"]
        alpha = trial.params.get("alpha", .4)
        scores = []
        for fit, evaluate in KFold(3, shuffle=True, random_state=42).split(x[:64]):
            prediction = _prediction(x[fit], y[fit], x[evaluate], threshold, k, alpha)
            scores.append(np.mean((prediction-y[evaluate])**2))
        assert trial.mean_score == pytest.approx(np.mean(scores), rel=1e-8)
        assert trial.std_score == pytest.approx(np.std(scores, ddof=1), rel=1e-8)
    assert result.best_score == min(row.mean_score for row in result.trials)
    assert result.best_params == result.trials[0].params
    assert result.best_recipe_knobs == result.trials[0].recipe_knobs
    if refit:
        knobs = result.best_recipe_knobs
        expected = _prediction(x[:64], y[:64], x[80:], knobs["zscore_threshold"],
            knobs["select_k"], result.best_params.get("alpha", .4))
        np.testing.assert_allclose(session.predict(partition="test"), expected, rtol=1e-8, atol=1e-8)
        path = tmp_path / "trusted-pipeline"
        session.save_pipeline(path, evaluate_partition=None)
        bundle = load_pipeline_bundle(path, trusted=True)
        source = session.partition("test").copy(deep=True)
        weights = joblib.hash(bundle)
        files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file()}
        for stored in (path, bundle):
            replay = predict_from_pipeline(stored, source, trusted=True)
            np.testing.assert_allclose(replay.predictions, expected, rtol=1e-8, atol=1e-8)
            pd.testing.assert_index_equal(replay.predictions.index, source.index)
        assert joblib.hash(bundle) == weights
        assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file()} == files
    else:
        assert session.fit_result is None
    pd.testing.assert_frame_equal(session.to_pandas(), before)
    assert session.split_plan == split


@pytest.mark.parametrize("backing", BACKINGS)
def test_explicit_parameter_prefixes_match_joint_recipe_namespace(backing, tmp_path):
    session, _, _ = _session(backing, False, tmp_path)
    first = _search(session, "joint", False)
    second = session.randomized_search(Ridge(alpha=.4), param_distributions={
        "alpha": loguniform(.1, 5), "recipe__zscore_threshold": [.75, 1.5, 3.], "recipe__select_k": [1, 2]},
        preprocess=_recipe(), n_iter=5, cv=3, random_state=42, task="regression", ranking_metric="mse", refit=False)
    assert [(v.params, v.recipe_knobs) for v in first.trials] == [(v.params, v.recipe_knobs) for v in second.trials]
    np.testing.assert_allclose([v.mean_score for v in first.trials], [v.mean_score for v in second.trials], atol=1e-12)
    assert session.fit_result is None


PROBLEMS = ("missing", "empty", "unknown-knob", "prefix-in-recipe", "no-base", "missing-selector", "null-selector", "nonnumeric-selector")


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("problem", PROBLEMS)
def test_invalid_recipe_spaces_and_feature_sets_preserve_fitted_state(backing, problem, tmp_path):
    session, x, y = _session(backing, False, tmp_path, categorical=problem == "nonnumeric-selector")
    expected = None
    if problem != "nonnumeric-selector":
        session.fit(Pipeline([("impute", SimpleImputer(strategy="median")), ("ridge", Ridge(alpha=17))]), task="regression")
        expected = oracle._prediction(x[:64], y[:64], x[80:], None, 3, 17)
        np.testing.assert_allclose(session.predict(partition="test"), expected, rtol=1e-8, atol=1e-8)
    previous_fit, previous_search = session.fit_result, session.last_search
    before, history = session.to_pandas().copy(deep=True), copy.deepcopy(session.history)
    options = dict(preprocess=_recipe(), recipe_distributions={"select_k": [1, 2]}, n_iter=2,
        cv=3, task="regression", ranking_metric="mse", refit=True)
    if problem == "missing":
        options["recipe_distributions"] = None
    elif problem == "empty":
        options["recipe_distributions"] = {}
    elif problem in {"unknown-knob", "prefix-in-recipe"}:
        options["recipe_distributions"] = {"not_a_knob" if problem == "unknown-knob" else "recipe__select_k": [1, 2]}
    elif problem == "no-base":
        options["preprocess"] = None
    else:
        columns = ("absent",) if problem == "missing-selector" else (("category",) if problem == "nonnumeric-selector" else ("x", "z"))
        options["preprocess"] = PreprocessRecipe(impute=None, select="univariate", select_columns=columns)
    with pytest.raises(ValidationError):
        session.randomized_search(Ridge(), **options)
    assert session.fit_result is previous_fit and session.last_search is previous_search
    assert session.history == history
    pd.testing.assert_frame_equal(session.to_pandas(), before)
    if expected is not None:
        np.testing.assert_allclose(session.predict(partition="test"), expected, rtol=1e-8, atol=1e-8)
