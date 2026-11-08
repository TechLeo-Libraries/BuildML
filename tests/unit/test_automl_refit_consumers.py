"""Search-only AutoML preserves models and independently verified selection evidence."""

import inspect
import sys
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.preprocess import PreprocessRecipe


def population(weighted):
    rng = np.random.default_rng(418)
    x = rng.normal(size=(72, 2))
    frame = pd.DataFrame(
        {
            "x": x[:, 0],
            "z": x[:, 1],
            "y": x @ [2.0, -0.4] + rng.normal(size=72),
            "weight": np.where(np.arange(72) % 7 == 0, 13.0, 1.0),
        },
        index=[f"source-{i}" for i in range(72)],
    )
    return (
        Session.ingest(frame)
        .set_roles(
            {
                "x": "feature",
                "z": "feature",
                "y": "target",
                "weight": "weight" if weighted else "ignore",
            }
        )
        .split(test_size=0.2, validation_size=0.2, random_state=9)
    )


def predict(fitting, scoring, alpha, weighted):
    design = np.column_stack([np.ones(len(fitting)), fitting[["x", "z"]]])
    probe = np.column_stack([np.ones(len(scoring)), scoring[["x", "z"]]])
    weights = fitting.weight.to_numpy() if weighted else np.ones(len(fitting))
    cross = design.T * weights
    coefficients = np.linalg.solve(
        cross @ design + np.diag([0.0, alpha, alpha]), cross @ fitting.y.to_numpy()
    )
    return probe @ coefficients


def score(frame, prediction, metric, weighted):
    weights = frame.weight if weighted else None
    error = frame.y.to_numpy() - prediction
    if metric == "mae":
        return np.average(abs(error), weights=weights)
    mse = np.average(error**2, weights=weights)
    if metric == "mse":
        return mse
    mean = np.average(frame.y, weights=weights)
    return 1 - mse / np.average((frame.y - mean) ** 2, weights=weights)


def cv_scores(frame, alpha, metric, weighted):
    return [
        score(
            frame.iloc[evaluate],
            predict(frame.iloc[fit], frame.iloc[evaluate], alpha, weighted),
            metric,
            weighted,
        )
        for fit, evaluate in KFold(2, shuffle=True, random_state=42).split(frame)
    ]


def options(selection, metric):
    return dict(
        method="grid",
        selection=selection,
        task="regression",
        families=("ridge",),
        n_trials=3,
        cv=2,
        outer_cv=2,
        ranking_metric=metric,
        preprocess=PreprocessRecipe(impute=None),
        include_recipe_search=False,
        include_industry_families=False,
        random_state=7,
    )


@contextmanager
def ridge_fits():
    """Observe real sklearn fits without replacing their numerical implementation."""
    code = inspect.unwrap(Ridge.fit).__code__
    previous = sys.getprofile()
    populations = []

    def observe(frame, event, arg):
        if previous is not None:
            previous(frame, event, arg)
        if event == "call" and frame.f_code is code:
            populations.append(np.asarray(frame.f_locals["X"]).copy())

    sys.setprofile(observe)
    try:
        yield populations
    finally:
        sys.setprofile(previous)


def assert_selection(session, result, selection, metric, weighted):
    train, validation = session.partition("train"), session.partition("validation")
    assert len(result.trials) == 3
    for trial in result.trials:
        alpha = trial.params["alpha"]
        values = (
            [score(validation, predict(train, validation, alpha, weighted), metric, weighted)]
            if selection == "validation"
            else cv_scores(train, alpha, metric, weighted)
        )
        assert trial.mean_score == pytest.approx(np.mean(values), rel=1e-9, abs=1e-10)
        if len(values) > 1:
            assert trial.std_score == pytest.approx(np.std(values, ddof=1), abs=1e-10)
    choose = max if metric == "r2" else min
    winner = choose(result.trials, key=lambda trial: trial.mean_score)
    assert result.best_params == winner.params
    assert result.best_score == pytest.approx(winner.mean_score)
    if selection == "nested":
        outer = []
        for fit, evaluate in KFold(2, shuffle=True, random_state=42).split(train):
            fitting, scoring = train.iloc[fit], train.iloc[evaluate]
            candidates = [
                (np.mean(cv_scores(fitting, alpha, metric, weighted)), alpha)
                for alpha in [0.1, 1.0, 10.0]
            ]
            _, alpha = choose(candidates, key=lambda candidate: candidate[0])
            outer.append(
                score(scoring, predict(fitting, scoring, alpha, weighted), metric, weighted)
            )
        assert result.outer_score_mean == pytest.approx(np.mean(outer), abs=1e-9)
        assert result.outer_score_std == pytest.approx(np.std(outer, ddof=1), abs=1e-9)


@pytest.mark.parametrize("selection", ["cv", "nested", "validation"])
@pytest.mark.parametrize("metric", ["mae", "mse", "r2"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("prior_fit", [False, True])
def test_search_only_retains_selection_evidence_and_previous_model(
    selection, metric, weighted, prior_fit, tmp_path
):
    session = population(weighted)
    before = session.to_pandas().copy(deep=True)
    if prior_fit:
        session.fit(LinearRegression(), task="regression")
    previous = session.fit_result
    with ridge_fits() as fits:
        result = session.automl.run(refit=False, **options(selection, metric))
    assert len(fits) == {"cv": 6, "nested": 20, "validation": 3}[selection]
    allowed = set(map(tuple, session.partition("train")[["x", "z"]].to_numpy()))
    assert all(set(map(tuple, fitting)) <= allowed for fitting in fits)
    assert_selection(session, result, selection, metric, weighted)
    assert session.fit_result is previous
    assert session.automl_plan.estimator_ is None
    assert session.automl_result is result
    run = next(record for record in reversed(session.history) if record["action"] == "run_automl")
    assert run["parameters"]["refit"] is False
    history = session.history.copy()
    with pytest.raises(ValidationError, match="AutoML winner"):
        session.automl.evaluate()
    destination = tmp_path / "search-only"
    with pytest.raises(ValidationError):
        session.automl.save_bundle(destination)
    assert not destination.exists()
    assert session.history == history
    if prior_fit:
        np.testing.assert_allclose(
            session.predict(partition="test"),
            predict(session.partition("train"), session.partition("test"), 0.0, weighted),
            atol=1e-10,
        )
    pd.testing.assert_frame_equal(session.to_pandas(), before)


@pytest.mark.parametrize("selection", ["cv", "nested", "validation"])
@pytest.mark.parametrize("metric", ["mae", "mse", "r2"])
@pytest.mark.parametrize("weighted", [False, True])
def test_refitted_winner_evaluation_and_real_bundle_match_closed_form(
    selection, metric, weighted, tmp_path
):
    session = population(weighted)
    with ridge_fits() as fits:
        result = session.automl.run(refit=True, **options(selection, metric))
    assert len(fits) == {"cv": 7, "nested": 21, "validation": 4}[selection]
    assert_selection(session, result, selection, metric, weighted)
    assert session.fit_result.estimator is session.automl_plan.estimator_
    train, test = session.partition("train"), session.partition("test")
    expected = predict(train, test, result.best_params["alpha"], weighted)
    np.testing.assert_allclose(session.predict(partition="test"), expected, atol=1e-10)
    evaluation = session.automl.evaluate()
    assert evaluation.metrics[metric] == pytest.approx(score(test, expected, metric, weighted))
    assert evaluation.diagnostics["automl"]["best_params"] == result.best_params
    path = session.automl.save_bundle(tmp_path / "winner")
    restored = population(weighted)
    restored.automl.load_bundle(path, trusted=True)
    assert restored.fit_result.estimator is restored.automl_plan.estimator_
    np.testing.assert_allclose(restored.predict(partition="test"), expected, atol=1e-10)
    assert restored.automl.evaluate().metrics[metric] == pytest.approx(evaluation.metrics[metric])


@pytest.mark.parametrize("state", ["classical_only", "winner_replaced", "search_then_fit"])
@pytest.mark.parametrize("partition", ["train", "validation", "test"])
def test_automl_evaluation_never_attributes_an_unrelated_fit_to_the_winner(state, partition):
    session = population(False)
    if state != "classical_only":
        session.automl.run(refit=state == "winner_replaced", **options("cv", "mse"))
    session.fit(LinearRegression(), task="regression")
    history = session.history.copy()
    with pytest.raises(ValidationError, match="AutoML winner"):
        session.automl.evaluate(partition=partition)
    assert session.history == history
    np.testing.assert_allclose(
        session.predict(partition=partition),
        predict(session.partition("train"), session.partition(partition), 0.0, False),
        atol=1e-10,
    )
