"""Integral fold counts must retain population strategy and search budgets."""

import copy
import importlib.util
import json
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
)

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.model.selection import optuna_search

OPTUNA = importlib.util.find_spec("optuna") is not None
COUNTS = [int, np.int32, np.int64]
BAD_COUNTS = [True, np.bool_(True), 1.5, 2.0, "3", None, np.nan, np.inf, 0, -1]


def forbid_fit(monkeypatch):
    fitting = Mock(side_effect=AssertionError("Invalid control reached fitting"))

    def guarded_fit(self, X, y, sample_weight=None):
        return fitting(X, y, sample_weight=sample_weight)

    monkeypatch.setattr(Ridge, "fit", guarded_fit)
    return fitting


def make_session(kind="kfold", classification=False):
    rows = np.arange(90)
    x = np.sin(rows / 3) + rows / 10
    z = np.cos(rows / 5)
    groups = rows // 6
    y = groups % 2 if classification else 3 * x - z + .02 * rows ** 2
    frame = pd.DataFrame({"x": x, "z": z, "y": y, "subject": groups,
                          "clock": pd.Timestamp("2024-01-01") + pd.to_timedelta(rows, unit="D"),
                          "weight": 1 + rows % 4})
    frame = frame.iloc[np.random.default_rng(91).permutation(len(frame))]
    frame.index = np.arange(len(frame)) // 2 + 1000
    roles = {"x": "feature", "z": "feature", "y": "target", "weight": "weight",
             "subject": "group" if "group" in kind else "ignore",
             "clock": "time" if "time" in kind else "ignore"}
    return (Session.ingest(frame).set_roles(roles)
            .split(test_size=.2, validation_size=.1, random_state=19))


def regression_metrics(fit, score, alpha=.7):
    design = np.column_stack([np.ones(len(fit)), fit[["x", "z"]]])
    weighted = design.T * fit.weight.to_numpy()
    coefficients = np.linalg.solve(weighted @ design + np.diag([0., alpha, alpha]),
                                   weighted @ fit.y.to_numpy())
    prediction = np.column_stack([np.ones(len(score)), score[["x", "z"]]]) @ coefficients
    errors = score.y.to_numpy() - prediction
    mse = np.average(errors ** 2, weights=score.weight)
    variance = np.average((score.y - np.average(score.y, weights=score.weight)) ** 2,
                          weights=score.weight)
    return {"mae": np.average(abs(errors), weights=score.weight), "mse": mse,
            "rmse": np.sqrt(mse), "r2": 1 - mse / variance}


def classification_metrics(fit, score):
    labels = [0, 1]
    weights = np.array([fit.weight[fit.y == label].sum() for label in labels])
    predicted = labels[int(np.argmax(weights))]
    confusion = np.zeros((2, 2))
    for label in labels:
        confusion[label, predicted] = score.weight[score.y == label].sum()
    support = confusion.sum(axis=1)
    recall = np.diag(confusion) / support
    f1 = np.divide(2 * np.diag(confusion), support + confusion.sum(axis=0),
                   out=np.zeros(2), where=support + confusion.sum(axis=0) > 0)
    return {"accuracy": np.trace(confusion) / support.sum(),
            "balanced_accuracy": recall.mean(), "f1_macro": f1.mean(),
            "f1_weighted": np.average(f1, weights=support)}


def fold_oracles(train, strategy, classification=False, count=3):
    if strategy == "time":
        train = train.sort_values("clock", kind="mergesort")
        splitter = TimeSeriesSplit(count)
    elif strategy == "group":
        splitter = GroupKFold(count)
    elif strategy == "stratified_group":
        splitter = StratifiedGroupKFold(count, shuffle=True, random_state=42)
    elif strategy == "stratified":
        splitter = StratifiedKFold(count, shuffle=True, random_state=42)
    else:
        splitter = KFold(count, shuffle=True, random_state=42)
    group = train.subject.to_numpy() if "group" in strategy else None
    splits = splitter.split(train[["x", "z"]], train.y, group) if group is not None else splitter.split(train, train.y)
    expected = []
    for fit_pos, score_pos in splits:
        fit, score = train.iloc[fit_pos], train.iloc[score_pos]
        if strategy == "time":
            assert fit.clock.max() < score.clock.min()
        if "group" in strategy:
            assert set(fit.subject).isdisjoint(score.subject)
        metrics = classification_metrics(fit, score) if classification else regression_metrics(fit, score)
        expected.append((len(fit), len(score), metrics))
    return expected


@pytest.mark.parametrize("count_type", COUNTS)
@pytest.mark.parametrize("kind,requested,resolved,classification", [
    ("time", "time", "time", False), ("time", "auto", "time", False),
    ("group", "group", "group", False), ("group", "auto", "group", False),
    ("group", "stratified_group", "stratified_group", True),
    ("group", "auto", "stratified_group", True),
    ("kfold", "kfold", "kfold", False), ("stratified", "stratified", "stratified", True),
])
def test_integer_fold_types_match_population_and_independent_metrics(count_type, kind, requested,
                                                                    resolved, classification):
    session = make_session(kind, classification)
    train = session.partition("train").copy(deep=True)
    holdout = session.partition("test").copy(deep=True)
    expected = fold_oracles(train, resolved, classification)
    estimator = DummyClassifier() if classification else Ridge(alpha=.7)
    result = session.cv_score(estimator, cv=count_type(3), cv_strategy=requested,
                              task="classification" if classification else "regression")
    assert result.cv_strategy == resolved
    for actual, (n_fit, n_score, metrics) in zip(result.folds, expected, strict=True):
        assert (actual.n_train, actual.n_eval) == (n_fit, n_score)
        assert actual.metrics == pytest.approx(metrics)
    for key in expected[0][2]:
        values = [fold[2][key] for fold in expected]
        assert result.mean_metrics[key] == pytest.approx(np.mean(values))
        assert result.std_metrics[key] == pytest.approx(np.std(values, ddof=1))
    assert result.population == "train" and result.n_splits == 3
    assert session.history[-1]["parameters"]["cv"] == 3
    json.dumps(result.to_dict(), allow_nan=False)
    pd.testing.assert_frame_equal(session.partition("test"), holdout)
    assert session.fit_result is None


@pytest.mark.parametrize("count_type", COUNTS)
@pytest.mark.parametrize("strategy", ["time", "group"])
def test_nested_integer_folds_preserve_both_loops_and_outer_numerical_scores(count_type, strategy):
    session = make_session(strategy)
    expected = fold_oracles(session.partition("train"), strategy)
    result = session.nested_cv_score(Ridge(), param_grid={"alpha": [.7]},
                                     outer_cv=count_type(3), inner_cv=count_type(2),
                                     cv_strategy=strategy, task="regression", scoring_metric="rmse")
    assert result.outer_cv_strategy == strategy and result.inner_cv_strategy == strategy
    assert result.n_inner_splits == 2
    for actual, (_, _, metrics) in zip(result.outer_folds, expected, strict=True):
        assert actual.metrics == pytest.approx(metrics)
        assert actual.best_params == {"alpha": .7}
        assert actual.inner_n_trials == 1
    assert session.history[-1]["parameters"]["outer_cv"] == 3
    assert session.history[-1]["parameters"]["inner_cv"] == 2
    assert session.fit_result is None


@pytest.mark.parametrize("route", ["cv_score", "grid_search", "nested_outer", "nested_inner"])
@pytest.mark.parametrize("value", [value for value in BAD_COUNTS if value is not None])
def test_invalid_fold_controls_preserve_existing_model_and_history(route, value, monkeypatch):
    session = make_session().fit(Ridge(alpha=.7), task="regression")
    oracle = session.predict().copy()
    fitted, history = session.fit_result, copy.deepcopy(session.history)
    fitting = forbid_fit(monkeypatch)
    with pytest.raises(ValidationError, match="cv|CV"):
        if route == "cv_score":
            session.cv_score(Ridge(), cv=value, task="regression")
        elif route == "grid_search":
            session.grid_search(Ridge(), {"alpha": [.7]}, cv=value, task="regression")
        else:
            options = {"outer_cv": value, "inner_cv": 2} if route == "nested_outer" else {
                "outer_cv": 2, "inner_cv": value}
            session.nested_cv_score(Ridge(), param_grid={"alpha": [.7]}, task="regression", **options)
    fitting.assert_not_called()
    assert session.fit_result is fitted and session.history == history
    np.testing.assert_allclose(session.predict(), oracle, atol=1e-12)


SEARCH_ROUTES = ["randomized", "nested_randomized", "nested_evolutionary",
                 pytest.param("optuna", marks=pytest.mark.skipif(not OPTUNA, reason="Optuna absent")),
                 pytest.param("nested_optuna", marks=pytest.mark.skipif(not OPTUNA, reason="Optuna absent"))]


@pytest.mark.parametrize("route", SEARCH_ROUTES)
@pytest.mark.parametrize("value", BAD_COUNTS)
def test_invalid_search_budgets_refuse_before_backend_fitting(route, value, monkeypatch):
    session = make_session().fit(Ridge(alpha=.7), task="regression")
    fitted = session.fit_result
    oracle = session.predict().copy()
    history = copy.deepcopy(session.history)
    fitting = forbid_fit(monkeypatch)
    name = "n_iter" if "randomized" in route else "n_trials"
    with pytest.raises(ValidationError, match=name):
        if route == "randomized":
            session.randomized_search(Ridge(), {"alpha": [.7, 2.]}, n_iter=value, cv=2)
        elif route == "optuna":
            session.optuna_search(Ridge(), param_space={"alpha": [.7]}, n_trials=value, cv=2)
        else:
            options = {"param_distributions": {"alpha": [.7, 2.]}} if route == "nested_randomized" else {
                "param_space": {"alpha": [.7]}}
            session.nested_cv_score(Ridge(), outer_cv=2, inner_cv=2,
                                     inner_search=route.removeprefix("nested_"),
                                     population_size=2, n_generations=1, **{name: value}, **options)
    fitting.assert_not_called()
    assert session.fit_result is fitted and session.last_search is None and session.last_nested_cv is None
    assert session.history == history
    np.testing.assert_allclose(session.predict(), oracle, atol=1e-12)


@pytest.mark.skipif(not OPTUNA, reason="Optuna absent")
@pytest.mark.parametrize("value", [True, 2.5, "2", np.nan, np.inf])
def test_invalid_budget_cannot_advance_existing_real_study(value, monkeypatch):
    import optuna

    session = make_session()
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: 1., n_trials=1)
    previous = copy.deepcopy(study.get_trials())
    fitting = forbid_fit(monkeypatch)
    with pytest.raises(ValidationError, match="n_trials"):
        optuna_search(session.dataset, session.split_plan, Ridge(), study=study,
                      param_space={"alpha": [.7]}, n_trials=value, cv=2, ranking_metric="rmse")
    assert study.get_trials() == previous
    fitting.assert_not_called()


@pytest.mark.parametrize("count_type", COUNTS)
@pytest.mark.parametrize("engine", ["randomized", pytest.param("optuna", marks=pytest.mark.skipif(
    not OPTUNA, reason="Optuna absent"))])
def test_valid_integer_budgets_bound_actual_trials_and_retain_independent_scores(count_type, engine, capsys):
    session = make_session()
    expected = fold_oracles(session.partition("train"), "kfold")
    if engine == "randomized":
        result = session.randomized_search(Ridge(), {"alpha": [.7, .7]},
                                            n_iter=count_type(2), cv=count_type(3), cv_strategy="kfold",
                                            ranking_metric="rmse", task="regression", refit=False)
    else:
        result = session.optuna_search(Ridge(), param_space={"alpha": [.7]}, n_trials=count_type(2),
                                       cv=count_type(3), cv_strategy="kfold", ranking_metric="rmse",
                                       task="regression", refit=False)
        assert len(result.study.trials) == 2
    assert len(result.trials) == 2
    values = [fold[2]["rmse"] for fold in expected]
    for trial in result.trials:
        assert trial.mean_score == pytest.approx(np.mean(values))
        assert trial.std_score == pytest.approx(np.std(values, ddof=1))
    assert result.best_cv.cv_strategy == "kfold"
    assert session.history[-1]["parameters"]["cv"] == 3
    assert session.history[-1]["parameters"]["n_iter" if engine == "randomized" else "n_trials"] == 2
    assert session.fit_result is None
    table = result.to_frame()
    assert table["mean_score"].tolist() == pytest.approx([np.mean(values)] * 2)
    assert table["std_score"].tolist() == pytest.approx([np.std(values, ddof=1)] * 2)
    assert table["param_alpha"].tolist() == [.7, .7]
    payload = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert payload["method"] == engine and payload["refit"] is None
    assert payload["best_params"] == {"alpha": .7}
    assert [trial["mean_score"] for trial in payload["trials"]] == pytest.approx([np.mean(values)] * 2)
    result.show()
    display = capsys.readouterr().out
    assert engine in display and "ranked by rmse" in display
    assert f"{np.mean(values):.6f}" in display and f"{np.std(values, ddof=1):.6f}" in display


@pytest.mark.parametrize("use_default", [False, True])
def test_custom_splitter_and_none_default_keep_sklearn_compatibility(use_default):
    session = make_session()
    count = 5 if use_default else 3
    cv = None if use_default else KFold(count, shuffle=True, random_state=42)
    result = session.cv_score(Ridge(alpha=.7), cv=cv, task="regression")
    assert result.n_splits == count and result.cv_strategy == "KFold"
    if not use_default:
        expected = fold_oracles(session.partition("train"), "kfold")
        assert result.mean_metrics["rmse"] == pytest.approx(np.mean([item[2]["rmse"] for item in expected]))


@pytest.mark.parametrize("count_type", [int, np.int64])
@pytest.mark.parametrize("problem", ["multiple_group_roles", "unparseable_time", "regression_stratified_group",
                                     "unknown_strategy"])
def test_population_strategy_failures_cannot_be_bypassed_by_integer_type(count_type, problem, monkeypatch):
    session = make_session("group")
    strategy = "group"
    if problem == "multiple_group_roles":
        session.set_roles({"clock": "group"})
    elif problem == "unparseable_time":
        frame = session.to_pandas().assign(clock="unparseable")
        session = (Session.ingest(frame).set_roles({"x": "feature", "z": "feature", "y": "target",
                                                    "clock": "time", "subject": "ignore", "weight": "weight"})
                   .split(test_size=.2, random_state=19))
        strategy = "time"
    elif problem == "regression_stratified_group":
        strategy = "stratified_group"
    else:
        strategy = "unknown"
    history = copy.deepcopy(session.history)
    fitting = forbid_fit(monkeypatch)
    with pytest.raises(ValidationError):
        session.cv_score(Ridge(), task="regression", cv=count_type(3), cv_strategy=strategy)
    fitting.assert_not_called()
    assert session.fit_result is None and session.history == history


def test_unused_nested_search_budgets_do_not_change_grid_behavior():
    session = make_session()
    result = session.nested_cv_score(Ridge(), param_grid={"alpha": [.7]}, outer_cv=2, inner_cv=2,
                                     task="regression", n_iter=None, n_trials=None)
    assert result.search_method == "grid"
    assert all(fold.inner_n_trials == 1 for fold in result.outer_folds)


@pytest.mark.parametrize("count_type", [int, np.int64])
@pytest.mark.parametrize("method", ["randomized", "evolutionary", pytest.param("optuna", marks=pytest.mark.skipif(
    not OPTUNA, reason="Optuna absent"))])
def test_nested_used_budget_is_recorded_as_a_number_and_bounds_actual_trials(count_type, method):
    session = make_session()
    budget = 3 if method == "evolutionary" else 2
    options = {"param_distributions": {"alpha": [.7, .7]}, "n_iter": count_type(2)} if method == "randomized" else {
        "param_space": {"alpha": [.7]}, "n_trials": count_type(budget),
        "population_size": budget, "n_generations": 1}
    result = session.nested_cv_score(Ridge(), inner_search=method, outer_cv=count_type(2),
                                     inner_cv=count_type(2), task="regression", **options)
    key = "n_iter" if method == "randomized" else "n_trials"
    recorded = session.history[-1]["parameters"][key]
    assert type(recorded) is int and recorded == budget
    assert all(1 <= fold.inner_n_trials <= budget for fold in result.outer_folds)
    if method != "evolutionary":
        assert all(fold.inner_n_trials == 2 for fold in result.outer_folds)
    expected = fold_oracles(session.partition("train"), "kfold", count=2)
    for actual, (_, _, metrics) in zip(result.outer_folds, expected, strict=True):
        assert actual.metrics == pytest.approx(metrics)
