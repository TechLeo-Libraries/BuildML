"""Undefined fold evidence cannot choose a model; valid losses retain raw-row arithmetic."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, LeaveOneOut, PredefinedSplit

from buildml import Session
from buildml.core.errors import LeakageError, ValidationError
from buildml.model.selection import cv_score, nested_cv_score
from buildml.pipeline import predict_from_pipeline

ROUTES = ["cv_score", "grid_search", "randomized_search", "evolutionary_search"]
ALPHAS = [0.1, 2.0]
FOLD_IDS = np.asarray([0] + [1] * 5 + [2] * 6)


def population(weighted):
    rng = np.random.default_rng(763)
    x, z = rng.normal(size=(2, 18))
    frame = pd.DataFrame(
        {
            "x": x,
            "z": z,
            "y": 0.8 + 1.7 * x - 0.4 * z + 0.5 * np.sin(np.arange(18)),
            "weight": np.asarray([0.125, 0.3, 0.7, 1.2, 2.1, 0.25])[np.arange(18) % 6],
        },
        index=np.repeat([9, 3, 7], 6),
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
        .inject_split(
            train_indices=[7, 2, 10, 0, 5, 11, 4, 1, 9, 3, 8, 6],
            validation_indices=[12, 13, 14],
            test_indices=[15, 16, 17],
        )
    )


def prepared(weighted):
    s = population(weighted)
    s.grid_search(Ridge(), {"alpha": [0.7]}, cv=2, ranking_metric="mse")
    s.cv_score(Ridge(), cv=2, scoring_metric="mse")
    s.nested_cv_score(
        Ridge(), param_grid={"alpha": [0.7]}, outer_cv=2, inner_cv=2, scoring_metric="mse"
    )
    return s


def snapshot(s):
    return (
        s.fit_result,
        s.last_cv,
        s.last_search,
        s.last_nested_cv,
        s.history,
        s.dataset.to_pandas().copy(deep=True),
    )


def unchanged(s, before):
    fit, cv, search, nested, history, frame = before
    assert s.fit_result is fit and s.last_cv is cv and s.last_search is search
    assert s.last_nested_cv is nested and s.history == history
    pd.testing.assert_frame_equal(s.dataset.to_pandas(), frame)


def invoke(s, route, splitter, metric="r2", **kwargs):
    template = Ridge(alpha=0.7)
    common = dict(task="regression", cv=splitter, **kwargs)
    if route == "cv_score":
        result = s.cv_score(template, scoring_metric=metric, **common)
    elif route == "grid_search":
        result = s.grid_search(template, {"alpha": ALPHAS}, ranking_metric=metric, **common)
    elif route == "randomized_search":
        result = s.randomized_search(
            template, {"alpha": ALPHAS}, n_iter=2, ranking_metric=metric, **common
        )
    elif route == "evolutionary_search":
        result = s.evolutionary_search(
            template,
            param_space={"alpha": ALPHAS},
            population_size=4,
            n_generations=2,
            elite_size=1,
            tournament_size=2,
            max_evaluations=4,
            random_state=4,
            ranking_metric=metric,
            **common,
        )
    else:
        pytest.importorskip("optuna")
        result = s.optuna_search(
            template,
            param_space={"alpha": ALPHAS},
            n_trials=4,
            random_state=4,
            ranking_metric=metric,
            **common,
        )
    assert not hasattr(template, "coef_")
    return result


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("folds", ["leave_one_out", "mixed_sizes", "singleton_group"])
def test_undefined_selected_fold_refuses_before_replacing_any_session_results(
    route, weighted, folds
):
    s = prepared(weighted)
    before = snapshot(s)
    splitter = LeaveOneOut() if folds == "leave_one_out" else PredefinedSplit(FOLD_IDS)
    options = {}
    if folds == "singleton_group":
        splitter = GroupKFold(3)
        options["groups"] = pd.Series(FOLD_IDS, index=np.arange(12)[::-1] + 100)
    with pytest.warns(UndefinedMetricWarning, match="not well-defined"):
        with pytest.raises(ValidationError, match="finite.*r2|r2.*finite"):
            invoke(s, route, splitter, **options)
    unchanged(s, before)


@pytest.mark.parametrize("search", ["grid", "randomized", "evolutionary"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("undefined", ["inner", "outer"])
def test_nested_selected_evidence_requires_finite_inner_and_outer_folds(
    search, weighted, undefined
):
    s = prepared(weighted)
    before = snapshot(s)
    space = (
        {"param_grid": {"alpha": ALPHAS}}
        if search == "grid"
        else {"param_distributions": {"alpha": ALPHAS}, "n_iter": 2}
        if search == "randomized"
        else {
            "param_space": {"alpha": ALPHAS},
            "population_size": 4,
            "n_generations": 2,
            "n_trials": 4,
        }
    )
    with pytest.warns(UndefinedMetricWarning, match="not well-defined"):
        with pytest.raises(ValidationError, match="finite.*r2|r2.*finite"):
            s.nested_cv_score(
                Ridge(),
                task="regression",
                scoring_metric="r2",
                inner_search=search,
                outer_cv=LeaveOneOut() if undefined == "outer" else 2,
                inner_cv=LeaveOneOut() if undefined == "inner" else 2,
                **space,
            )
    unchanged(s, before)


def prediction(train, score, weighted, alpha):
    design = np.column_stack([np.ones(len(train)), train[["x", "z"]]])
    probe = np.column_stack([np.ones(len(score)), score[["x", "z"]]])
    w = train.weight.to_numpy() if weighted else np.ones(len(train))
    lhs = design.T * w
    beta = np.linalg.solve(lhs @ design + np.diag([0.0, alpha, alpha]), lhs @ train.y.to_numpy())
    return probe @ beta


def loss(frame, predicted, weighted, metric):
    errors = frame.y.to_numpy() - predicted
    w = frame.weight.to_numpy() if weighted else np.ones(len(frame))
    value = np.average(np.abs(errors) if metric == "mae" else errors**2, weights=w)
    return np.sqrt(value) if metric == "rmse" else value


def check_loss_consumers(s, result, route, weighted, metric, directory, capsys):
    train = s.partition("train")
    candidate_alphas = (
        [0.7] if route == "cv_score" else [trial.params["alpha"] for trial in result.trials]
    )
    means = []
    for alpha in candidate_alphas:
        expected = []
        for fold in [0, 1, 2]:
            fit, score = (
                train.iloc[np.flatnonzero(FOLD_IDS != fold)],
                train.iloc[np.flatnonzero(FOLD_IDS == fold)],
            )
            expected.append(loss(score, prediction(fit, score, weighted, alpha), weighted, metric))
        card = (
            result
            if route == "cv_score"
            else next(t.cv for t in result.trials if t.params["alpha"] == alpha)
        )
        assert [f.metrics[metric] for f in card.folds] == pytest.approx(expected, abs=1e-11)
        assert card.mean_metrics[metric] == pytest.approx(np.mean(expected), abs=1e-11)
        assert card.std_metrics[metric] == pytest.approx(np.std(expected, ddof=1), abs=1e-11)
        assert [(f.n_train, f.n_eval) for f in card.folds] == [(11, 1), (7, 5), (6, 6)]
        assert np.isnan(card.folds[0].metrics["r2"]) and np.isnan(card.mean_metrics["r2"])
        encoded = json.loads(json.dumps(card.to_dict(), allow_nan=False))
        assert encoded["folds"][0]["metrics"]["r2"] is None
        assert encoded["mean_metrics"]["r2"] is encoded["std_metrics"]["r2"] is None
        assert all(encoded["folds"][i]["metrics"]["r2"] is not None for i in [1, 2])
        assert card.to_frame()[metric].tolist() == pytest.approx(expected)
        assert card.population == "train" and set(card.held_out_partitions) == {
            "test",
            "validation",
        }
        means.append((np.mean(expected), alpha))
    json.dumps(result.to_dict(), allow_nan=False)
    result.show()
    assert metric in capsys.readouterr().out
    if route != "cv_score":
        score, alpha = min(means)
        assert result.best_score == pytest.approx(score, abs=1e-11)
        assert result.best_params == {"alpha": alpha}
        assert s.fit_result is result.refit_result
        path = s.save_pipeline(directory / "winner", evaluate_partition=None)
        incoming = s.partition("test")
        saved = predict_from_pipeline(path, incoming[["z", "x"]], trusted=True)
        np.testing.assert_allclose(
            saved.predictions, prediction(train, incoming, weighted, alpha), atol=1e-11
        )


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("metric", ["mae", "mse", "rmse"])
def test_defined_loss_scores_and_saved_winners_match_weighted_normal_equations(
    route, weighted, metric, tmp_path, capsys
):
    s = population(weighted)
    before = s.dataset.to_pandas().copy(deep=True)
    with pytest.warns(UndefinedMetricWarning, match="not well-defined"):
        result = invoke(s, route, PredefinedSplit(FOLD_IDS), metric)
    check_loss_consumers(s, result, route, weighted, metric, tmp_path, capsys)
    pd.testing.assert_frame_equal(s.to_pandas(), before)


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("weighted", [False, True])
def test_real_predefined_split_with_no_eval_folds_refuses_and_preserves_results(route, weighted):
    s = prepared(weighted)
    before = snapshot(s)
    with pytest.raises(ValidationError, match="no folds"):
        invoke(s, route, PredefinedSplit([-1] * 12), metric="mse")
    unchanged(s, before)


@pytest.mark.parametrize("empty", ["inner", "outer"])
@pytest.mark.parametrize("weighted", [False, True])
def test_nested_predefined_empty_evidence_refuses_and_preserves_results(empty, weighted):
    s = prepared(weighted)
    before = snapshot(s)
    # With two outer folds, the inner train population has exactly six rows.
    with pytest.raises(ValidationError, match="no .*folds|no folds"):
        s.nested_cv_score(
            Ridge(),
            param_grid={"alpha": ALPHAS},
            scoring_metric="mse",
            outer_cv=PredefinedSplit([-1] * 12) if empty == "outer" else 2,
            inner_cv=PredefinedSplit([-1] * 6) if empty == "inner" else 2,
        )
    unchanged(s, before)


@pytest.mark.parametrize("operation", ["cv_score", "nested_cv_score"])
@pytest.mark.parametrize("partition", ["test", "validation"])
@pytest.mark.parametrize("weighted", [False, True])
def test_external_split_plan_with_holdout_overlap_is_refused(operation, partition, weighted):
    s = prepared(weighted)
    before = snapshot(s)
    plan = replace(s.split_plan, **{partition + "_indices": (s.split_plan.train_indices[0],)})
    template = Ridge()
    with pytest.raises(LeakageError, match="overlap"):
        if operation == "cv_score":
            cv_score(s.dataset, plan, template, task="regression", cv=2)
        else:
            nested_cv_score(
                s.dataset, plan, template, param_grid={"alpha": ALPHAS}, outer_cv=2, inner_cv=2
            )
    assert not hasattr(template, "coef_")
    unchanged(s, before)


@pytest.mark.parametrize("case", ["undefined", "valid_loss", "no_folds"])
@pytest.mark.parametrize("weighted", [False, True])
def test_real_optuna_uses_the_same_finite_fold_and_artifact_contract(
    case, weighted, tmp_path, capsys
):
    pytest.importorskip("optuna")
    s = prepared(weighted)
    before = snapshot(s)
    if case == "no_folds":
        with pytest.raises(ValidationError, match="no folds"):
            invoke(s, "optuna_search", PredefinedSplit([-1] * 12), metric="mse")
        unchanged(s, before)
    elif case == "undefined":
        with pytest.warns(UndefinedMetricWarning, match="not well-defined"):
            with pytest.raises(ValidationError, match="finite.*r2|r2.*finite"):
                invoke(s, "optuna_search", LeaveOneOut())
        unchanged(s, before)
    else:
        with pytest.warns(UndefinedMetricWarning, match="not well-defined"):
            result = invoke(s, "optuna_search", PredefinedSplit(FOLD_IDS), metric="mae")
        check_loss_consumers(s, result, "optuna_search", weighted, "mae", tmp_path, capsys)


@pytest.mark.parametrize("search", ["grid", "randomized", "evolutionary"])
@pytest.mark.parametrize("weighted", [False, True])
def test_nested_singleton_outer_losses_match_independent_inner_and_outer_fits(
    search, weighted, capsys
):
    s = prepared(weighted)
    previous = s.fit_result
    train = s.partition("train")
    space = (
        {"param_grid": {"alpha": ALPHAS}}
        if search == "grid"
        else {"param_distributions": {"alpha": ALPHAS}, "n_iter": 2}
        if search == "randomized"
        else {
            "param_space": {"alpha": ALPHAS},
            "population_size": 4,
            "n_generations": 2,
            "n_trials": 4,
        }
    )
    with pytest.warns(UndefinedMetricWarning, match="not well-defined"):
        result = s.nested_cv_score(
            Ridge(),
            task="regression",
            inner_search=search,
            scoring_metric="mse",
            outer_cv=LeaveOneOut(),
            inner_cv=2,
            random_state=4,
            **space,
        )
    outer_scores = []
    for held_out, fold in enumerate(result.outer_folds):
        outer_fit = train.iloc[[i for i in range(len(train)) if i != held_out]]
        outer_eval = train.iloc[[held_out]]
        order = np.arange(len(outer_fit))
        np.random.RandomState(42).shuffle(order)
        candidate_scores = {}
        for alpha in ALPHAS:
            scores = []
            for eval_positions in [order[:6], order[6:]]:
                inner_fit = outer_fit.iloc[
                    [i for i in range(len(outer_fit)) if i not in eval_positions]
                ]
                inner_eval = outer_fit.iloc[eval_positions]
                scores.append(
                    loss(
                        inner_eval,
                        prediction(inner_fit, inner_eval, weighted, alpha),
                        weighted,
                        "mse",
                    )
                )
            candidate_scores[alpha] = np.mean(scores)
        chosen = fold.best_params["alpha"]
        assert chosen in ALPHAS
        assert fold.inner_best_score == pytest.approx(candidate_scores[chosen], abs=1e-11)
        if search != "evolutionary":
            assert chosen == min(candidate_scores, key=candidate_scores.get)
        assert 1 <= fold.inner_n_trials <= 2
        expected = loss(
            outer_eval, prediction(outer_fit, outer_eval, weighted, chosen), weighted, "mse"
        )
        assert fold.metrics["mse"] == pytest.approx(expected, abs=1e-11)
        assert (fold.n_train, fold.n_eval) == (11, 1)
        outer_scores.append(expected)
    assert result.mean_metrics["mse"] == pytest.approx(np.mean(outer_scores), abs=1e-11)
    assert result.std_metrics["mse"] == pytest.approx(np.std(outer_scores, ddof=1), abs=1e-11)
    payload = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert payload["mean_metrics"]["r2"] is payload["std_metrics"]["r2"] is None
    assert all(f["metrics"]["r2"] is None for f in payload["outer_folds"])
    assert result.to_frame()["mse"].tolist() == pytest.approx(outer_scores)
    result.show()
    assert "outer mse" in capsys.readouterr().out
    assert s.fit_result is previous
