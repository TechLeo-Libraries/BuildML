"""Nested AutoML publishes finite outer evidence from the intended rows."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    LeaveOneGroupOut,
    LeavePGroupsOut,
    StratifiedGroupKFold,
)

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.preprocess.fold import PreprocessRecipe


def _session(classification=False):
    rows = np.arange(72)
    groups = rows // 6
    x = np.sin(rows / 4) + rows / 10
    z = np.cos(rows / 7)
    target = (rows % 2) if classification else (1.8 * x + z + 0.01 * rows**2)
    data = pd.DataFrame({"x": x, "z": z, "y": target, "group": groups, "weight": 1 + rows % 4})
    order = np.r_[np.random.default_rng(13).permutation(48), np.arange(48, 72)]
    data = data.iloc[order].set_axis(np.arange(72) // 3)
    session = Session.ingest(data).set_roles(
        {"x": "feature", "z": "feature", "y": "target", "group": "group", "weight": "weight"}
    )
    return session.inject_split(
        train_indices=list(range(48)),
        validation_indices=list(range(48, 60)),
        test_indices=list(range(60, 72)),
    )


def _automl(session, *, metric, outer_cv, cv=2, n_trials=1):
    return session.automl.run(
        method="grid",
        selection="nested",
        task="regression",
        families=("ridge",),
        n_trials=n_trials,
        cv=cv,
        outer_cv=outer_cv,
        ranking_metric=metric,
        include_recipe_search=False,
        preprocess=PreprocessRecipe(impute=None),
        random_state=21,
    )


def _mse(fit, evaluate, alpha=0.1):
    design = np.column_stack([np.ones(len(fit)), fit[["x", "z"]]])
    weighted = design.T * fit.weight.to_numpy()
    coefficients = np.linalg.solve(
        weighted @ design + np.diag([0.0, alpha, alpha]), weighted @ fit.y.to_numpy()
    )
    inputs = np.column_stack([np.ones(len(evaluate)), evaluate[["x", "z"]]])
    return np.average((evaluate.y.to_numpy() - inputs @ coefficients) ** 2, weights=evaluate.weight)


@pytest.mark.parametrize("case", ("empty", "undefined_r2"))
def test_outer_cv_rejects_missing_or_undefined_evidence_without_replacing_session_state(case):
    session = _session()
    prior = _automl(session, metric="mse", outer_cv=3)
    assert np.isfinite(prior.outer_score_mean)
    plan, fitted, history = session.automl_plan, session.fit_result, session.history
    if case == "empty":
        folds, message = [], "outer CV produced no folds"
    else:
        folds = [(np.arange(47), np.array([47]))]
        message = "outer fold 0 has no finite 'r2' score"
    with pytest.raises(ValidationError, match=message):
        _automl(session, metric="r2", outer_cv=folds)
    assert session.automl_plan is plan
    assert session.fit_result is fitted
    assert session.history == history


def test_one_row_outer_fold_retains_a_defined_weighted_loss():
    session = _session()
    train = session.partition("train")
    fit, evaluation = np.arange(47), np.array([47])
    expected = _mse(train.iloc[fit], train.iloc[evaluation])
    result = _automl(session, metric="mse", outer_cv=[(fit, evaluation)])
    assert result.outer_score_mean == pytest.approx(expected, rel=1e-10)
    assert result.outer_score_std == 0.0
    assert json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    "splitter",
    (
        GroupKFold(3),
        GroupShuffleSplit(n_splits=3, test_size=2, random_state=7),
        LeaveOneGroupOut(),
        LeavePGroupsOut(n_groups=2),
    ),
    ids=("group-kfold", "group-shuffle", "leave-one-group", "leave-two-groups"),
)
def test_explicit_regression_group_splitters_use_the_group_role(splitter):
    session = _session()
    train = session.partition("train")
    expected = []
    for fit, evaluate in splitter.split(train, train.y, groups=train.group):
        assert set(train.group.iloc[fit]).isdisjoint(train.group.iloc[evaluate])
        expected.append(_mse(train.iloc[fit], train.iloc[evaluate]))
    observed = session.cv_score(
        Ridge(alpha=0.1), task="regression", cv=splitter, scoring_metric="mse"
    )
    assert [fold.metrics["mse"] for fold in observed.folds] == pytest.approx(expected, rel=1e-10)


def test_explicit_stratified_group_splitter_uses_the_group_role():
    session = _session(classification=True)
    train = session.partition("train")
    splitter = StratifiedGroupKFold(3, shuffle=True, random_state=23)
    expected = []
    for fit, evaluate in splitter.split(train, train.y, groups=train.group):
        assert set(train.group.iloc[fit]).isdisjoint(train.group.iloc[evaluate])
        fit_rows, score_rows = train.iloc[fit], train.iloc[evaluate]
        label = int(np.argmax(np.bincount(fit_rows.y, weights=fit_rows.weight)))
        expected.append(np.average(score_rows.y == label, weights=score_rows.weight))
    observed = session.cv_score(
        DummyClassifier(strategy="most_frequent"),
        task="classification",
        cv=splitter,
        scoring_metric="accuracy",
    )
    assert [fold.metrics["accuracy"] for fold in observed.folds] == pytest.approx(expected)


def test_native_nested_automl_uses_the_role_for_explicit_outer_and_inner_group_cv():
    session = _session()
    train = session.partition("train")
    outer = GroupKFold(3)
    inner = GroupKFold(2)
    alphas = (0.1, 1.0, 10.0)
    expected = []
    for fit, evaluate in outer.split(train, train.y, groups=train.group):
        outer_train = train.iloc[fit]
        inner_folds = list(inner.split(outer_train, outer_train.y, groups=outer_train.group))
        scores = [
            np.mean(
                [
                    _mse(outer_train.iloc[fit_inner], outer_train.iloc[eval_inner], alpha)
                    for fit_inner, eval_inner in inner_folds
                ]
            )
            for alpha in alphas
        ]
        winner = alphas[np.argmin(scores)]
        expected.append(_mse(outer_train, train.iloc[evaluate], winner))
    result = _automl(session, metric="mse", outer_cv=outer, cv=inner, n_trials=len(alphas))
    assert result.outer_score_mean == pytest.approx(np.mean(expected), rel=1e-10)
    assert result.outer_score_std == pytest.approx(np.std(expected, ddof=1), rel=1e-10)


def test_explicit_groups_override_the_role_without_reordering_rows():
    session = _session()
    train = session.partition("train")
    labels = np.repeat(np.arange(12), 4)
    splitter = GroupKFold(3)
    expected = [
        _mse(train.iloc[fit], train.iloc[evaluate])
        for fit, evaluate in splitter.split(train, train.y, labels)
    ]
    result = session.cv_score(
        Ridge(alpha=0.1),
        task="regression",
        cv=splitter,
        scoring_metric="mse",
        groups=pd.Series(labels),
    )
    assert [fold.metrics["mse"] for fold in result.folds] == pytest.approx(expected, rel=1e-10)
