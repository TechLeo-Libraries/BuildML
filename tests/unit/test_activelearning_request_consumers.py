"""Explicit active-learning requests preserve policy and real saved query consumers."""

import copy
import hashlib

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from buildml import Session
from buildml.activelearning.checkpoint import load_active_learning_bundle
from buildml.activelearning.query import suggest_query
from buildml.core.errors import ValidationError

BACKINGS = ("pandas", "pandas-parquet", "polars-lazy", "duckdb-lazy")
STRATEGIES = ("least_confidence", "margin", "entropy")
INVALID = ("typo", "", "auto", "Entropy", "margin ", None, False, 0)
INVALID_IDS = ("typo", "empty", "auto", "case", "space", "none", "false", "zero")


def _pandas_session(frame):
    return Session.ingest(frame).set_roles({"x": "feature", "z": "feature", "target": "target"}).inject_split(
        train_indices=list(range(60)), validation_indices=list(range(60, 72)), test_indices=list(range(72, 84)))


def _session(backing, directory, poison=False):
    rng = np.random.default_rng(910)
    x = rng.normal(size=(84, 2))
    truth = np.where(x[:, 0]+.6*x[:, 1]+rng.normal(scale=.6, size=84) > 0, "yes", "no")
    labels = truth.astype(object)
    labels[np.arange(2, 60, 4)] = None
    if poison:
        x[60:] = 10000-rng.normal(size=(24, 2))
        labels[60:] = "holdout-only-class"
    frame = pd.DataFrame(dict(x=x[:, 0], z=x[:, 1], target=labels))
    engine = backing.split("-")[0]
    if engine != "pandas":
        pytest.importorskip(engine)
    if backing == "pandas":
        session = _pandas_session(frame)
    else:
        path = directory / "pool.parquet"
        frame.to_parquet(path, index=False)
        session = Session.ingest(path, engine=engine, mode="memory" if engine == "pandas" else "lazy")
        session.set_roles({"x": "feature", "z": "feature", "target": "target"}).inject_split(
            train_indices=list(range(60)), validation_indices=list(range(60, 72)), test_indices=list(range(72, 84)))
    return session, truth


def _fit(session, strategy="entropy"):
    return session.active_learning.fit(strategy=strategy, random_state=13, label_budget=4,
        epochs=1, mc_samples=3, device="cpu")


def _assert_query(session, strategy):
    plan = session.activelearning_plan
    assert plan.backend == "sklearn" and plan.strategy == strategy
    train = session.to_pandas().iloc[:60]
    labeled = train.target.notna().to_numpy()
    x = train[["x", "z"]].to_numpy()
    design = np.column_stack([x[labeled], np.ones(labeled.sum())])
    truth = (train.target[labeled] == "yes").to_numpy(dtype=float)
    # Independent Newton solve for sum logistic loss + L2, with an unpenalized intercept.
    coefficient, penalty = np.zeros(3), np.diag([1., 1., 0.])
    for _ in range(40):
        p = expit(design@coefficient)
        gradient = design.T@(p-truth)+penalty@coefficient
        hessian = design.T@(design*(p*(1-p))[:, None])+penalty
        step = np.linalg.solve(hessian, gradient)
        coefficient -= step
        if np.max(np.abs(step)) < 1e-12:
            break
    actual = np.r_[plan.estimator_.coef_.ravel(), plan.estimator_.intercept_]
    np.testing.assert_allclose(actual, coefficient, rtol=1e-3, atol=1e-3)
    p = expit(np.column_stack([x[~labeled], np.ones((~labeled).sum())])@actual)
    if strategy == "least_confidence":
        scores = 1-np.maximum(p, 1-p)
    elif strategy == "margin":
        scores = -np.abs(2*p-1)
    else:
        scores = -p*np.log(p)-(1-p)*np.log1p(-p)
    pool = np.flatnonzero(~labeled)
    remaining = 4-plan.n_queries_used
    chosen = np.argsort(-scores, kind="stable")[:min(3, remaining)]
    result = session.active_learning.suggest_query(batch_size=3)
    assert result.indices == tuple(pool[chosen])
    np.testing.assert_allclose(result.scores, scores[chosen], rtol=1e-11, atol=1e-11)
    assert result.strategy == strategy and result.budget_remaining == remaining
    assert result.n_unlabeled_pool == len(pool)
    assert result.batch_size_requested == min(3, remaining)
    assert all(index < 60 for index in result.indices)
    return result


def _snapshot(session):
    frame = session.to_pandas().copy(deep=True)
    return (session.activelearning_plan, joblib.hash(session.activelearning_plan),
        session._activelearning_fit_result, session._activelearning_query_result,
        copy.deepcopy(session.history), frame, copy.deepcopy(session.split_plan))


def _unchanged(session, before):
    plan, fingerprint, fitted, query, history, frame, split = before
    assert session.history == history
    assert session.activelearning_plan is plan and joblib.hash(plan) == fingerprint
    assert session._activelearning_fit_result is fitted and session._activelearning_query_result is query
    assert session.split_plan == split
    pd.testing.assert_frame_equal(session.to_pandas(), frame)


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("poison", (False, True))
def test_named_query_label_refit_and_saved_consumers_match_independent_equations(backing, strategy, poison, tmp_path):
    session, truth = _session(backing, tmp_path, poison)
    _fit(session, strategy)
    first = _assert_query(session, strategy)
    frame = session.to_pandas().copy(deep=True)
    selected = list(first.indices[:2])
    session.active_learning.label_rows(indices=selected, labels=truth[selected].tolist(), refit=True)
    frame.loc[selected, "target"] = truth[selected]
    pd.testing.assert_frame_equal(session.to_pandas(), frame)
    assert session.activelearning_plan.n_queries_used == 2
    expected = _assert_query(session, strategy)
    path = session.active_learning.save_bundle(tmp_path / "trusted-active-plan")
    loaded = load_active_learning_bundle(path, trusted=True)
    fingerprint = joblib.hash(loaded)
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file()}
    replay = suggest_query(session.dataset, loaded, session.split_plan, batch_size=3)
    assert replay.indices == expected.indices
    np.testing.assert_allclose(replay.scores, expected.scores, rtol=1e-12, atol=1e-12)
    receiver = _pandas_session(frame)
    receiver.active_learning.load_bundle(path, trusted=True)
    replay = _assert_query(receiver, strategy)
    assert replay.indices == expected.indices
    np.testing.assert_allclose(replay.scores, expected.scores, rtol=1e-12, atol=1e-12)
    assert joblib.hash(loaded) == fingerprint
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file()} == files


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("existing", (False, True))
@pytest.mark.parametrize("strategy", INVALID, ids=INVALID_IDS)
def test_invalid_fit_requests_do_not_select_another_strategy_or_replace_a_prior_learner(backing, existing, strategy, tmp_path):
    session, _ = _session(backing, tmp_path)
    if existing:
        _fit(session)
        _assert_query(session, "entropy")
    before = _snapshot(session)
    with pytest.raises(ValidationError, match="strategy"):
        _fit(session, strategy)
    _unchanged(session, before)


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("strategy", INVALID[:5]+INVALID[6:], ids=INVALID_IDS[:5]+INVALID_IDS[6:])
def test_invalid_query_strategy_is_not_treated_as_an_omitted_override(backing, strategy, tmp_path):
    session, _ = _session(backing, tmp_path)
    _fit(session)
    _assert_query(session, "entropy")
    before = _snapshot(session)
    with pytest.raises(ValidationError, match="strategy"):
        session.active_learning.suggest_query(batch_size=3, strategy=strategy)
    _unchanged(session, before)


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("backend", ("", False, 0, "typo"), ids=("empty", "false", "zero", "typo"))
def test_invalid_direct_query_backend_is_not_treated_as_the_saved_backend(backing, backend, tmp_path):
    session, _ = _session(backing, tmp_path)
    _fit(session)
    _assert_query(session, "entropy")
    before = _snapshot(session)
    with pytest.raises(ValidationError, match="backend"):
        suggest_query(session.dataset, session.activelearning_plan, session.split_plan,
            batch_size=3, backend=backend)
    _unchanged(session, before)
