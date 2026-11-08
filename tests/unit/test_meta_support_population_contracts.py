"""Adaptation reports and learns from one explicit support population."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.metalearning.adapt import adapt_to_task


def _fitted(method):
    rng = np.random.default_rng(512)
    n = 432
    labels = np.tile(["alpha", "beta", "gamma"], n // 3)
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "z": rng.normal(size=n),
            "label": labels,
            "task": np.repeat(np.arange(6), 72),
        }
    )
    session = Session.ingest(frame).set_roles(
        {"x": "feature", "z": "feature", "label": "target", "task": "group"}
    )
    session.inject_split(
        train_indices=list(range(216)),
        validation_indices=list(range(216, 288)),
        test_indices=list(range(288, 432)),
    )
    session.metalearning.fit(
        backend="sklearn",
        method=method,
        k_shot=2,
        n_query=6,
        n_way=3,
        n_episodes=3,
        random_state=21,
    )
    return session


def _support(index_kind="offset"):
    frame = pd.DataFrame(
        {
            "x": np.arange(12, dtype=float),
            "z": np.arange(12, dtype=float) ** 2,
            "label": ["alpha", "gamma"] * 6,
            "task": ["novel"] * 12,
        }
    )
    if index_kind == "offset":
        frame.index = np.arange(9000, 9012)
    elif index_kind == "duplicates":
        frame.index = np.arange(12) // 2
    else:
        frame.index = np.random.default_rng(19).permutation(12)
    return frame


@pytest.mark.parametrize("method", ["prototypical", "warm_start"])
@pytest.mark.parametrize("cap", [None, 2])
@pytest.mark.parametrize("index_kind", ["offset", "permuted", "duplicates"])
def test_adaptation_matches_support_oracles_and_reports_only_present_classes(
    method, cap, index_kind
):
    session = _fitted(method)
    plan = session.metalearning_plan
    support = _support(index_kind)
    original, fingerprint = support.copy(), joblib.hash(plan)
    actual = adapt_to_task(
        session.dataset,
        plan,
        session.split_plan,
        support_frame=support,
        max_support_per_class=cap,
        random_state=7,
    )
    expected = support.copy()
    if cap is not None:
        rng = np.random.default_rng(7)
        expected = pd.concat(
            [
                group.iloc[rng.choice(len(group), cap, replace=False)]
                for _, group in support.groupby("label", sort=False)
            ]
        )
    assert actual.task_id == "novel" and actual.n_support == len(expected)
    assert actual.n_classes_adapted == 2
    assert actual.classes_ == ("alpha", "gamma")
    if method == "prototypical":
        for label, group in expected.groupby("label"):
            np.testing.assert_allclose(
                actual.prototypes_[label], group[["x", "z"]].mean().to_numpy()
            )
        assert set(actual.prototypes_) == {"alpha", "gamma"}
    else:
        oracle = clone(plan.init_estimator_)
        oracle.fit(expected[["x", "z"]].to_numpy(), plan.label_encoder_.transform(expected.label))
        probe = support[["x", "z"]].to_numpy()
        np.testing.assert_allclose(
            actual.adapted_estimator_.predict_proba(probe), oracle.predict_proba(probe)
        )
        assert actual.adapted_estimator_ is not plan.init_estimator_
    assert joblib.hash(plan) == fingerprint
    pd.testing.assert_frame_equal(support, original)


@pytest.mark.parametrize("method", ["prototypical", "warm_start"])
@pytest.mark.parametrize("cap", [None, 2])
def test_support_cap_cannot_silently_remove_missing_target_rows(method, cap):
    session = _fitted(method)
    support = _support()
    support.iloc[0, support.columns.get_loc("label")] = None
    plan = session.metalearning_plan
    fingerprint = joblib.hash(plan)
    with pytest.raises(ValidationError, match="null|missing"):
        adapt_to_task(
            session.dataset,
            plan,
            session.split_plan,
            support_frame=support,
            max_support_per_class=cap,
        )
    assert joblib.hash(plan) == fingerprint


@pytest.mark.parametrize("method", ["prototypical", "warm_start"])
@pytest.mark.parametrize("identity", ["mixed", "conflict", "missing"])
def test_explicit_support_refuses_ambiguous_or_contradictory_task_identity(method, identity):
    session = _fitted(method)
    support = _support()
    task_id = None
    if identity == "mixed":
        support.iloc[0, support.columns.get_loc("task")] = "another-task"
    elif identity == "conflict":
        task_id = "different-declared-task"
    else:
        support.iloc[0, support.columns.get_loc("task")] = None
    plan = session.metalearning_plan
    fingerprint = joblib.hash(plan)
    with pytest.raises(ValidationError, match="task"):
        adapt_to_task(
            session.dataset, plan, session.split_plan, support_frame=support, task_id=task_id
        )
    assert joblib.hash(plan) == fingerprint
