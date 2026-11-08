"""User annotations select row positions and commit only after successful refit."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.activelearning.label import label_rows
from buildml.activelearning.query import suggest_query
from buildml.core.errors import ValidationError


def _ready(index_kind="range", *, marker=None, strategy="entropy"):
    rng = np.random.default_rng(821)
    truth = np.where(np.arange(180) % 2, "accepted", "rejected")
    y = truth.astype(object)
    y[np.arange(0, 120, 3)] = np.nan if marker is None else marker
    frame = pd.DataFrame({"x": rng.normal(size=180), "z": rng.normal(size=180), "y": y})
    if index_kind == "offset":
        frame.index = np.arange(8000, 8180)
    elif index_kind == "permuted":
        frame.index = rng.permutation(len(frame))
    elif index_kind == "duplicates":
        frame.index = np.arange(len(frame)) // 2
    session = Session.ingest(frame).set_roles({"x": "feature", "z": "feature", "y": "target"})
    session.inject_split(
        train_indices=list(range(120)),
        validation_indices=list(range(120, 150)),
        test_indices=list(range(150, 180)),
    )
    session.active_learning.fit(
        backend="sklearn",
        strategy=strategy,
        committee_size=3,
        label_budget=12,
        random_state=31,
        unlabeled_marker=marker,
        auto_refit=False,
    )
    return session, truth


@pytest.mark.parametrize("index_kind", ["range", "offset", "permuted", "duplicates"])
@pytest.mark.parametrize("refit", [False, True])
def test_annotations_use_split_positions_and_leave_original_inputs_unchanged(index_kind, refit):
    session, truth = _ready(index_kind)
    original = session.to_pandas().copy()
    plan = session.activelearning_plan
    fingerprint = joblib.hash(plan)
    chosen = list(plan.unlabeled_pool_indices[:3])
    dataset, updated, result, fitted = label_rows(
        session.dataset,
        plan,
        session.split_plan,
        indices=chosen,
        labels=truth[chosen].tolist(),
        refit=refit,
    )
    expected = original.copy()
    expected.iloc[chosen, expected.columns.get_loc("y")] = truth[chosen]
    pd.testing.assert_frame_equal(dataset._ensure_pandas(), expected)
    pd.testing.assert_frame_equal(session.to_pandas(), original)
    assert joblib.hash(plan) == fingerprint
    assert updated is not plan
    assert updated.n_queries_used == 3 and result.budget_remaining == 9
    assert result.n_newly_labeled == 3 and updated.n_unlabeled_pool == plan.n_unlabeled_pool - 3
    assert set(chosen).isdisjoint(updated.unlabeled_pool_indices)
    assert (fitted is not None) == refit


@pytest.mark.parametrize("invalid", [None, np.nan, pd.NA, np.datetime64("NaT", "ns"), "?"])
def test_missing_or_marker_annotations_are_refused_without_consuming_budget(invalid):
    session, _ = _ready(marker="?")
    plan = session.activelearning_plan
    original, fingerprint = session.to_pandas().copy(), joblib.hash(plan)
    with pytest.raises(ValidationError, match="label"):
        session.active_learning.label_rows(
            indices=[plan.unlabeled_pool_indices[0]], labels=[invalid], refit=False
        )
    pd.testing.assert_frame_equal(session.to_pandas(), original)
    assert joblib.hash(plan) == fingerprint


@pytest.mark.parametrize("labels", [["accepted", "accepted"], ["accepted", "rejected"]])
def test_repeated_row_positions_cannot_double_charge_or_ambiguously_overwrite(labels):
    session, _ = _ready()
    plan = session.activelearning_plan
    original, fingerprint = session.to_pandas().copy(), joblib.hash(plan)
    index = plan.unlabeled_pool_indices[0]
    with pytest.raises(ValidationError, match="duplicate|unique|repeat"):
        session.active_learning.label_rows(indices=[index, index], labels=labels, refit=False)
    pd.testing.assert_frame_equal(session.to_pandas(), original)
    assert joblib.hash(plan) == fingerprint


@pytest.mark.parametrize("failure_target", ["estimator", "committee"])
def test_failed_refit_cannot_consume_query_budget_or_publish_partial_history(
    monkeypatch, failure_target
):
    session, _ = _ready(strategy="committee" if failure_target == "committee" else "entropy")
    plan = session.activelearning_plan
    original, fingerprint = session.to_pandas().copy(), joblib.hash(plan)

    def fail_fit(*args, **kwargs):
        raise RuntimeError("deliberate estimator refit failure")

    target = plan.estimator_ if failure_target == "estimator" else plan.committee_
    monkeypatch.setattr(type(target), "fit", fail_fit)
    with pytest.raises(ValidationError, match="deliberate estimator refit failure"):
        session.active_learning.label_rows(
            indices=[plan.unlabeled_pool_indices[0]], labels=["previously-unseen-class"], refit=True
        )
    pd.testing.assert_frame_equal(session.to_pandas(), original)
    assert session.activelearning_plan is plan
    assert joblib.hash(plan) == fingerprint


@pytest.mark.parametrize("index", [True, 0.0, "0"])
def test_noninteger_annotation_positions_are_refused_without_state_changes(index):
    session, _ = _ready()
    plan = session.activelearning_plan
    original, fingerprint = session.to_pandas().copy(), joblib.hash(plan)
    with pytest.raises(ValidationError, match="integer row positions|train partition"):
        session.active_learning.label_rows(indices=[index], labels=["accepted"], refit=False)
    pd.testing.assert_frame_equal(session.to_pandas(), original)
    assert joblib.hash(plan) == fingerprint


def test_successful_refit_can_extend_classes_without_mutating_prior_vocabulary():
    session, _ = _ready()
    plan = session.activelearning_plan
    fingerprint = joblib.hash(plan)
    session.active_learning.label_rows(
        indices=[plan.unlabeled_pool_indices[0]], labels=["previously-unseen-class"], refit=True
    )
    assert "previously-unseen-class" in session.activelearning_plan.classes_
    assert "previously-unseen-class" not in plan.classes_
    assert joblib.hash(plan) == fingerprint


@pytest.mark.parametrize("index_kind", ["offset", "permuted"])
@pytest.mark.parametrize(
    "strategy", ["least_confidence", "margin", "entropy", "committee", "expected_model_change_lite"]
)
def test_query_scores_match_probability_oracles_and_ignore_holdout_features(index_kind, strategy):
    session, _ = _ready(index_kind, strategy=strategy)
    plan = session.activelearning_plan
    original, fingerprint = session.to_pandas().copy(), joblib.hash(plan)
    positions = list(plan.unlabeled_pool_indices)
    x = original.iloc[positions][list(plan.columns)].to_numpy()
    if strategy == "committee":
        votes = np.array([member.predict(x) for member in plan.committee_.estimators_])
        fractions = np.array([(votes == label).mean(axis=0) for label in np.unique(votes)])
        positive = np.where(fractions > 0, fractions, 1)
        expected_scores = -(fractions * np.log(positive)).sum(axis=0)
    else:
        probabilities = plan.estimator_.predict_proba(x)
        maximum = probabilities.max(axis=1)
        if strategy == "least_confidence":
            expected_scores = 1 - maximum
        elif strategy == "margin":
            ordered = np.sort(probabilities, axis=1)
            expected_scores = ordered[:, -2] - ordered[:, -1]
        elif strategy == "entropy":
            expected_scores = -(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum(
                axis=1
            )
        else:
            expected_scores = np.sqrt((x**2).sum(axis=1)) * (1 - maximum)
    result = suggest_query(session.dataset, plan, session.split_plan, batch_size=20)
    assert len(result.indices) == 12  # budget caps the requested batch
    assert len(set(result.indices)) == 12
    assert set(result.indices) <= set(positions)
    score_by_position = dict(zip(positions, expected_scores))
    np.testing.assert_allclose(
        result.scores, [score_by_position[p] for p in result.indices], atol=1e-12
    )
    assert min(result.scores) >= np.sort(expected_scores)[-12] - 1e-12
    changed = original.copy()
    holdout = [*session.split_plan.test_indices, *session.split_plan.validation_indices]
    changed.iloc[holdout, changed.columns.get_indexer(["x", "z"])] = 1e6
    from buildml.data.dataset import Dataset

    after = suggest_query(Dataset.from_pandas(changed), plan, session.split_plan, batch_size=20)
    assert after.indices == result.indices
    np.testing.assert_allclose(after.scores, result.scores, rtol=0, atol=0)
    assert joblib.hash(plan) == fingerprint
    pd.testing.assert_frame_equal(session.to_pandas(), original)
