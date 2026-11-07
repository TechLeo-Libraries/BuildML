"""Categorical train catalogs preserve observed pairs, scores, and saved memory."""

import json

import numpy as np
import pandas as pd
import pytest
from test_recommender_numeric_memory_consumers import (
    fit,
    frame,
    raw_memory,
    raw_scores,
    ready,
    state,
    unchanged,
)

from buildml import Session
from buildml.core.errors import ValidationError

METHODS = ["item_knn", "user_knn", "svd", "content"]


def categorical_session(kind="both", ordered=False, small=None, null_ids=False):
    raw = frame()
    base = ready(raw)
    split = base.split_plan
    for col, count in [("user", 6), ("item", 7)]:
        if small in {col, "both"}:
            raw.loc[raw.index[list(split.train_indices)], col] = 0
        if null_ids:
            raw.loc[raw.index[list(split.train_indices)], col] = np.nan
        if kind in {col, "both"}:
            # Reverse declared order deliberately; levels 99 are holdout-only,
            # and levels 100/101 never occur anywhere in the input rows.
            raw[col] = pd.Categorical(
                raw[col],
                categories=list(reversed(range(count))) + [101, 99, 100],
                ordered=ordered,
            )
        elif kind == "object":
            raw[col] = raw[col].astype(object)
    return (
        Session.ingest(raw)
        .set_roles(
            {
                "user": "id",
                "item": "id",
                "rating": "target",
                "a": "feature",
                "b": "feature",
                "c": "feature",
            }
        )
        .inject_split(
            train_indices=split.train_indices,
            validation_indices=split.validation_indices,
            test_indices=split.test_indices,
        )
    )


def assert_rankings(owner, matrix, observed, users, items, expected):
    for exclude in [False, True]:
        result = owner.recommender.recommend(
            user_ids=[0, 4, 99, 100], k=99, exclude_train_items=exclude
        )
        assert result.cold_start_users == (99, 100)
        for user, recommended, scores in zip(
            result.user_ids, result.recommendations, result.scores
        ):
            eligible = [
                item
                for item in items
                if user not in users
                or not exclude
                or not observed[users.index(user), items.index(item)]
            ]
            assert set(recommended) == set(eligible)
            assert len(recommended) == len(eligible)
            wanted = observed.sum(axis=0) if user not in users else expected[users.index(user)]
            np.testing.assert_allclose(
                scores,
                [wanted[items.index(item)] for item in recommended],
                rtol=2e-9,
                atol=2e-9,
            )
            assert all(a >= b - 1e-12 for a, b in zip(scores, scores[1:]))
        json.dumps(result.to_dict(), allow_nan=False)

    # Raw row iteration deduplicates holdout pairs independently of groupby.
    holdout = set((r.user, r.item) for r in owner.partition("test").itertuples())
    relevant = {}
    for user, item in holdout:
        if user in users and item in items:
            relevant.setdefault(user, set()).add(item)
    evaluation = owner.recommender.evaluate(partition="test", k=99)
    assert evaluation.n_holdout_interactions == len(holdout)
    assert evaluation.n_users_scored == len(relevant)
    assert evaluation.n_cold_start_users == 1
    assert evaluation.metrics["recall_at_k"] == 1.0
    assert evaluation.metrics["precision_at_k"] == pytest.approx(
        np.mean([len(v) / 99 for v in relevant.values()])
    )


def check_saved_arithmetic(s, method, feedback, tmp_path):
    original = s.dataset.to_pandas().copy(deep=True)
    fit(s, method, feedback=feedback, n_neighbors=3, n_factors=2)
    plan = s.recommender_plan
    train = s.partition("train")
    users, items = list(plan.user_ids), list(plan.item_ids)
    assert set(users) == set(train.user) == set(range(6))
    assert set(items) == set(train.item) == set(range(7))
    matrix, observed, _, _ = raw_memory(train, feedback, users, items)
    np.testing.assert_allclose(plan.matrix_, matrix)
    np.testing.assert_array_equal(plan.observed_mask_, observed)
    np.testing.assert_array_equal(plan.item_popularity_, observed.sum(axis=0))
    assert plan.user_index_ == {u: i for i, u in enumerate(users)}
    assert plan.item_index_ == {v: i for i, v in enumerate(items)}
    # Build side-feature means by raw row arithmetic, independent of dtype/order.
    features = np.asarray(
        [
            np.mean(
                [[r.a, r.b, r.c] for r in train.itertuples() if r.item == item],
                axis=0,
            )
            for item in items
        ]
    )
    mean, scale = features.mean(axis=0), features.std(axis=0)
    features = (features - mean) / scale
    if method == "content":
        np.testing.assert_allclose(plan.item_feature_mean_, mean)
        np.testing.assert_allclose(plan.item_feature_scale_, scale)
        np.testing.assert_allclose(plan.item_features_, features)
    expected = raw_scores(method, matrix, observed, features, 3, 2, feedback)
    saved = s.recommender.save_bundle(tmp_path / "categorical")
    restored = categorical_session()
    restored.recommender.load_bundle(saved, trusted=True)
    for owner in [s, restored]:
        assert_rankings(owner, matrix, observed, users, items, expected)
    pd.testing.assert_frame_equal(s.dataset.to_pandas(), original)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("feedback", ["explicit", "implicit"])
@pytest.mark.parametrize("kind", ["user", "item", "both"])
@pytest.mark.parametrize("ordered", [False, True])
def test_categorical_rankings_and_bundles_follow_only_observed_train_pairs(
    method, feedback, kind, ordered, tmp_path
):
    check_saved_arithmetic(categorical_session(kind, ordered), method, feedback, tmp_path)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("feedback", ["explicit", "implicit"])
def test_object_ids_retain_the_same_saved_numerical_protocol(method, feedback, tmp_path):
    check_saved_arithmetic(categorical_session("object"), method, feedback, tmp_path)


@pytest.fixture
def prior_bundle(tmp_path):
    s = ready()
    fit(s, n_neighbors=3, n_factors=2)
    return s.recommender.save_bundle(tmp_path / "prior")


@pytest.mark.parametrize("feedback", ["explicit", "implicit"])
@pytest.mark.parametrize("small", ["user", "item", "both"])
def test_unused_levels_cannot_inflate_a_too_small_training_population(
    feedback, small, prior_bundle
):
    s = categorical_session(small=small)
    s.recommender.load_bundle(prior_bundle, trusted=True)
    before = state(s)
    n_users, n_items = (
        (1 if small in {"user", "both"} else 6),
        (1 if small in {"item", "both"} else 7),
    )
    with pytest.raises(ValidationError, match=f"got {n_users} users / {n_items} items"):
        fit(s, feedback=feedback)
    unchanged(s, before)


def test_unused_levels_cannot_rescue_all_null_train_ids(prior_bundle):
    s = categorical_session(null_ids=True)
    s.recommender.load_bundle(prior_bundle, trusted=True)
    before = state(s)
    with pytest.raises(ValidationError, match="dropping null user/item"):
        fit(s)
    unchanged(s, before)


def test_real_rating_floor_can_remove_the_entire_categorical_population(prior_bundle):
    s = categorical_session()
    s.recommender.load_bundle(prior_bundle, trusted=True)
    before = state(s)
    with pytest.raises(ValidationError, match="rating filters / null drops"):
        fit(s, min_rating=999)
    unchanged(s, before)
