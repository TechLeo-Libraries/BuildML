"""Observed zero ratings remain history through scoring and saved consumers."""

import copy
import itertools

import joblib
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.recommenders.checkpoint import load_recommender_bundle, save_recommender_bundle
from buildml.recommenders.recommend import recommend_for_users

METHODS = ["item_knn", "user_knn", "svd", "nmf", "content"]


def _data(layout):
    rows = [(0, 0, 0), (0, 1, 2), (0, 2, 4), (1, 0, 3), (1, 3, 0), (1, 4, 4),
            (2, 1, 0), (2, 2, 5), (2, 5, 1), (3, 0, 0), (3, 4, 0), (3, 5, 0)]
    if layout == "cancelling":
        rows = [entry for u, i, r in rows for entry in
                ([(u, i, -1), (u, i, 1)] if r == 0 else [(u, i, r)])]
    elif layout == "positive":
        rows = [(u, i, r+1) for u, i, r in rows]
    train_count = len(rows)
    held = [(0, 3, 5), (0, 5, 4), (1, 2, 3), (2, 4, 4), (3, 1, 2),
            (4, 0, 5), (0, 6, 2)]
    all_rows = rows + held + held
    frame = pd.DataFrame([dict(user=f"u{u}", item=f"i{i}", rating=float(r),
                               a=float(i), b=float(i*i % 7)) for u, i, r in all_rows],
                         index=np.arange(len(all_rows)) % 5)
    return frame, train_count, len(held)


def _session(frame, train_count, held_count):
    session = Session.ingest(frame).set_roles({"user": "id", "item": "id", "rating": "target",
                                               "a": "feature", "b": "feature"})
    session.inject_split(train_indices=list(range(train_count)),
                         validation_indices=list(range(train_count, train_count+held_count)),
                         test_indices=list(range(train_count+held_count, len(frame))))
    return session


def _fit_data(method, layout):
    frame, n, h = _data(layout)
    session = _session(frame, n, h)
    session.recommender.fit(backend="sklearn", method=method, user_column="user",
                            item_column="item", rating_column="rating", feedback="explicit",
                            item_feature_columns=["a", "b"], n_neighbors=99, n_factors=2,
                            random_state=7)
    return session, frame, n, h, method


@pytest.fixture(scope="module", params=list(itertools.product(METHODS, ["zero", "cancelling", "positive"])))
def fitted(request):
    return _fit_data(*request.param)


def _arrays(frame, n, plan):
    observed = np.zeros((plan.n_users, plan.n_items), dtype=bool)
    matrix = np.zeros(observed.shape)
    # Independent aggregation by scalar user/item identity, including zero means.
    grouped = {}
    for row in frame.iloc[:n].itertuples():
        grouped.setdefault((row.user, row.item), []).append(row.rating)
    for (user, item), ratings in grouped.items():
        u, i = plan.user_index_[user], plan.item_index_[item]
        observed[u, i] = True
        matrix[u, i] = sum(ratings)/len(ratings)
    return matrix, observed


def _scores(plan, matrix, observed, user):
    if user == "cold":
        return observed.sum(axis=0).astype(float)
    u = plan.user_index_[user]
    if plan.method == "svd":
        centered = np.where(observed, matrix-matrix[observed].mean(), 0)
        left, singular, right = np.linalg.svd(centered, full_matrices=False)
        return (left[u, :2]*singular[:2])@right[:2]+matrix[observed].mean()
    if plan.method == "nmf":
        return plan.user_factors_[u]@plan.item_factors_.T
    if plan.method == "content":
        weights = np.abs(matrix[u])
        if weights.sum() == 0:
            return np.zeros(plan.n_items)
        profile = weights@plan.item_features_/weights.sum()
        norm = np.linalg.norm(profile)
        if norm < 1e-12:
            return np.zeros(plan.n_items)
        return (plan.item_features_@profile)/(np.maximum(np.linalg.norm(plan.item_features_, axis=1), 1e-12)*norm)
    vectors = matrix.T if plan.method == "item_knn" else matrix
    norms = np.linalg.norm(vectors, axis=1)
    similarity = vectors@vectors.T/np.maximum(norms[:, None]*norms[None, :], 1e-300)
    scores = np.zeros(plan.n_items)
    for i in range(plan.n_items):
        if plan.method == "item_knn":
            weights = similarity[i, observed[u]]
            ratings = matrix[u, observed[u]]
        else:
            keep = observed[:, i].copy()
            keep[u] = False
            weights, ratings = similarity[u, keep], matrix[keep, i]
        denom = sum(abs(value) for value in weights)
        scores[i] = sum(a*b for a, b in zip(weights, ratings, strict=True))/denom if denom >= 1e-12 else 0
    return scores


@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("exclude", [False, True])
def test_zero_rated_history_scores_popularity_and_replay_match_observation_oracles(fitted, replay, exclude, tmp_path):
    original, frame, n, h, method = fitted
    session = original
    if replay:
        original.recommender.save_bundle(tmp_path/"policy")
        session = _session(frame, n, h).recommender.load_bundle(tmp_path/"policy", trusted=True)
    plan = session.recommender_plan
    matrix, observed = _arrays(frame, n, plan)
    before = joblib.hash(plan)
    users = [*plan.user_ids, "cold"]
    result = session.recommender.recommend(user_ids=users, k=99, exclude_train_items=exclude)
    for user, items, values in zip(result.user_ids, result.recommendations, result.scores, strict=True):
        scores = _scores(plan, matrix, observed, user)
        eligible = np.ones(plan.n_items, dtype=bool)
        if exclude and user != "cold":
            eligible &= ~observed[plan.user_index_[user]]
        expected_items = {item for i, item in enumerate(plan.item_ids) if eligible[i]}
        assert set(items) == expected_items
        np.testing.assert_allclose(values, [scores[plan.item_index_[item]] for item in items], atol=2e-10)
        assert list(values) == sorted(values, reverse=True)
    np.testing.assert_array_equal(plan.matrix_, matrix)
    assert plan.global_mean_ == pytest.approx(matrix[observed].mean())
    np.testing.assert_array_equal(plan.item_popularity_, observed.sum(axis=0))
    assert joblib.hash(plan) == before


@pytest.mark.parametrize("partition", ["validation", "test"])
@pytest.mark.parametrize("replay", [False, True])
def test_known_item_ranking_metrics_use_zero_rated_history(fitted, partition, replay, tmp_path):
    original, frame, n, h, method = fitted
    session = original
    if replay:
        original.recommender.save_bundle(tmp_path/"policy")
        session = _session(frame, n, h).recommender.load_bundle(tmp_path/"policy", trusted=True)
    plan = session.recommender_plan
    matrix, observed = _arrays(frame, n, plan)
    expected = []
    truth = {"u0": {"i3", "i5"}, "u1": {"i2"}, "u2": {"i4"}, "u3": {"i1"}}
    recommendations = session.recommender.recommend(user_ids=plan.user_ids, k=2, exclude_train_items=True)
    for user, ranked in zip(recommendations.user_ids, recommendations.recommendations, strict=True):
        hits = [item in truth[user] for item in ranked]
        precision = sum(hits)/2
        recall = sum(hits)/len(truth[user])
        dcg = sum(hit/np.log2(i+2) for i, hit in enumerate(hits))
        ideal = sum(1/np.log2(i+2) for i in range(min(2, len(truth[user]))))
        ap = sum(sum(hits[:i+1])/(i+1) for i, hit in enumerate(hits) if hit)/min(2, len(truth[user]))
        expected.append([precision, recall, dcg/ideal, ap])
    result = session.recommender.evaluate(partition=partition, k=2)
    keys = ["precision_at_k", "recall_at_k", "ndcg_at_k", "map_at_k"]
    assert result.metrics == pytest.approx(dict(zip(keys, np.mean(expected, axis=0), strict=True)))
    assert result.n_users_scored == 4 and result.n_cold_start_users == 1


@pytest.mark.parametrize("damage", ["shape", "dtype", "unobserved_nonzero", "count"])
@pytest.mark.parametrize("boundary", ["save", "load", "recommend"])
def test_inconsistent_observation_metadata_refuses_before_consumption(damage, boundary, tmp_path):
    original, frame, n, h, _ = _fit_data("item_knn", "zero")
    plan = copy.deepcopy(original.recommender_plan)
    _, observed = _arrays(frame, n, plan)
    if damage == "shape":
        observed = observed[:, :-1]
    elif damage == "dtype":
        observed = observed.astype(float)
    elif damage == "unobserved_nonzero":
        observed[tuple(np.argwhere(plan.matrix_ != 0)[0])] = False
    else:
        observed[tuple(np.argwhere(~observed)[0])] = True
    plan.observed_mask_ = observed
    path = tmp_path/"invalid"
    if boundary == "load":
        save_recommender_bundle(path, original.recommender_plan)
        payload = joblib.load(path/"recommender_plan.joblib")
        payload["plan"] = plan
        joblib.dump(payload, path/"recommender_plan.joblib")
    with pytest.raises(ValidationError, match="observ|matrix"):
        if boundary == "save":
            save_recommender_bundle(path, plan)
        elif boundary == "load":
            load_recommender_bundle(path, trusted=True)
        else:
            recommend_for_users(plan, plan.user_ids, k=2)
    if boundary == "save":
        assert not path.exists()


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("payload_kind", ["direct", "separate"])
def test_observation_history_restores_from_direct_and_separate_payloads(method, payload_kind, tmp_path):
    original, frame, n, _, _ = _fit_data(method, "zero")
    expected = original.recommender.recommend(user_ids=["u0", "u3"], k=99)
    path = save_recommender_bundle(tmp_path/"history", original.recommender_plan)
    payload = joblib.load(path/"recommender_plan.joblib")
    if payload_kind == "direct":
        payload = payload["plan"]
    else:
        payload["plan"].observed_mask_ = None
    joblib.dump(payload, path/"recommender_plan.joblib")
    plan = load_recommender_bundle(path, trusted=True)
    _, observed = _arrays(frame, n, plan)
    np.testing.assert_array_equal(plan.observed_mask_, observed)
    actual = recommend_for_users(plan, ["u0", "u3"], k=99)
    assert actual.recommendations == expected.recommendations
    for a, b in zip(actual.scores, expected.scores, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_legacy_missing_observation_metadata_discloses_refit_boundary(method):
    original, _, _, _, _ = _fit_data(method, "zero")
    plan = copy.deepcopy(original.recommender_plan)
    plan.observed_mask_ = None
    result = recommend_for_users(plan, ["u0", "u3"], k=99)
    assert any("legacy" in note.lower() and "refit" in note.lower() for note in result.warnings)
    for user, items in zip(result.user_ids, result.recommendations, strict=True):
        row = plan.matrix_[plan.user_index_[user]]
        assert set(items) == {item for i, item in enumerate(plan.item_ids) if row[i] == 0}
