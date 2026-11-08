"""Real recommender agent workflows preserve exclusions and numerical scores."""
import math

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.executor import execute_tool, propose_tool_execution
from buildml.ai.tools import build_default_registry


def execute(session, tool, args):
    registry = build_default_registry()
    result = execute_tool(session, propose_tool_execution(tool, args, registry), confirmed=True, registry=registry)
    assert result.executed, result.error
    return result.result


@pytest.fixture(params=["item_knn", "user_knn", "svd", "nmf", "content"])
def trained(request):
    frame = pd.DataFrame([{"user": f"u{u}", "item": f"i{i}", "rating": float(1 + (u * 3 + i * 5) % 7), "f1": float(i % 3), "f2": float(i)} for u in range(6) for i in range(8)])
    session = Session.ingest(frame).set_roles({"user": "id", "item": "id", "rating": "target", "f1": "feature", "f2": "feature"}).split(test_size=.3, validation_size=.15, random_state=7)
    execute(session, "fit_recommender", dict(backend="sklearn", method=request.param, user_column="user", item_column="item", rating_column="rating", feedback="explicit", n_neighbors=99, n_factors=3, n_iterations=30, item_feature_columns=["f1", "f2"], random_state=5))
    return session


def independent_scores(plan, user):
    matrix = plan.matrix_
    row = matrix[user]
    if plan.method in {"svd", "nmf"}:
        result = plan.user_factors_[user] @ plan.item_factors_.T
        return result + (plan.global_mean_ if plan.method == "svd" else 0)
    if plan.method == "content":
        indices = np.flatnonzero(row)
        profile = np.average(plan.item_features_[indices], axis=0, weights=np.abs(row[indices]))
        norms = np.linalg.norm(plan.item_features_, axis=1)
        denom = np.where(norms > 1e-12, norms, 1) * np.linalg.norm(profile)
        return np.divide(plan.item_features_ @ profile, denom, out=np.zeros(plan.n_items), where=denom > 1e-12)
    vectors = matrix.T if plan.method == "item_knn" else matrix
    lengths = np.linalg.norm(vectors, axis=1)
    denom = lengths[:, None] * lengths[None, :]
    sim = np.divide(vectors @ vectors.T, denom, out=np.zeros_like(denom), where=denom > 1e-12)
    if plan.method == "item_knn":
        rated = np.flatnonzero(row)
        weights = sim[:, rated]
        denom = np.abs(weights).sum(axis=1)
        return np.divide(weights @ row[rated], denom, out=np.zeros(plan.n_items), where=denom > 1e-12)
    weights = sim[user].copy()
    weights[user] = 0
    denom = np.sum(np.abs(weights[:, None]) * (matrix != 0), axis=0)
    return np.divide(weights @ matrix, denom, out=np.zeros(plan.n_items), where=denom > 1e-12)


@pytest.mark.parametrize("exclude", [False, True])
@pytest.mark.parametrize("k", [3, 100])
def test_agent_recommendations_match_scores_and_eligible_catalog(trained, exclude, k):
    plan = trained.recommender_plan
    before = plan.matrix_.copy()
    actual = execute(trained, "recommend", {"partition": "test", "k": k, "exclude_train_items": exclude})
    for user, items, scores in zip(actual.user_ids, actual.recommendations, actual.scores, strict=True):
        idx = plan.user_index_[user]
        eligible = {item for j, item in enumerate(plan.item_ids) if not exclude or plan.matrix_[idx, j] == 0}
        assert set(items) <= eligible
        assert len(items) == min(k, len(eligible))
        expected = independent_scores(plan, idx)
        np.testing.assert_allclose(scores, [expected[plan.item_index_[item]] for item in items], atol=1e-9)
        assert list(scores) == sorted(scores, reverse=True)
        if k >= len(eligible):
            assert set(items) == eligible
        elif items:
            assert min(scores) >= sorted([expected[plan.item_index_[i]] for i in eligible], reverse=True)[k-1] - 1e-9
    np.testing.assert_array_equal(plan.matrix_, before)


def test_agent_metrics_match_independent_recommendation_counts(trained):
    k = 3
    plan = trained.recommender_plan
    rec = execute(trained, "recommend", {"partition": "test", "k": k, "exclude_train_items": True})
    relevant = {}
    for user, item in trained.partition("test")[["user", "item"]].itertuples(index=False, name=None):
        if user in plan.user_index_ and item in plan.item_index_:
            relevant.setdefault(user, set()).add(item)
    values = []
    for user, items in zip(rec.user_ids, rec.recommendations, strict=True):
        if user not in relevant:
            continue
        gold = relevant[user]
        bits = [item in gold for item in items]
        count = sum(bits)
        dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(bits, 1))
        ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(gold)) + 1))
        ap = sum(sum(bits[:rank]) / rank for rank, hit in enumerate(bits, 1) if hit) / min(k, len(gold))
        values.append([count/k, count/len(gold), dcg/ideal, ap])
    result = execute(trained, "evaluate_recommender", {"partition": "test", "k": k})
    expected = np.mean(values, axis=0)
    for key, value in zip(["precision_at_k", "recall_at_k", "ndcg_at_k", "map_at_k"], expected, strict=True):
        assert result.metrics[key] == pytest.approx(value)


def test_agent_saved_bundle_roundtrip_requires_separate_trust(trained, tmp_path):
    path = tmp_path / "recommendations"
    execute(trained, "save_recommender_bundle", {"path": str(path)})
    other = Session.ingest(pd.DataFrame({"x": [1, 2]}))
    registry = build_default_registry()
    attempt = execute_tool(other, propose_tool_execution("load_recommender_bundle", {"path": str(path)}, registry), confirmed=True, registry=registry)
    assert not attempt.executed and "trust" in attempt.error.lower()
    assert other.recommender_plan is None
    other.recommender.load_bundle(path, trusted=True)
    expected = execute(trained, "recommend", {"partition": "test", "k": 3})
    actual = other.recommender.recommend(user_ids=expected.user_ids, k=3)
    assert actual.recommendations == expected.recommendations
    for a, b in zip(actual.scores, expected.scores, strict=True):
        np.testing.assert_allclose(a, b)


def test_degenerate_content_vectors_still_exclude_seen_items():
    frame = pd.DataFrame([{"user": f"u{u}", "item": f"i{i}", "rating": 1., "constant": 0.} for u in range(4) for i in range(6)])
    session = Session.ingest(frame).set_roles({"user": "id", "item": "id", "rating": "target", "constant": "feature"}).split(test_size=.3, random_state=3)
    execute(session, "fit_recommender", {"method": "content", "user_column": "user", "item_column": "item", "rating_column": "rating", "item_feature_columns": ["constant"]})
    result = execute(session, "recommend", {"partition": "test", "k": 100, "exclude_train_items": True})
    plan = session.recommender_plan
    for user, items in zip(result.user_ids, result.recommendations, strict=True):
        seen = {plan.item_ids[j] for j, value in enumerate(plan.matrix_[plan.user_index_[user]]) if value != 0}
        assert set(items).isdisjoint(seen)
        assert len(items) == plan.n_items - len(seen)
