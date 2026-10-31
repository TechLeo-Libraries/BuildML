"""Actual LTR consumers retain train scaling, per-query metrics and legacy scoring."""

import itertools
from dataclasses import replace

import joblib
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError


@pytest.fixture(
    scope="module",
    params=list(itertools.product(["pointwise", "pairwise"], ["strings", "integers"])),
)
def fitted(request):
    method, ids = request.param
    rng = np.random.default_rng(318)
    grades = rng.integers(0, 4, size=72).astype(float)
    grades[54:60] = 0
    grades[66:72] = 0
    q = np.repeat(np.arange(12), 6)
    frame = pd.DataFrame(
        {
            "query": q if ids == "integers" else np.array([f"q{i}" for i in q]),
            "item": [f"item-{i}" for i in range(72)],
            "x": grades + rng.normal(0, 0.7, 72),
            "z": rng.normal(size=72),
            "grade": grades,
        }
    )
    s = ready(frame)
    s.ranking.fit(
        backend="sklearn",
        method=method,
        query_column="query",
        item_column="item",
        relevance_column="grade",
        feature_columns=["x", "z"],
        alpha=0.8,
        C=0.7,
        random_state=4,
    )
    return s, frame, method


def ready(frame):
    s = Session.ingest(frame).set_roles(
        {"query": "group", "item": "id", "x": "feature", "z": "feature", "grade": "target"}
    )
    s.inject_split(
        train_indices=list(range(48)),
        validation_indices=list(range(48, 60)),
        test_indices=list(range(60, 72)),
    )
    return s


def scores(source, frame, method):
    train = source.partition("train")
    mean = train[["x", "z"]].to_numpy().mean(axis=0)
    std = train[["x", "z"]].to_numpy().std(axis=0)
    np.testing.assert_allclose(source.ranker_plan.feature_mean_, mean)
    np.testing.assert_allclose(source.ranker_plan.feature_scale_, std)
    design = (train[["x", "z"]].to_numpy() - mean) / std
    x = (frame[["x", "z"]].to_numpy() - mean) / std
    if method == "pointwise":
        y = train.grade.to_numpy()
        coef = np.linalg.solve(design.T @ design + 0.8 * np.eye(2), design.T @ (y - y.mean()))
        return x @ coef + y.mean()
    state = source.ranker_plan.estimator_
    return x @ state.coef_ + state.intercept_


def metric_oracle(relevance, k):
    top = relevance[:k]
    dcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(top) if r > 0)
    ideal = sum(
        (2**r - 1) / np.log2(i + 2)
        for i, r in enumerate(sorted(relevance, reverse=True)[:k])
        if r > 0
    )
    hits = [i + 1 for i, r in enumerate(top) if r > 0]
    precision = sum((j + 1) / position for j, position in enumerate(hits)) / min(
        k, sum(r > 0 for r in relevance)
    )
    return dcg / ideal if ideal else 0.0, precision, 1 / hits[0] if hits else 0.0


@pytest.mark.parametrize("partition", ["validation", "test"])
@pytest.mark.parametrize("k", [1, 3, 20])
@pytest.mark.parametrize("layout", ["current", "direct_legacy", "redundant"])
def test_scores_order_metrics_and_bundle_match_independent_arithmetic(
    fitted, partition, k, layout, tmp_path
):
    source, frame, method = fitted
    fingerprint = joblib.hash(source.ranker_plan)
    rows = source.partition(partition)
    expected = scores(source, rows, method)
    queries = list(dict.fromkeys(rows["query"]))
    result = source.ranking.rank(partition=partition, k=k)
    assert list(result.query_ids) == queries
    metrics = []
    for j, query in enumerate(queries):
        indices = [i for i, value in enumerate(rows["query"]) if value == query]
        order = sorted(indices, key=lambda i: -expected[i])
        assert result.rankings[j] == tuple(rows["item"].iloc[order[:k]])
        np.testing.assert_allclose(result.scores[j], expected[order[:k]], atol=1e-10)
        relevance = rows.grade.iloc[order].tolist()
        if any(r > 0 for r in relevance):
            metrics.append(metric_oracle(relevance, k))
    evaluated = source.ranking.evaluate(partition=partition, k=k)
    assert evaluated.n_queries_scored == len(metrics)
    for name, value in zip(
        ["ndcg_at_k", "map_at_k", "mrr_at_k"], np.mean(metrics, axis=0), strict=True
    ):
        assert evaluated.metrics[name] == pytest.approx(value)
    reverse = source.ranking.rank(partition=partition, k=k, query_ids=queries[::-1])
    assert reverse.rankings == result.rankings[::-1]
    root = source.ranking.save_bundle(tmp_path / "ranker")
    payload = joblib.load(root / "ranker_plan.joblib")
    if layout == "direct_legacy":
        state = payload["plan"].estimator_
        payload = replace(payload["plan"], estimator_=state.estimator_)
    elif layout == "redundant":
        payload["plan"] = replace(payload["plan"], estimator_=None, coef_=None)
    joblib.dump(payload, root / "ranker_plan.joblib")
    restored = ready(frame)
    restored.ranking.load_bundle(root, trusted=True)
    again = restored.ranking.rank(partition=partition, k=k)
    assert again.rankings == result.rankings
    for a, b in zip(again.scores, result.scores, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-12)
    assert restored.ranking.evaluate(partition=partition, k=k).metrics == pytest.approx(
        evaluated.metrics
    )
    assert joblib.hash(source.ranker_plan) == fingerprint
    pd.testing.assert_frame_equal(source.to_pandas(), frame)


@pytest.mark.parametrize(
    "fault",
    [
        "query_none",
        "item_none",
        "query_missing",
        "item_missing",
        "same_ids",
        "feature_missing",
        "feature_empty",
        "feature_target",
        "feature_text",
        "relevance_missing",
    ],
)
def test_invalid_column_contract_refuses_without_replacing_fitted_ranker(fitted, fault):
    source, _, method = fitted
    arguments = dict(
        backend="sklearn",
        method=method,
        query_column="query",
        item_column="item",
        relevance_column="grade",
        feature_columns=["x", "z"],
    )
    changes = {
        "query_none": {"query_column": None},
        "item_none": {"item_column": None},
        "query_missing": {"query_column": "absent"},
        "item_missing": {"item_column": "absent"},
        "same_ids": {"item_column": "query"},
        "feature_missing": {"feature_columns": ["absent"]},
        "feature_empty": {"feature_columns": []},
        "feature_target": {"feature_columns": ["grade"]},
        "feature_text": {"feature_columns": ["item"]},
        "relevance_missing": {"relevance_column": "absent"},
    }
    arguments.update(changes[fault])
    plan = source.ranker_plan
    before = joblib.hash(plan)
    with pytest.raises(ValidationError):
        source.ranking.fit(**arguments)
    assert source.ranker_plan is plan and joblib.hash(plan) == before
