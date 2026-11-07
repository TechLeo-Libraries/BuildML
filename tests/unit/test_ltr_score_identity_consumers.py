"""Ranking preserves user identities and graded labels through score ordering."""

import copy
import json
import math

import joblib
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.executor import execute_tool, propose_tool_execution
from buildml.ai.tools import build_default_registry
from buildml.core.errors import ValidationError


def _session(collision="none", method="pointwise"):
    rng = np.random.default_rng(318)
    grades = rng.integers(0, 4, size=72).astype(float)
    grades[54:60] = 0
    grades[66:72] = 0
    frame = pd.DataFrame({"query": np.repeat([f"q{i}" for i in range(12)], 6),
                          "item": [f"item-{i}" for i in range(72)],
                          "x": grades + rng.normal(0, .7, 72),
                          "z": rng.normal(size=72), "grade": grades},
                         index=[i % 7 for i in range(72)])
    names = {column: "__score__" if column == collision else column for column in frame.columns}
    frame = frame.rename(columns=names)
    session = Session.ingest(frame).set_roles({names["query"]: "group", names["item"]: "id",
                                               names["x"]: "feature", "z": "feature",
                                               names["grade"]: "target"})
    session.inject_split(train_indices=list(range(48)), validation_indices=list(range(48, 60)),
                         test_indices=list(range(60, 72)))
    session.ranking.fit(backend="sklearn", method=method, query_column=names["query"],
                         item_column=names["item"], relevance_column=names["grade"],
                         feature_columns=[names["x"], "z"], alpha=.8, C=.7, random_state=4)
    return session, names


def _scores(session, names, rows, method):
    train = session.partition("train")
    columns = [names["x"], "z"]
    mean = train[columns].to_numpy().mean(axis=0)
    std = train[columns].to_numpy().std(axis=0)
    design = (train[columns].to_numpy() - mean) / std
    score_design = (rows[columns].to_numpy() - mean) / std
    if method == "pointwise":
        target = train[names["grade"]].to_numpy()
        coefficients = np.linalg.solve(design.T @ design + .8 * np.eye(2),
                                        design.T @ (target - target.mean()))
        return score_design @ coefficients + target.mean()
    state = session.ranker_plan.estimator_
    return score_design @ state.coef_ + state.intercept_


def _metrics(grades, k):
    hits = [position for position, value in enumerate(grades[:k], 1) if value > 0]
    dcg = sum((2**value-1) / math.log2(position+2) for position, value in enumerate(grades[:k]))
    ideal = sum((2**value-1) / math.log2(position+2)
                for position, value in enumerate(sorted(grades, reverse=True)[:k]))
    ap = sum(hit/position for hit, position in enumerate(hits, 1)) / min(k, sum(value > 0 for value in grades))
    return dcg/ideal, ap, 1/hits[0] if hits else 0


@pytest.mark.parametrize("method", ["pointwise", "pairwise"])
@pytest.mark.parametrize("collision", ["none", "query", "item", "grade", "x"])
@pytest.mark.parametrize("partition", ["validation", "test"])
@pytest.mark.parametrize("k", [3, np.int64(20)])
def test_rank_identity_metrics_and_saved_replay_follow_independent_arithmetic(
    method, collision, partition, k, tmp_path
):
    session, names = _session(collision, method)
    before = session.dataset.frame.copy(deep=True)
    frozen = joblib.hash(session.ranker_plan)
    rows = session.partition(partition)
    expected_scores = _scores(session, names, rows, method)
    queries = list(dict.fromkeys(rows[names["query"]]))
    ranked = session.ranking.rank(partition=partition, k=k)
    assert ranked.query_ids == tuple(queries)
    metrics = []
    for j, query in enumerate(queries):
        positions = [i for i, value in enumerate(rows[names["query"]]) if value == query]
        ordered = sorted(positions, key=lambda position: -expected_scores[position])
        assert ranked.n_candidates[j] == len(positions)
        assert ranked.rankings[j] == tuple(rows[names["item"]].iloc[ordered[:int(k)]])
        np.testing.assert_allclose(ranked.scores[j], expected_scores[ordered[:int(k)]], atol=1e-10)
        relevance = rows[names["grade"]].iloc[ordered].tolist()
        if any(value > 0 for value in relevance):
            metrics.append(_metrics(relevance, int(k)))
    evaluated = session.ranking.evaluate(partition=partition, k=k)
    assert type(session.history[-1]["parameters"]["k"]) is int
    assert type(evaluated.k) is int and type(ranked.k) is int
    assert evaluated.n_queries_scored == len(metrics)
    expected_metrics = np.mean(metrics, axis=0)
    for name, expected in zip(["ndcg_at_k", "map_at_k", "mrr_at_k"], expected_metrics, strict=True):
        assert evaluated.metrics[name] == pytest.approx(expected)
    assert joblib.hash(session.ranker_plan) == frozen
    bundle = session.ranking.save_bundle(tmp_path / "ranker")
    session.ranking.load_bundle(bundle, trusted=True)
    assert session.ranking.rank(partition=partition, k=k).rankings == ranked.rankings
    assert session.ranking.evaluate(partition=partition, k=k).metrics == pytest.approx(evaluated.metrics)
    pd.testing.assert_frame_equal(session.dataset.frame, before)
    json.dumps(ranked.to_dict(), allow_nan=False)


@pytest.mark.parametrize("operation", ["rank", "evaluate"])
@pytest.mark.parametrize("count", [True, np.bool_(True), 2.7, "3", None, float("nan"), float("inf"), 0, -1])
def test_invalid_cutoff_refuses_before_overwriting_results_or_history(operation, count):
    session, _ = _session()
    session.ranking.rank(k=3)
    session.ranking.evaluate(k=3)
    before = session.dataset.frame.copy(deep=True)
    ranked, evaluated, fitted = session.ranker_rank_result, session.ranker_eval_result, session.ranker_plan
    history = copy.deepcopy(session.history)
    with pytest.raises(ValidationError, match="k must"):
        getattr(session.ranking, operation)(k=count)
    assert session.ranker_rank_result is ranked and session.ranker_eval_result is evaluated
    assert session.ranker_plan is fitted and session.history == history
    pd.testing.assert_frame_equal(session.dataset.frame, before)


@pytest.mark.parametrize("route", ["executor", "provider", "autonomous"])
@pytest.mark.parametrize("partition", [None, "validation"])
def test_ai_query_subset_order_reaches_frozen_ranker(route, partition):
    session, names = _session()
    session.ai.configure(provider="mock")
    arguments = dict(query_ids=["q9", "q8"], k=3, backend="sklearn")
    if partition is not None:
        arguments["partition"] = partition
    expected_rows = session.dataset.frame if partition is None else session.partition(partition)
    expected_scores = _scores(session, names, expected_rows, "pointwise")
    if route == "executor":
        registry = build_default_registry()
        proposal = propose_tool_execution("rank", arguments, registry)
        assert not proposal.requires_confirmation
        execution = execute_tool(session, proposal, confirmed=False, registry=registry)
        assert execution.executed, execution.error
        result = execution.result
    else:
        session._ai_provider.queue_responses([json.dumps({"steps": [
            {"operation": "rank", "parameters": arguments},
        ]})])
        if route == "autonomous":
            execution = session.ai.run_autonomous("Rank specified queries", confirm_autonomy=True,
                                                  max_steps=1, tool_allowlist=("rank",))
            assert execution.steps[0].arguments == arguments
        else:
            plan = session.ai.plan("Rank specified queries")
            execution = session.ai.run_plan(plan)
        assert execution.completed_steps == 1
        result = session.ranker_rank_result
    assert result.query_ids == ("q9", "q8")
    for j, query in enumerate(result.query_ids):
        positions = [i for i, value in enumerate(expected_rows["query"]) if value == query]
        ordered = sorted(positions, key=lambda position: -expected_scores[position])[:3]
        assert result.rankings[j] == tuple(expected_rows["item"].iloc[ordered])
        np.testing.assert_allclose(result.scores[j], expected_scores[ordered], atol=1e-10)


def test_ai_schema_advertises_explicit_query_scope():
    spec = build_default_registry().get("rank")
    assert "query_ids" in spec.parameters["properties"]


@pytest.mark.parametrize("collision", ["none", "query", "item", "grade"])
@pytest.mark.parametrize("partition", ["validation", "test"])
def test_equal_scores_keep_candidate_order_and_metric_labels(collision, partition):
    frame = pd.DataFrame({"query": np.repeat(["a", "b", "c", "d"], 24),
                          "item": [f"item-{95-i}" for i in range(96)],
                          "x": np.zeros(96), "grade": [3., 0., 1., 0., 0., 2.] * 16},
                         index=[5] * 96)
    names = {column: "__score__" if column == collision else column for column in frame.columns}
    frame = frame.rename(columns=names)
    session = Session.ingest(frame).set_roles({names["query"]: "group", names["item"]: "id",
                                               "x": "feature", names["grade"]: "target"})
    session.inject_split(train_indices=list(range(48)), validation_indices=list(range(48, 72)),
                         test_indices=list(range(72, 96)))
    session.ranking.fit(backend="sklearn", method="pointwise", query_column=names["query"],
                         item_column=names["item"], relevance_column=names["grade"],
                         feature_columns=["x"])
    rows = session.partition(partition)
    result = session.ranking.rank(partition=partition, k=17)
    assert result.rankings == (tuple(rows[names["item"]].iloc[:17]),)
    np.testing.assert_allclose(result.scores[0], np.full(17, frame[names["grade"]].iloc[:48].mean()))
    expected = _metrics(rows[names["grade"]].tolist(), 17)
    actual = session.ranking.evaluate(partition=partition, k=17)
    for name, value in zip(["ndcg_at_k", "map_at_k", "mrr_at_k"], expected, strict=True):
        assert actual.metrics[name] == pytest.approx(value)


@pytest.mark.parametrize("requested", [[], ["q9", "q8", "q9"]])
def test_explicit_empty_or_repeated_ai_queries_do_not_default_to_test_population(requested):
    session, _ = _session()
    registry = build_default_registry()
    proposal = propose_tool_execution("rank", {"query_ids": requested, "k": 2}, registry)
    result = execute_tool(session, proposal, confirmed=False, registry=registry)
    assert result.executed, result.error
    assert result.result.query_ids == tuple(requested)
    assert result.result.n_queries == len(requested)


def test_missing_ai_query_refuses_without_replacing_previous_ranking():
    session, _ = _session()
    previous = session.ranking.rank()
    before = copy.deepcopy(session.history)
    registry = build_default_registry()
    proposal = propose_tool_execution("rank", {"query_ids": ["unknown-query"], "k": 2}, registry)
    result = execute_tool(session, proposal, confirmed=False, registry=registry)
    assert not result.executed and "query_ids" in result.error
    assert session.ranker_rank_result is previous and session.history == before
