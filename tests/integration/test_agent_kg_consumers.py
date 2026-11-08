"""Real KG agent workflows agree with independent scores and graph edges."""
import itertools

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.executor import execute_tool, propose_tool_execution
from buildml.ai.tools import build_default_registry


def execute(session, name, arguments):
    registry = build_default_registry()
    result = execute_tool(session, propose_tool_execution(name, arguments, registry), confirmed=True, registry=registry)
    assert result.executed, result.error
    return result.result


@pytest.fixture(params=[("transe", "l1"), ("transe", "l2"), ("distmult", "l1")])
def trained(request):
    triples = [(h, r, t) for h, t in itertools.permutations("abcd", 2) for r in ["r", "s"]]
    frame = pd.DataFrame(triples, columns=["head", "relation", "tail"])
    session = Session.ingest(frame).set_roles({"head": "id", "relation": "id", "tail": "id"}).split(test_size=.25, validation_size=.25, random_state=5)
    method, norm = request.param
    execute(session, "fit_kg", dict(backend="native", method=method, norm=norm, head_column="head", relation_column="relation", tail_column="tail", embedding_dim=4, epochs=2, batch_size=4, random_state=6))
    return session


def score(plan, triple):
    h, r, t = triple
    a = plan.entity_embeddings_[plan.entity_index_[h]]
    b = plan.relation_embeddings_[plan.relation_index_[r]]
    c = plan.entity_embeddings_[plan.entity_index_[t]]
    if plan.method == "distmult":
        return float(np.sum(a * b * c))
    return -float(np.linalg.norm(a + b - c, ord=1 if plan.norm == "l1" else 2))


@pytest.mark.parametrize("partition", ["train", "validation", "test"])
def test_agent_scores_use_frozen_fitted_embeddings(trained, partition):
    plan = trained.kg_plan
    before = plan.entity_embeddings_.copy()
    expected = [score(plan, triple) for triple in trained.partition(partition).itertuples(index=False, name=None)]
    actual = execute(trained, "score_triples", {"partition": partition})
    np.testing.assert_allclose(actual.scores, expected)
    np.testing.assert_array_equal(plan.entity_embeddings_, before)


@pytest.mark.parametrize("mode", ["head", "tail", "relation"])
@pytest.mark.parametrize("filtered", [False, True])
def test_agent_link_ranks_match_explicit_candidate_scores(trained, mode, filtered):
    plan = trained.kg_plan
    h, r, t = next(trained.partition("train").itertuples(index=False, name=None))
    known = set(trained.partition("train").itertuples(index=False, name=None))
    candidates = plan.relation_ids if mode == "relation" else plan.entity_ids
    expected = []
    for value in candidates:
        triple = (value, r, t) if mode == "head" else (h, value, t) if mode == "relation" else (h, r, value)
        if not filtered or triple not in known:
            expected.append((value, score(plan, triple)))
    expected.sort(key=lambda pair: pair[1], reverse=True)
    args = {"mode": mode, "k": 10, "filtered": filtered}
    if mode != "head":
        args["heads"] = [h]
    if mode != "relation":
        args["relations"] = [r]
    if mode != "tail":
        args["tails"] = [t]
    actual = execute(trained, "predict_links", args)
    assert actual.predictions[0] == tuple(x[0] for x in expected)
    np.testing.assert_allclose(actual.scores[0], [x[1] for x in expected])


@pytest.mark.parametrize("direction", ["out", "in", "both"])
def test_agent_neighbors_use_only_training_edges(trained, direction):
    entity = trained.partition("train").iloc[0]["head"]
    expected = set()
    for h, r, t in trained.partition("train").itertuples(index=False, name=None):
        if direction in {"out", "both"} and h == entity:
            expected.add((t, r, "out"))
        if direction in {"in", "both"} and t == entity:
            expected.add((h, r, "in"))
    actual = execute(trained, "query_kg", {"mode": "neighbors", "entity": entity, "direction": direction})
    assert set(actual.results) == expected


def test_agent_evaluation_keeps_model_frozen(trained):
    plan = trained.kg_plan
    before = plan.entity_embeddings_.copy()
    result = execute(trained, "evaluate_kg", {"partition": "test", "k": 2})
    assert result.n_triples_scored == len(trained.partition("test"))
    holdout = list(trained.partition("test").itertuples(index=False, name=None))
    known = set(trained.partition("train").itertuples(index=False, name=None)) | set(holdout)
    ranks = []
    for true in holdout:
        for position in [0, 1, 2]:
            candidates = plan.relation_ids if position == 1 else plan.entity_ids
            true_score = score(plan, true)
            better = 0
            for candidate in candidates:
                triple = list(true)
                triple[position] = candidate
                triple = tuple(triple)
                if triple != true and triple not in known and score(plan, triple) > true_score:
                    better += 1
            ranks.append(1 + better)
    assert result.metrics["mrr"] == pytest.approx(np.mean(1 / np.asarray(ranks)))
    assert result.metrics["mean_rank"] == pytest.approx(np.mean(ranks))
    for cutoff in [1, 2, 3]:
        assert result.metrics[f"hits_at_{cutoff}"] == pytest.approx(np.mean(np.asarray(ranks) <= cutoff))
    np.testing.assert_array_equal(plan.entity_embeddings_, before)


def test_agent_bundle_save_requires_separate_deserialization_trust(trained, tmp_path):
    path = tmp_path / "knowledge bundle"
    original = trained.kg_plan
    result = execute(trained, "save_kg_bundle", {"path": str(path)})
    assert result["path"] == str(path)
    other = Session.ingest(pd.DataFrame({"x": [1, 2]}))
    registry = build_default_registry()
    attempt = execute_tool(other, propose_tool_execution("load_kg_bundle", {"path": str(path)}, registry), confirmed=True, registry=registry)
    assert not attempt.executed
    assert "trust" in attempt.error.lower()
    assert other.kg_plan is None
    other.kg.load_bundle(path, trusted=True)
    np.testing.assert_array_equal(other.kg_plan.entity_embeddings_, original.entity_embeddings_)
    before = execute(trained, "score_triples", {"partition": "test"})
    explicit = other.kg.score_triples(triples=list(zip(before.heads, before.relations, before.tails, strict=True)))
    np.testing.assert_allclose(explicit.scores, before.scores)
