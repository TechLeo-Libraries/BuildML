"""Undirected topology uses endpoint identity rather than display strings."""

import copy
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError

KINDS = ["strings", "integers", "mixed_integer", "mixed_float"]
METHODS = ["classical", "gcn", "pyg_gcn", "pyg_graphsage", "pyg_gat"]


def _data(kind):
    ids = {"strings": [f"node-{i:03d}" for i in range(24)],
           "integers": list(range(24)),
           "mixed_integer": [1, "1"] + [f"n{i}" for i in range(22)],
           "mixed_float": [1.5, "1.5"] + [f"n{i}" for i in range(22)]}[kind]
    rng = np.random.default_rng(13)
    frame = pd.DataFrame({"node": pd.Series(ids, dtype=object), "x": rng.normal(size=24),
                          "z": rng.normal(size=24), "y": ["a", "b"]*12,
                          "context": ["east", "west"]*12})
    frame.index = pd.Index(np.arange(24) % 5, name="duplicate_rows")
    pairs = [(i, i+1) for i in range(23)] + [(0, 5), (1, 6), (4, 18)]
    return frame, ids, pairs


def _session(kind, form="pairs", *, duplicate=False, directed=False):
    frame, ids, pairs = _data(kind)
    edges = [(ids[a], ids[b]) for a, b in pairs] + [(ids[0], "unknown-node")]
    if duplicate:
        edges += [(ids[b], ids[a]) for a, b in pairs[:3]]
        edges += [edges[0], (ids[0], ids[0]), (None, ids[0])]
    if form == "frame":
        edges = pd.DataFrame(edges, columns=["from", "to"], dtype=object)
    elif form == "array":
        edges = np.asarray(edges, dtype=object)
    session = Session.ingest(frame).set_roles({"node": "id", "x": "feature", "z": "feature",
                                             "y": "target", "context": "group"})
    session.inject_split(train_indices=list(range(16)), validation_indices=list(range(16, 20)),
                         test_indices=list(range(20, 24)))
    session.graph.set_spec(edges, node_id_col="node", source_col="from", target_col="to", directed=directed)
    return session


def _fit(session, method, mode, *, columns=None, metrics=True):
    if method == "classical" and metrics:
        pytest.importorskip("networkx")
    if method != "classical":
        torch = pytest.importorskip("torch")
        torch.set_num_threads(1)
    if method.startswith("pyg_"):
        pytest.importorskip("torch_geometric")
    return session.graph.fit(method="pyg" if method.startswith("pyg_") else method,
        pyg_model=method.removeprefix("pyg_") if method.startswith("pyg_") else "gcn",
        mode=mode, columns=columns, include_graph_metrics=metrics, epochs=3,
        n_layers=1, hidden_dim=5, heads=1, dropout=0., random_state=13)


def _edges(mode, *, fitting=False, directed=False, reverse=False):
    pairs = _data("strings")[2]
    if reverse:
        pairs += [(b, a) for a, b in pairs[:3]]
    if mode == "inductive":
        pairs = ([(a, b) for a, b in pairs if a < 16 and b < 16] if fitting else
                 [(a, b) for a, b in pairs if a < 16 or b < 16])
    adjacency = np.zeros((24, 24))
    for a, b in pairs:
        adjacency[b, a] = 1
        if not directed:
            adjacency[a, b] = 1
    return pairs, adjacency


def _design(session, mode, *, fitting=False, reverse=False):
    nx = pytest.importorskip("networkx")
    plan = session.graph_plan
    frame = session.to_pandas()
    blocks = [frame[list(plan.feature_columns)].to_numpy(dtype=float)]
    if plan.graph_metric_names:
        pairs, _ = _edges(mode, fitting=fitting, directed=plan.directed, reverse=reverse)
        graph = nx.DiGraph() if plan.directed else nx.Graph()
        graph.add_nodes_from(range(24))
        graph.add_edges_from(pairs)
        degree = dict(graph.degree())
        cluster = dict.fromkeys(range(24), 0.) if plan.directed else nx.clustering(graph)
        neighbor = dict.fromkeys(range(24), 0.) if plan.directed else nx.average_neighbor_degree(graph)
        rank = nx.pagerank(graph, alpha=.85, max_iter=100)
        between = nx.betweenness_centrality(graph, normalized=True)
        blocks.append(np.array([[degree[i], cluster[i], rank[i], neighbor[i], between[i]] for i in range(24)]))
    return np.hstack(blocks)


def _probabilities(session, method, mode, *, reverse=False):
    plan = session.graph_plan
    if method == "classical":
        values = _design(session, mode, reverse=reverse)
        training = _design(session, mode, fitting=True, reverse=reverse)[:16]
        scale = training.std(axis=0)
        scale[scale == 0] = 1
        scaler = plan.estimator_.named_steps["standardscaler"]
        np.testing.assert_allclose(scaler.mean_, training.mean(axis=0), atol=1e-12)
        np.testing.assert_allclose(scaler.scale_, scale, atol=1e-12)
        model = plan.estimator_.named_steps["logisticregression"]
        logits = ((values-training.mean(axis=0))/scale) @ model.coef_.T + model.intercept_
        positive = 1/(1+np.exp(-logits[:, 0]))
        return np.column_stack([1-positive, positive])
    values = session.to_pandas()[list(plan.feature_columns)].to_numpy(dtype=np.float32).astype(float)
    _, adjacency = _edges(mode)
    weights = {k: v.numpy().astype(float) for k, v in plan.estimator_.to_state()["state_dict"].items()}
    if method in {"gcn", "pyg_gcn"}:
        adjacency += np.eye(24)
        inverse = 1/np.sqrt(adjacency.sum(axis=1))
        normalized = inverse[:, None]*adjacency*inverse[None, :]
        if method == "gcn":
            logits = (normalized @ values) @ weights["lin1.weight"].T + weights["lin1.bias"]
            if "lin2.weight" in weights:
                logits = (normalized @ np.maximum(logits, 0)) @ weights["lin2.weight"].T + weights["lin2.bias"]
        else:
            logits = (normalized @ values) @ weights["conv1.lin.weight"].T + weights["conv1.bias"]
    elif method == "pyg_graphsage":
        counts = adjacency.sum(axis=1)
        means = (adjacency @ values)/np.maximum(counts[:, None], 1)
        logits = (means @ weights["conv1.lin_l.weight"].T + weights["conv1.lin_l.bias"]
                  + values @ weights["conv1.lin_r.weight"].T)
    else:
        projected = values @ weights["conv1.lin.weight"].T
        source = weights["conv1.att_src"].ravel()
        target = weights["conv1.att_dst"].ravel()
        logits = []
        for i in range(24):
            neighbors = sorted(set(np.flatnonzero(adjacency[i]).tolist()+[i]))
            scores = [sum(projected[j]*source)+sum(projected[i]*target) for j in neighbors]
            scores = np.array([s if s >= 0 else .2*s for s in scores])
            attention = np.exp(scores-scores.max())
            attention /= attention.sum()
            logits.append(sum(weight*projected[j] for weight, j in zip(attention, neighbors)) + weights["conv1.bias"])
        logits = np.asarray(logits)
    shifted = logits-logits.max(axis=1, keepdims=True)
    return np.exp(shifted)/np.exp(shifted).sum(axis=1, keepdims=True)


def _check(session, method, mode, *, reverse=False):
    before, fingerprint = session.to_pandas(), joblib.hash(session.graph_plan)
    probabilities = _probabilities(session, method, mode, reverse=reverse)
    for partition, positions in [("all", list(range(24))), ("validation", list(range(16, 20))),
                                  ("test", list(range(20, 24)))]:
        actual = session.graph.predict(partition=partition)
        expected = probabilities[positions]
        np.testing.assert_allclose(actual.probabilities, expected, atol=4e-7, rtol=4e-7)
        labels = tuple(plan_label for plan_label in np.asarray(session.graph_plan.classes_, dtype=object)[expected.argmax(axis=1)])
        assert actual.predictions == labels
        if partition != "all":
            report = session.graph.evaluate(partition=partition)
            truth = before.y.iloc[positions].tolist()
            assert report.metrics["accuracy"] == pytest.approx(sum(a == b for a, b in zip(truth, labels))/len(positions))
            json.dumps(report.to_dict(), allow_nan=False)
    assert joblib.hash(session.graph_plan) == fingerprint
    pd.testing.assert_frame_equal(session.to_pandas(), before)
    return probabilities


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("form", ["pairs", "frame", "array"])
@pytest.mark.parametrize("mode", ["inductive", "transductive"])
@pytest.mark.parametrize("method", METHODS)
def test_reverse_duplicates_do_not_change_topology_or_fitted_consumers(kind, form, mode, method, tmp_path):
    reference = _session(kind, form)
    _fit(reference, method, mode)
    expected = _check(reference, method, mode)
    session = _session(kind, form, duplicate=True)
    _fit(session, method, mode)
    actual = _check(session, method, mode)
    np.testing.assert_allclose(actual, expected, atol=4e-7, rtol=4e-7)
    assert session.graph_spec.n_edges == reference.graph_spec.n_edges == 27
    assert session.graph_spec.n_nodes_in_edges == 25
    assert session.graph_plan.n_edges_fit == reference.graph_plan.n_edges_fit
    assert any("Skipped 1 edge" in s for s in session.graph_plan.disclosures)
    bundle = session.graph.save_bundle(tmp_path / "owned")
    receiver = _session(kind, form)
    receiver.graph.load_bundle(bundle, trusted=True)
    np.testing.assert_allclose(_check(receiver, method, mode), expected, atol=4e-7, rtol=4e-7)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("form", ["pairs", "frame", "array"])
def test_directed_reverse_edges_remain_distinct_in_graph_metric_only_consumers(kind, form):
    session = _session(kind, form, duplicate=True, directed=True)
    _fit(session, "classical", "transductive", columns=[])
    assert session.graph_spec.n_edges == 30
    assert session.graph_plan.feature_columns == ()
    _check(session, "classical", "transductive", reverse=True)


def _snapshot(session):
    return (copy.deepcopy(session.history), session.graph_spec, session.graph_plan,
            session.graph_fit_result, session.graph_predict_result, joblib.hash(session.graph_plan))


def _unchanged(session, snapshot):
    assert session.history == snapshot[0]
    assert session.graph_spec is snapshot[1] and session.graph_plan is snapshot[2]
    assert session.graph_fit_result is snapshot[3] and session.graph_predict_result is snapshot[4]
    assert joblib.hash(session.graph_plan) == snapshot[5]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("bad", ["shape", "columns", "null_endpoints", "self_loops", "duplicate_ids"])
def test_invalid_graph_attachment_preserves_working_spec_plan_and_history(kind, bad):
    session = _session(kind)
    _fit(session, "classical", "transductive", metrics=False)
    session.graph.predict(partition="test")
    if bad == "duplicate_ids":
        session.dataset.frame.iloc[1, session.dataset.frame.columns.get_loc("node")] = session.dataset.frame.node.iloc[0]
    snapshot = _snapshot(session)
    first = session.dataset.frame.node.iloc[0]
    edges = {"shape": [first], "columns": pd.DataFrame({"wrong": [first]}),
             "null_endpoints": [(None, first)], "self_loops": [(first, first)],
             "duplicate_ids": [(first, session.dataset.frame.node.iloc[2])]}[bad]
    with pytest.raises(ValidationError):
        session.graph.set_spec(edges, node_id_col="node")
    _unchanged(session, snapshot)


@pytest.mark.parametrize("mode", ["inductive", "transductive"])
@pytest.mark.parametrize("bad", ["unknown", "nonnumeric", "null_feature", "protected_only", "no_features"])
def test_invalid_graph_feature_inputs_preserve_working_fit(mode, bad):
    session = _session("strings")
    _fit(session, "classical", mode, metrics=False)
    session.graph.predict(partition="test")
    columns = {"unknown": ["absent"], "nonnumeric": ["context"], "null_feature": ["x"],
               "protected_only": ["node", "y"], "no_features": []}[bad]
    if bad == "nonnumeric":
        session.set_roles({"context": "feature"})
    elif bad == "null_feature":
        session.dataset.frame.iloc[0, session.dataset.frame.columns.get_loc("x")] = np.nan
    snapshot = _snapshot(session)
    with pytest.raises(ValidationError):
        _fit(session, "classical", mode, columns=columns, metrics=False)
    _unchanged(session, snapshot)


@pytest.mark.parametrize("mode", ["inductive", "transductive"])
def test_new_unmapped_graph_spec_refuses_fit_without_recording_a_model(mode):
    session = _session("strings")
    session.graph.set_spec([("unknown-a", "unknown-b")], node_id_col="node")
    snapshot = _snapshot(session)
    assert session.graph_plan is None
    with pytest.raises(ValidationError, match="No edges map"):
        _fit(session, "classical", mode, metrics=False)
    _unchanged(session, snapshot)
