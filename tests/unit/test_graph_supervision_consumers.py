"""Graph supervision and evaluation use their own labeled populations."""

import copy
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.executor import execute_tool, propose_tool_execution
from buildml.ai.planner import map_plan_step_to_tool
from buildml.ai.results import PlanStep
from buildml.ai.tools import build_default_registry
from buildml.core.errors import ValidationError

METHODS = ["classical", "gcn", "pyg_gcn", "pyg_graphsage", "pyg_gat"]
TRAIN = [11, 0, 9, 2, 7, 4, 5, 6, 3, 8, 1, 10]
VALID = [17, 12, 16, 13, 15, 14]
TEST = [23, 18, 22, 19, 21, 20]
EDGES = [(i, i + 1) for i in range(23)] + [(0, 6), (2, 8), (3, 15), (8, 20), (14, 22)]


def _settings(method):
    if method != "classical":
        pytest.importorskip("torch")
    if method.startswith("pyg_"):
        pytest.importorskip("torch_geometric")
    return dict(method="pyg" if method.startswith("pyg_") else method,
                pyg_model=method.removeprefix("pyg_") if method.startswith("pyg_") else "gcn",
                epochs=3, hidden_dim=5, n_layers=1, dropout=0., random_state=17,
                include_graph_metrics=False)


def _session(holdout="known", *, n_classes=2):
    x = np.linspace(-2, 2, 24)
    labels = np.asarray([f"class-{i % n_classes}" for i in range(24)], dtype=object)
    if holdout == "unseen":
        labels[12:18] = "unseen-class"
    elif holdout == "null":
        labels[12:18] = None
    frame = pd.DataFrame({"node": [f"node-{i}" for i in range(24)], "x": x,
                          "z": np.cos(x), "y": labels}, index=[i % 5 for i in range(24)])
    session = Session.ingest(frame).set_roles({"node": "id", "x": "feature", "z": "feature", "y": "target"})
    session.inject_split(train_indices=TRAIN, validation_indices=VALID, test_indices=TEST)
    edges = [[f"node-{a}", f"node-{b}"] for a, b in EDGES]
    session.graph.set_spec(edges, node_id_col="node")
    return session


def _mask_probabilities(session, mode):
    """Independent dense GCN forward or classical standardized affine forward."""
    plan = session.graph_plan
    frame = session.dataset.frame
    x = frame[["x", "z"]].to_numpy()
    if plan.method == "classical":
        train_x = x[np.sort(TRAIN)]
        scale = train_x.std(axis=0)
        scale[scale == 0] = 1
        model = plan.estimator_.named_steps["logisticregression"]
        logits = ((x - train_x.mean(axis=0)) / scale) @ model.coef_.T + model.intercept_
        if logits.shape[1] == 1:
            positive = 1/(1+np.exp(-logits[:, 0]))
            return np.column_stack([1-positive, positive])
    else:
        assert plan.method == "gcn"
        adjacency = np.eye(len(frame))
        for a, b in EDGES:
            if mode == "transductive" or a in TRAIN or b in TRAIN:
                adjacency[a, b] = adjacency[b, a] = 1
        inverse = 1 / np.sqrt(adjacency.sum(axis=1))
        adjacency = inverse[:, None] * adjacency * inverse[None, :]
        weights = plan.estimator_.to_state()["state_dict"]
        logits = (adjacency @ x) @ weights["lin1.weight"].numpy().T + weights["lin1.bias"].numpy()
        if "lin2.weight" in weights:
            logits = (adjacency @ np.maximum(logits, 0)) @ weights["lin2.weight"].numpy().T + weights["lin2.bias"].numpy()
    shifted = logits - logits.max(axis=1, keepdims=True)
    return np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("mode", ["inductive", "transductive"])
@pytest.mark.parametrize("holdout", ["known", "unseen", "null"])
@pytest.mark.parametrize("n_classes", [2, 3])
def test_nontraining_labels_do_not_change_fit_predictions_or_saved_replay(method, mode, holdout, n_classes, tmp_path):
    settings = _settings(method)
    reference = _session(n_classes=n_classes)
    reference.graph.fit(mode=mode, **settings)
    expected = reference.graph.predict(partition="all")
    session = _session(holdout, n_classes=n_classes)
    original = session.dataset.frame.copy(deep=True)
    fitted = session.graph.fit(mode=mode, **settings)
    assert fitted.n_train_nodes == 12 and fitted.n_classes == n_classes
    assert session.graph_plan.classes_ == tuple(f"class-{i}" for i in range(n_classes))
    actual = session.graph.predict(partition="all")
    np.testing.assert_allclose(actual.probabilities, expected.probabilities, atol=0, rtol=0)
    assert actual.predictions == expected.predictions
    if method in {"classical", "gcn"}:
        np.testing.assert_allclose(actual.probabilities, _mask_probabilities(session, mode),
                                   atol=3e-7, rtol=3e-7)
    before = session.graph.predict(partition="test")
    bundle = session.graph.save_bundle(tmp_path / "graph")
    restored = _session(holdout, n_classes=n_classes)
    restored.graph.load_bundle(bundle, trusted=True)
    replayed = restored.graph.predict(partition="test")
    np.testing.assert_allclose(replayed.probabilities, before.probabilities, atol=0, rtol=0)
    assert replayed.predictions == before.predictions
    # Irrelevant unlabeled validation rows cannot block a labeled test report.
    result = restored.graph.evaluate(partition="test")
    assert result.n_nodes == 6 and all(np.isfinite(list(result.metrics.values())))
    pd.testing.assert_frame_equal(session.dataset.frame, original)
    json.dumps(restored.graph_plan.to_dict(), allow_nan=False)


@pytest.mark.parametrize("method", METHODS)
def test_missing_training_labels_refuse_without_replacing_prior_graph_model(method):
    session = _session()
    settings = _settings(method)
    session.graph.fit(**settings)
    plan, fitted = session.graph_plan, session.graph_fit_result
    before = copy.deepcopy(session.history)
    session.dataset.frame.iloc[TRAIN[0], session.dataset.frame.columns.get_loc("y")] = None
    with pytest.raises(ValidationError, match="Target.*nulls"):
        session.graph.fit(**settings)
    assert session.graph_plan is plan and session.graph_fit_result is fitted
    assert session.history == before


@pytest.mark.parametrize("outside", [None, "new-label"])
@pytest.mark.parametrize("partition,positions", [("validation", VALID), ("test", TEST), ("train", TRAIN)])
def test_evaluation_validates_only_selected_label_population(outside, partition, positions):
    session = _session()
    session.graph.fit(method="classical", include_graph_metrics=False)
    selected = set(positions)
    column = session.dataset.frame.columns.get_loc("y")
    for position in range(24):
        if position not in selected:
            session.dataset.frame.iloc[position, column] = outside
    result = session.graph.evaluate(partition=partition)
    predicted = session.graph.predict(partition=partition)
    truth = session.dataset.frame.y.iloc[sorted(positions)].tolist()
    accuracy = sum(a == b for a, b in zip(truth, predicted.predictions, strict=True)) / len(truth)
    assert result.metrics["accuracy"] == pytest.approx(accuracy)
    assert all(np.isfinite(list(result.metrics.values())))


@pytest.mark.parametrize("selected", [None, "unseen-class", "class-0"])
def test_evaluation_nulls_refuse_and_undefined_auc_is_disclosed(selected):
    session = _session()
    session.graph.fit(method="classical", include_graph_metrics=False)
    previous = session.graph.evaluate(partition="test")
    session.dataset.frame.iloc[sorted(TEST), session.dataset.frame.columns.get_loc("y")] = selected
    before = copy.deepcopy(session.history)
    if selected is None:
        with pytest.raises(ValidationError, match="Target.*nulls"):
            session.graph.evaluate(partition="test")
        assert session.graph_eval_result is previous and session.history == before
    else:
        result = session.graph.evaluate(partition="test")
        assert "roc_auc" not in result.metrics
        assert any("roc_auc" in warning for warning in result.warnings)
        assert all(np.isfinite(list(result.metrics.values())))
        if selected == "unseen-class":
            assert result.metrics["accuracy"] == 0


@pytest.mark.parametrize("route", ["executor", "provider", "autonomous"])
@pytest.mark.parametrize("directed", [False, True])
def test_ai_graph_attachment_and_real_supervision_preserve_spec_and_label_boundary(route, directed):
    session = _session("null").ai.configure(provider="mock")
    edges = [[f"node-{a}", f"node-{b}"] for a, b in EDGES]
    operations = [
        ("set_graph", dict(edges=edges, node_id_col="node", source_col="from", target_col="to", directed=directed)),
        ("fit_graph", dict(method="classical", mode="inductive", include_graph_metrics=False, random_state=17)),
        ("predict_graph", dict(partition="test")),
        ("evaluate_graph", dict(partition="test")),
    ]
    if route == "executor":
        registry = build_default_registry()
        for tool, args in operations:
            result = execute_tool(session, propose_tool_execution(tool, args, registry),
                                  confirmed=True, registry=registry)
            assert result.executed, result.error
    else:
        session._ai_provider.queue_responses([json.dumps({"steps": [
            {"operation": name, "parameters": args} for name, args in operations
        ]})])
        if route == "autonomous":
            result = session.ai.run_autonomous("Attach graph and fit on labeled train nodes", confirm_autonomy=True,
                                               max_steps=4, tool_allowlist=tuple(name for name, _ in operations))
        else:
            plan = session.ai.plan("Attach graph and fit on labeled train nodes")
            result = session.ai.run_plan(plan, confirmations={i: True for i in range(4)})
        assert result.completed_steps == 4
    assert session.graph_spec.directed is directed
    assert session.graph_spec.source_col == "from" and session.graph_spec.target_col == "to"
    assert session.graph_spec.n_edges == len(EDGES)
    assert session.graph_plan.n_train_nodes == 12
    assert session.graph_eval_result.n_nodes == 6


@pytest.mark.parametrize("tool_name", [tool.name for tool in build_default_registry().tools])
def test_registered_plan_name_keeps_its_own_tool_identity(tool_name):
    step = PlanStep(operation=tool_name, description="", rationale="",
                    prerequisites=(), expected_changes=(), parameters={})
    mapping = map_plan_step_to_tool(step, build_default_registry())
    assert mapping == (tool_name, {})


@pytest.mark.parametrize("operation", ["FIT_GRAPH", "fit-graph", "EVALUATE_ASR", "fit-torch"])
def test_normalized_registered_names_win_over_generic_plan_aliases(operation):
    step = PlanStep(operation=operation, description="", rationale="",
                    prerequisites=(), expected_changes=(), parameters={"random_state": 17})
    mapping = map_plan_step_to_tool(step, build_default_registry())
    assert mapping == (operation.lower().replace("-", "_"), {"random_state": 17})


@pytest.mark.parametrize("method", METHODS[1:])
def test_configured_neural_worker_roundtrip_then_real_training_and_fitted_roundtrip(method, tmp_path):
    _settings(method)
    from buildml.graph.adapters.pyg import PyGNodeClassifier
    from buildml.graph.gnn import GCNClassifier

    parameters = dict(in_dim=2, n_classes=2, epochs=3, n_layers=1,
                      hidden_dim=5, dropout=0., random_state=17)
    model = (GCNClassifier(**parameters) if method == "gcn"
             else PyGNodeClassifier(pyg_model=method.removeprefix("pyg_"), **parameters))
    path = tmp_path / "configured.joblib"
    joblib.dump(model, path)
    model = joblib.load(path)
    session = _session("null")
    x = session.dataset.frame[["x", "z"]].to_numpy()
    labels = session.dataset.frame.y.to_numpy()
    train_mask = np.asarray([i in TRAIN for i in range(24)])
    edges = [(a, b) for a, b in EDGES if a in TRAIN and b in TRAIN]
    src, dst = np.asarray(edges).T
    if method == "gcn":
        adjacency = np.eye(24)
        adjacency[src, dst] = adjacency[dst, src] = 1
        inverse = 1/np.sqrt(adjacency.sum(axis=1))
        adjacency = inverse[:, None] * adjacency * inverse[None, :]
        model.fit(x, labels, adjacency, train_mask, {"class-0": 0, "class-1": 1})
        expected = model.predict_proba(x, adjacency)
    else:
        model.fit(x, labels, src, dst, directed=False, train_mask=train_mask,
                  class_to_index={"class-0": 0, "class-1": 1})
        expected = model.predict_proba(x, src, dst, directed=False)
    joblib.dump(model, path)
    loaded = joblib.load(path)
    actual = (loaded.predict_proba(x, adjacency) if method == "gcn"
              else loaded.predict_proba(x, src, dst, directed=False))
    np.testing.assert_allclose(actual, expected, atol=0, rtol=0)
    np.testing.assert_allclose(actual.sum(axis=1), 1, atol=1e-6)
    assert np.isfinite(actual).all() and (actual >= 0).all()
