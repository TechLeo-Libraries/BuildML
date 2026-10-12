"""Agent-driven native workflows must produce the same results as direct Session use."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.executor import execute_tool, propose_tool_execution
from buildml.ai.tools import build_default_registry
from buildml.core.errors import ValidationError


def _session(*, forecast=False, domain=None):
    rng = np.random.default_rng(491)
    x = rng.normal(size=180)
    frame = pd.DataFrame({"x": x, "z": rng.normal(size=180)})
    frame["y"] = (frame.x + .3 * frame.z > 0).astype(int)
    if forecast:
        frame["clock"] = pd.date_range("2024-01-01", periods=len(frame), freq="D")
        frame["y"] = np.arange(len(frame)) * .2 + np.sin(np.arange(len(frame)) / 5)
    roles = {"x": "feature", "z": "feature", "y": "target"}
    if domain in {"federated", "metalearning", "ranking"}:
        frame["group"] = np.arange(len(frame)) % 6
        roles["group"] = "group"
    if domain == "multitask":
        frame["second"] = (frame.z > 0).astype(int)
        roles["second"] = "target"
    if domain == "ranking":
        frame["item"] = np.arange(len(frame))
        roles["item"] = "id"
    if domain == "nlp":
        frame["text"] = np.where(frame.y == 1, "engine oil valve", "apple orchard fruit")
        roles["text"] = "feature"
    if domain == "causal":
        frame["t"] = rng.integers(0, 2, len(frame))
        frame["y"] = 1.6 * frame.t + .6 * frame.x - .3 * frame.z + rng.normal(0, .15, len(frame))
        roles["t"] = "feature"
    if domain == "graph":
        frame["node"] = np.arange(len(frame))
        roles["node"] = "id"
    if forecast:
        roles["clock"] = "time"
    session = Session.ingest(frame).set_roles(roles).ai.configure(provider="mock")
    if forecast:
        session.time_split(test_size=.2, validation_size=.2)
    else:
        session.split(test_size=.2, validation_size=.2, random_state=31)
    if domain == "graph":
        ordered = np.argsort(x)
        session.set_graph(np.column_stack([ordered[:-1], ordered[1:]]), node_id_col="node")
    if domain == "causal":
        session.declare_causal_assumptions(treatment="t", outcome="y", confounders=["x", "z"],
                                         acknowledge_unconfoundedness=True, acknowledge_positivity=True)
    return session


def _run(session, name, args):
    registry = build_default_registry()
    proposal = propose_tool_execution(name, args, registry)
    outcome = execute_tool(session, proposal, confirmed=True, registry=registry)
    assert outcome.executed, f"{name}: {outcome.error}"
    assert outcome.error is None
    return outcome.result


WORKFLOWS = [
    ("unsupervised", "fit_clusters", {"method": "kmeans", "n_clusters": 2}, "evaluate_clusters"),
    ("ensemble", "fit_voting", {"estimators": ["LogisticRegression", "KNeighborsClassifier"]}, "evaluate_ensemble"),
    ("ensemble", "fit_stacking", {"estimators": ["LogisticRegression", "KNeighborsClassifier"], "cv": 3}, "evaluate_ensemble"),
    ("ensemble", "fit_blending", {"estimators": ["LogisticRegression", "KNeighborsClassifier"], "holdout_fraction": .2}, "evaluate_ensemble"),
    ("forecast", "fit_forecast", {"method": "lag_ridge", "lags": [1, 2, 3], "horizon": 5}, "evaluate_forecast"),
    ("anomaly", "fit_anomaly", {"method": "isolation_forest", "contamination": .1}, "evaluate_anomaly"),
    ("online", "fit_online", {"backend": "sklearn", "chunk_size": 30, "n_init": 30}, "evaluate_online"),
    ("probabilistic", "fit_probabilistic", {"estimator": "gaussian_nb", "task": "classification", "alpha": .1}, "evaluate_probabilistic"),
    ("symbolic", "fit_symbolic", {"source": "decision_tree", "task": "classification"}, "evaluate_symbolic"),
    ("cbr", "fit_cbr", {"backend": "sklearn", "metric": "euclidean", "k": 3}, "evaluate_cbr"),
    ("multitask", "fit_multitask", {"backend": "sklearn", "targets": ["y", "second"], "task": "classification"}, "evaluate_multitask"),
    ("federated", "fit_federated", {"backend": "native", "client_column": "group", "n_rounds": 2, "local_epochs": 1}, "evaluate_federated"),
    ("metalearning", "fit_metalearning", {"backend": "sklearn", "task_column": "group", "k_shot": 2, "n_query": 3, "n_episodes": 3}, "evaluate_metalearning"),
    ("synthetic", "fit_synthesizer", {"backend": "native", "method": "bootstrap", "random_state": 11}, "evaluate_synthetic"),
    ("ranker", "fit_ranker", {"backend": "sklearn", "method": "pointwise", "query_column": "group", "item_column": "item", "relevance_column": "y"}, "evaluate_ranker"),
    ("nlp", "fit_text_classifier", {"text_column": "text"}, "evaluate_text_classifier"),
    ("causal", "fit_causal", {"backend": "native", "method": "t_learner", "bootstrap_samples": 20}, "evaluate_causal"),
    ("graph", "fit_graph", {"method": "classical", "include_graph_metrics": True}, "evaluate_graph"),
    ("imitation", "fit_imitation", {"backend": "sklearn", "action_column": "y", "task": "classification"}, "evaluate_imitation"),
]


@pytest.mark.parametrize("domain,fit_tool,args,evaluate_tool", WORKFLOWS)
def test_agent_fit_and_evaluate_match_direct_api(domain, fit_tool, args, evaluate_tool):
    via_agent = _session(forecast=domain == "forecast", domain="ranking" if domain == "ranker" else domain)
    direct = _session(forecast=domain == "forecast", domain="ranking" if domain == "ranker" else domain)
    train_before = tuple(via_agent.split_plan.train_indices)
    test_before = via_agent.partition("test").copy()
    _run(via_agent, fit_tool, args)
    if domain == "ensemble":
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier

        resolved = {**args, "estimators": {
            "logisticregression": LogisticRegression(),
            "kneighborsclassifier": KNeighborsClassifier(),
        }}
        getattr(direct, fit_tool)(**resolved)
    else:
        getattr(direct, fit_tool)(**args)
    agent_result = _run(via_agent, evaluate_tool, {"partition": "test"})
    expected = getattr(direct, evaluate_tool)(partition="test")
    assert agent_result.metrics.keys() == expected.metrics.keys()
    for metric, value in expected.metrics.items():
        if isinstance(value, (int, float)) and np.isfinite(value):
            assert agent_result.metrics[metric] == pytest.approx(value)
    assert tuple(via_agent.split_plan.train_indices) == train_before
    pd.testing.assert_frame_equal(via_agent.partition("test"), test_before)


@pytest.mark.parametrize("domain,fit_tool,args,evaluate_tool", WORKFLOWS)
def test_agent_domain_save_round_trip_requires_explicit_trust(
    domain, fit_tool, args, evaluate_tool, tmp_path
):
    session = _session(forecast=domain == "forecast", domain="ranking" if domain == "ranker" else domain)
    _run(session, fit_tool, args)
    before = _run(session, evaluate_tool, {"partition": "test"})
    path = str(tmp_path / domain)
    _run(session, f"save_{domain}_bundle", {"path": path})
    registry = build_default_registry()
    proposal = propose_tool_execution(f"load_{domain}_bundle", {"path": path}, registry)
    restored = _session(forecast=domain == "forecast", domain="ranking" if domain == "ranker" else domain)
    outcome = execute_tool(restored, proposal, confirmed=True, registry=registry)
    assert not outcome.executed
    assert "trust" in outcome.error.lower()
    getattr(restored, f"load_{domain}_bundle")(path, trusted=True)
    after = getattr(restored, evaluate_tool)(partition="test")
    for metric, value in before.metrics.items():
        if isinstance(value, (int, float)) and np.isfinite(value):
            assert after.metrics[metric] == pytest.approx(value)


def test_agent_online_honors_disabled_drift_and_buffer_budget():
    session = _session()
    _run(session, "fit_online", {"backend": "sklearn", "drift_detector": "none",
                                  "buffer_size": 37, "n_init": 30})
    assert session.online_plan.config["drift_detector"] == "none"
    assert session.online_plan.config["buffer_size"] == 37


def test_agent_probabilistic_honors_requested_real_mapie_backend():
    pytest.importorskip("mapie")
    session = _session()
    frame = session.to_pandas().copy()
    frame["y"] = 2 * frame.x - .3 * frame.z
    session = Session.ingest(frame).set_roles(
        {"x": "feature", "z": "feature", "y": "target"}
    ).split(test_size=.2, validation_size=.2, random_state=31).ai.configure(provider="mock")
    _run(session, "fit_probabilistic", {"backend": "mapie", "estimator": "split",
                                         "task": "regression", "alpha": .2})
    assert session.probabilistic_plan.backend == "mapie"
    intervals = _run(session, "predict_interval", {"partition": "test", "alpha": .2})
    assert np.all(np.asarray(intervals.lower) <= np.asarray(intervals.upper))


@pytest.mark.parametrize("tool,args", [
    ("run_automl", {"n_trials": 1, "include_recipe_search": False}),
    ("fit_symbolic", {"source": "decision_tree", "task": "classification"}),
    ("fit_neuro_symbolic", {"task": "classification"}),
    ("fit_imitation", {"task": "classification", "action_column": "y"}),
    ("fit_anomaly", {"method": "isolation_forest"}),
    ("fit_ranker", {"method": "pointwise", "query_column": "group", "item_column": "item", "relevance_column": "y"}),
    ("fit_recommender", {"method": "popularity", "user_column": "group", "item_column": "item", "rating_column": "y"}),
])
def test_invalid_agent_backend_cannot_silently_fit_a_different_backend(tool, args):
    session = _session(domain="ranking")
    before = {key: value for key, value in vars(session).items() if key.endswith("_plan")}
    registry = build_default_registry()
    args = {**args, "backend": "unsupported_backend"}
    try:
        proposal = propose_tool_execution(tool, args, registry)
    except ValidationError as error:
        assert "backend" in str(error).lower()
    else:
        outcome = execute_tool(session, proposal, confirmed=True, registry=registry)
        assert not outcome.executed
        assert "backend" in outcome.error.lower()
    for key, value in before.items():
        assert vars(session)[key] is value


def test_agent_causal_estimate_recovers_known_randomized_treatment_effect():
    session = _session(domain="causal")
    _run(session, "fit_causal", {"backend": "native", "method": "t_learner", "bootstrap_samples": 20})
    effect = _run(session, "estimate_causal", {"partition": "test", "bootstrap_samples": 20})
    assert effect.ate == pytest.approx(1.6, abs=.3)
