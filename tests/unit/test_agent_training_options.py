"""Valid caller-selected budgets, methods and devices must reach Session."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.executor import execute_tool, propose_tool_execution
from buildml.ai.tools import build_default_registry


@pytest.mark.parametrize("tool,options", [
    ("run_automl", {"time_budget": 7.5, "ensemble_mode": "stacking", "include_industry_families": True}),
    ("fit_probabilistic", {"n_estimators": 7, "learning_rate": .031}),
    ("fit_symbolic", {"method": "boosted_rules", "verify_constraints": True}),
    ("fit_neuro_symbolic", {"torch_method": "neural_additive_lite", "torch_epochs": 2, "device": "cpu"}),
    ("fit_recommender", {"n_iterations": 3, "lightfm_epochs": 2, "user_feature_columns": ["region"]}),
    ("fit_ranker", {"n_estimators": 7, "epochs": 2, "hidden_dim": 12, "learning_rate": .031, "device": "cpu"}),
])
def test_agent_training_preserves_explicit_options_at_session_boundary(tool, options, monkeypatch):
    captured = {}
    # Isolate dispatch from expensive optional fitting. Numerical integration
    # tests exercise actual models separately; this checks the caller contract.
    summary = SimpleNamespace(best_family="ridge", best_recipe_strategy="none",
                              estimator_name="test", task="classification", conformal=False,
                              n_fit_rows=10, n_conformal_calib_rows=0, source="test", n_rules=1,
                              provenance="test", mode="test", base_estimator_name="test",
                              rule_provenance="test", method="test", n_users=2, n_items=3,
                              n_train_interactions=10, n_train_queries=2, n_train_rows=10, n_features=2)

    def observe(self, **kwargs):
        captured.update(kwargs)
        return summary

    monkeypatch.setattr(Session, tool, observe)
    s = Session.ingest(pd.DataFrame({"x": [1., 2.], "y": [0, 1]}))
    registry = build_default_registry()
    proposal = propose_tool_execution(tool, options, registry)
    outcome = execute_tool(s, proposal, confirmed=True, registry=registry)
    assert outcome.executed, outcome.error
    assert {key: captured.get(key) for key in options} == options


def run(s, tool, options):
    registry = build_default_registry()
    proposal = propose_tool_execution(tool, options, registry)
    outcome = execute_tool(s, proposal, confirmed=True, registry=registry)
    assert outcome.executed, outcome.error
    return outcome


def test_real_ngboost_agent_fit_uses_requested_tree_budget_and_learning_rate():
    pytest.importorskip("ngboost")
    rng = np.random.default_rng(82)
    x = rng.normal(size=100)
    s = (Session.ingest(pd.DataFrame({"x": x, "y": 2 * x + rng.normal(scale=.2, size=100)}))
         .set_roles({"x": "feature", "y": "target"}).split(test_size=.2, random_state=7))
    run(s, "fit_probabilistic", {"backend": "ngboost", "estimator": "ngboost_regressor",
                                 "task": "regression", "conformal": False,
                                 "n_estimators": 7, "learning_rate": .031})
    estimator = s.probabilistic_plan.estimator_
    assert estimator.n_estimators == 7
    assert estimator.learning_rate == pytest.approx(.031)
    assert len(estimator.base_models) == 7
    assert np.isfinite(s.probabilistic.evaluate(partition="test").metrics["rmse"])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_neural_symbolic_agent_honors_method_device_and_epoch_budget(device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA runtime unavailable")
    rng = np.random.default_rng(37)
    x = rng.normal(size=(100, 2))
    s = (Session.ingest(pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "y": (x[:, 0] > 0).astype(int)}))
         .set_roles({"a": "feature", "b": "feature", "y": "target"})
         .split(test_size=.2, random_state=7))
    run(s, "fit_neuro_symbolic", {"backend": "torch", "mode": "constraint_overlay",
                                  "torch_method": "neural_additive_lite", "torch_epochs": 2,
                                  "device": device})
    plan = s.neuro_symbolic_plan
    assert plan.torch_method == "neural_additive_lite"
    assert plan.estimator_.epochs == 2
    assert next(plan.estimator_.module_.parameters()).device.type == device
    assert np.isfinite(s.symbolic.evaluate_neuro(partition="test").metrics["accuracy"])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_neural_ranker_agent_honors_architecture_device_and_training_budget(device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA runtime unavailable")
    rng = np.random.default_rng(781)
    frame = pd.DataFrame({"query": np.repeat(np.arange(12), 6), "item": np.tile(np.arange(6), 12),
                          "x": rng.normal(size=72), "y": np.tile(np.arange(6), 12).astype(float)})
    s = (Session.ingest(frame).set_roles({"query": "group", "item": "id", "x": "feature", "y": "target"})
         .group_split(test_size=.25, random_state=7))
    run(s, "fit_ranker", {"backend": "torch", "method": "listwise_lite", "query_column": "query",
                          "item_column": "item", "epochs": 2, "hidden_dim": 12,
                          "learning_rate": .031, "device": device})
    model = s.ranker_plan.estimator_
    assert model.epochs == 2 and model.hidden_dim == 12
    assert model.learning_rate == pytest.approx(.031)
    assert next(model.model_.parameters()).device.type == device
    assert np.isfinite(s.ranking.evaluate(k=3).metrics["ndcg_at_k"])
