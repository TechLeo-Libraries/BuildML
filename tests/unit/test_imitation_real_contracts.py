"""Real optional imitation APIs must train and preserve action contracts."""
import numpy as np
import pandas as pd
import pytest

from buildml.core.errors import ValidationError
from buildml.rl.adapters.imitation_industry import fit_tabular_bc_mlp, fit_tabular_gail_lite


@pytest.fixture(autouse=True)
def real_imitation():
    pytest.importorskip("imitation")
    pytest.importorskip("stable_baselines3")


@pytest.mark.parametrize("seed", [613, 991])
def test_real_bc_learns_known_action_boundary_on_separate_observations(seed):
    x = np.random.default_rng(seed).normal(size=(200, 3))
    y = (x[:, 0] > 0).astype(int)
    model, score, disclosures, warnings = fit_tabular_bc_mlp(
        x[:160], y[:160], n_actions=2, n_epochs=30, batch_size=32,
        learning_rate=.01, random_state=seed)
    pred = model.predict(x[160:])
    assert pred.shape == (40,)
    assert set(pred).issubset({0, 1})
    assert np.mean(pred == y[160:]) >= .85
    assert score == pytest.approx(np.mean(model.predict(x[:160]) == y[:160]))
    assert any("learning_rate=0.01" in note for note in disclosures)
    assert not warnings
    np.testing.assert_array_equal(model.predict(x[160:]), pred)


def test_real_bc_caps_batch_size_to_available_demonstrations():
    x = np.random.default_rng(41).normal(size=(12, 2))
    y = (x[:, 0] > 0).astype(int)
    model, score, disclosures, _ = fit_tabular_bc_mlp(
        x, y, n_actions=2, n_epochs=2, batch_size=64, random_state=0)
    assert model.predict(x[:1]).shape == (1,)
    assert np.isfinite(score)
    assert any("batch_size=12 (requested 64)" in note for note in disclosures)


@pytest.mark.parametrize("width,n_actions,env", [(3, 2, "CartPole-v1"),
                                                (4, 3, "CartPole-v1"),
                                                (4, 2, "no-such-buildml-env")])
def test_gail_rejects_environment_incompatible_demonstrations(width, n_actions, env):
    x = np.zeros((64, width))
    with pytest.raises(ValidationError):
        fit_tabular_gail_lite(x, np.arange(64) % 2, env_id=env, n_actions=n_actions,
                             total_timesteps=128, random_state=0)


def test_real_gail_runs_a_bounded_environment_loop_and_returns_finite_action_score():
    x = np.random.default_rng(81).normal(scale=.03, size=(128, 4))
    y = (x[:, 2] > 0).astype(int)
    model, score, disclosures, warnings = fit_tabular_gail_lite(
        x, y, env_id="CartPole-v1", n_actions=2, total_timesteps=256, random_state=0,
        allow_variable_horizon=True)
    pred = model.predict(x[:16])
    assert pred.shape == (16,)
    assert set(pred).issubset({0, 1})
    assert score == pytest.approx(np.mean(model.predict(x) == y))
    assert np.isfinite(score)
    assert any("gail" in note.lower() for note in disclosures)
    assert any("explicitly accepted" in warning for warning in warnings)


def test_variable_horizon_gail_is_refused_by_default():
    x = np.random.default_rng(81).normal(scale=.03, size=(128, 4))
    with pytest.raises(ValidationError, match="variable episode lengths"):
        fit_tabular_gail_lite(x, (x[:, 2] > 0).astype(int), env_id="CartPole-v1", n_actions=2,
                             total_timesteps=256, random_state=0)


@pytest.mark.parametrize("method", ["bc_mlp", "gail_lite"])
def test_agent_imitation_preserves_requested_method_budget_and_trusted_bundle(method, tmp_path):
    from buildml import Session
    from buildml.ai.executor import execute_tool, propose_tool_execution
    from buildml.ai.tools import build_default_registry

    x = np.random.default_rng(713).normal(scale=.03, size=(128, 4))
    columns = ["cart_position", "cart_velocity", "pole_angle", "pole_velocity"]
    frame = pd.DataFrame(x, columns=columns)
    frame["action"] = np.where(x[:, 2] > 0, "right", "left")
    roles = {**{column: "feature" for column in columns}, "action": "target"}
    s = Session.ingest(frame).set_roles(roles).split(test_size=.2, random_state=7)
    options = {"backend": "industry", "method": method, "columns": columns,
               "env_id": "CartPole-v1", "n_epochs": 2, "total_timesteps": 256,
               "allow_variable_horizon": method == "gail_lite", "random_state": 0}
    registry = build_default_registry()
    proposal = propose_tool_execution("fit_imitation", options, registry)
    if method == "gail_lite":
        assert any("leak reward information" in warning for warning in proposal.warnings)
    outcome = execute_tool(s, proposal, confirmed=True, registry=registry)
    assert outcome.executed, outcome.error
    config = dict(s.imitation_plan.config)
    assert config["method"] == method
    assert config["n_epochs"] == 2
    assert config["total_timesteps"] == 256
    assert config["allow_variable_horizon"] == (method == "gail_lite")
    before = s.rl.predict_imitation(partition="test")
    assert set(before.actions).issubset({"left", "right"})
    path = s.rl.save_imitation_bundle(tmp_path / method)
    restored = Session.ingest(frame).set_roles(roles).split(test_size=.2, random_state=7)
    with pytest.raises(ValidationError, match="trust"):
        restored.rl.load_imitation_bundle(path)
    restored.rl.load_imitation_bundle(path, trusted=True)
    assert restored.rl.predict_imitation(partition="test").actions == before.actions
    assert restored.imitation_plan.config == config


def test_seeded_gail_repeats_policy_weights_in_the_same_cpu_profile():
    x = np.random.default_rng(81).normal(scale=.03, size=(128, 4))
    y = (x[:, 2] > 0).astype(int)
    options = dict(env_id="CartPole-v1", n_actions=2, total_timesteps=256,
                   random_state=0, allow_variable_horizon=True)
    first, *_ = fit_tabular_gail_lite(x, y, **options)
    second, *_ = fit_tabular_gail_lite(x, y, **options)
    for name, weights in first.model.policy.state_dict().items():
        np.testing.assert_allclose(weights.detach().cpu().numpy(),
                                   second.model.policy.state_dict()[name].detach().cpu().numpy(),
                                   rtol=1e-6, atol=1e-7, err_msg=name)
