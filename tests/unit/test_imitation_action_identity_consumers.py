"""Real BC and GAIL policies retain the trained discrete-action identities."""

import itertools

import numpy as np
import pytest
from test_rl_action_identity_consumers import (
    _classification_metrics,
    _frame,
    _scalar,
    _session,
)


@pytest.fixture(scope="module", params=list(itertools.product(
    ["bc_mlp", "gail_lite"],
    [["001", "002"], ["1.00", "1.0"], [False, True], [2, 10],
     [np.float32(1.1), np.float32(2.2)]],
)))
def fitted_policy(request):
    pytest.importorskip("imitation")
    pytest.importorskip("stable_baselines3")
    method, vocabulary = request.param
    frame = _frame(vocabulary)
    # CartPole needs four observation features and two encoded discrete actions.
    frame["v"] = .01*frame.x
    frame["w"] = .01*frame.z
    session = _session(frame)
    session.rl.fit_imitation(backend="industry", method=method, task="classification",
                             columns=["x", "z", "v", "w"], env_id="CartPole-v1",
                             n_epochs=2, total_timesteps=128,
                             allow_variable_horizon=method == "gail_lite", random_state=23)
    return session, frame, sorted(map(_scalar, vocabulary), key=str)


def _weights(plan):
    wrapper = plan.estimator_
    policy = wrapper.model.policy if wrapper.method == "gail_lite" else wrapper.model
    return policy, {k: v.detach().cpu().numpy().copy() for k, v in policy.state_dict().items()}


def _numpy_policy_codes(policy, x):
    # Independent forward evaluation of the real SB3 MLP's affine/tanh layers.
    values = np.asarray(x, dtype=np.float32)
    for layer in policy.mlp_extractor.policy_net:
        name = type(layer).__name__
        if name == "Linear":
            values = values@layer.weight.detach().cpu().numpy().T + layer.bias.detach().cpu().numpy()
        else:
            assert name == "Tanh"
            values = np.tanh(values)
    logits = values@policy.action_net.weight.detach().cpu().numpy().T
    logits += policy.action_net.bias.detach().cpu().numpy()
    return logits.argmax(axis=1)


@pytest.mark.parametrize("partition", ["validation", "test"])
@pytest.mark.parametrize("replay", [False, True])
def test_real_neural_actions_metrics_and_replay_match_numpy_policy(fitted_policy, partition, replay, tmp_path):
    original, frame, vocabulary = fitted_policy
    session = original
    if replay:
        original.rl.save_imitation_bundle(tmp_path/"neural")
        session = _session(frame).rl.load_imitation_bundle(tmp_path/"neural", trusted=True)
    plan = session.imitation_plan
    policy, before = _weights(plan)
    rows = frame.iloc[list(session.split_plan.indices_for(partition))]
    codes = _numpy_policy_codes(policy, rows[list(plan.columns)].to_numpy())
    expected = [vocabulary[i] for i in codes]
    prediction = session.rl.predict_imitation(partition=partition)
    result = session.rl.evaluate_imitation(partition=partition)
    assert list(prediction.actions) == expected
    assert all(type(a) is type(b) for a, b in zip(prediction.actions, expected, strict=True))
    assert result.metrics == pytest.approx(_classification_metrics(rows.action, expected))
    _, after = _weights(plan)
    for name in before:
        np.testing.assert_array_equal(after[name], before[name])
