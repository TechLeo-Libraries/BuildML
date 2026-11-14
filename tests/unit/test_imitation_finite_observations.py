"""Neural imitation policies require finite observations at fit and replay."""

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError


def _session(frame):
    return (
        Session.ingest(frame)
        .set_roles(
            {"x": "feature", "z": "feature", "v": "feature", "w": "feature", "action": "target"}
        )
        .inject_split(train_indices=list(range(32)), test_indices=list(range(32, 40)))
    )


@pytest.fixture(scope="module")
def policy_bundle(tmp_path_factory):
    pytest.importorskip("imitation")
    pytest.importorskip("stable_baselines3")
    x = np.linspace(-1.0, 1.0, 40)
    frame = pd.DataFrame(
        {"x": x, "z": np.sin(x), "v": x**2, "w": x**3, "action": np.where(x > 0, 10, 2)}
    )
    source = _session(frame)
    source.rl.fit_imitation(
        backend="industry",
        method="bc_mlp",
        task="classification",
        columns=["x", "z", "v", "w"],
        n_epochs=1,
        random_state=23,
    )
    root = source.rl.save_imitation_bundle(tmp_path_factory.mktemp("imitation") / "policy")
    actions = source.rl.predict_imitation(partition="test").actions
    return frame, root, actions


@pytest.mark.parametrize(
    "bad", [np.inf, -np.inf, 1e50], ids=["positive-inf", "negative-inf", "overflow"]
)
def test_restored_policy_rejects_nonfinite_observations(policy_bundle, bad):
    frame, root, actions = policy_bundle
    clean = _session(frame).rl.load_imitation_bundle(root, trusted=True)
    assert clean.rl.predict_imitation(partition="test").actions == actions

    invalid = frame.copy()
    invalid.loc[32, "x"] = bad
    replay = _session(invalid).rl.load_imitation_bundle(root, trusted=True)
    plan, history = replay.imitation_plan, list(replay.history)
    with pytest.raises(ValidationError, match="finite float32 observations"):
        replay.rl.predict_imitation(partition="test")
    with pytest.raises(ValidationError, match="finite float32 observations"):
        replay.rl.evaluate_imitation(partition="test")
    assert replay.imitation_plan is plan
    assert replay.history == history


@pytest.mark.parametrize("method", ["bc_mlp", "gail_lite"])
def test_refit_rejects_overflow_before_replacing_restored_policy(policy_bundle, method):
    frame, root, actions = policy_bundle
    invalid = frame.copy()
    invalid.loc[0, "x"] = 1e50
    session = _session(invalid).rl.load_imitation_bundle(root, trusted=True)
    plan = session.imitation_plan
    previous = session.rl.predict_imitation(partition="test")
    assert previous.actions == actions
    history = list(session.history)
    with pytest.raises(ValidationError, match="finite float32 observations"):
        session.rl.fit_imitation(
            backend="industry",
            method=method,
            task="classification",
            columns=["x", "z", "v", "w"],
            env_id="CartPole-v1",
            n_epochs=1,
            total_timesteps=128,
            allow_variable_horizon=True,
            random_state=23,
        )
    assert session.imitation_plan is plan
    assert session.history == history
