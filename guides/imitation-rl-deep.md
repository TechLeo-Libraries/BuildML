# Imitation and RL deep

```bash
pip install buildml
# Gymnasium REINFORCE / tabular Q: pip install "buildml[rl]"
# SB3 PPO/DQN/A2C and imitation BC/GAIL-lite: pip install "buildml[rl-industry]"
```

Two surfaces live on `session.rl`. `fit_imitation` clones actions from a
demonstration table. `fit` is a contextual bandit on logged
(context, action, reward) rows, or a Gymnasium loop when you ask for
one.

`session.rl.fit()` with no extra knobs uses `algorithm="linucb"`, which
resolves to the **sklearn contextual bandit** even when
`buildml[rl-industry]` is installed. Gymnasium and Stable-Baselines3 are
opt-in via `mode=` / `algorithm=` / `backend=`. `fit_imitation()` with
`backend=None` and `method=None` stays **sklearn** behavioral cloning.
Industry MLP BC is `method="bc_mlp"` (or `backend="industry"`).

Gym loops still need a Session split. They do not train on those tabular
rows; the env is the signal. This is not robotics, not MuJoCo, and not
batch offline RL (CQL / IQL / Decision Transformer).

Short on-ramp: [imitation + RL quickstart](quickstart-imitation-rl.md).
Proof: [imitation-cartpole-control](../proofs/imitation-cartpole-control/).

## Behavioral cloning

Demonstration rows on train. `action_column` defaults to the Session
target. Classification estimators: `logistic_regression` (default) or
`hist_gradient_boosting`. Regression: `ridge` (default) or
`hist_gradient_boosting_regressor`. Holdout metrics compare predicted
actions to held-out demonstration actions. Validation and test never
update the policy.

Industry methods (`bc_mlp`, `gail_lite`) are classification-only.
`gail_lite` needs `env_id=` whose observation dim matches the demo
features.

```python
import numpy as np
import pandas as pd

from buildml import Session

rng = np.random.default_rng(0)
x = rng.normal(size=(220, 2))
action = (x[:, 0] + 0.3 * x[:, 1] > 0).astype(int)
frame = pd.DataFrame({"s0": x[:, 0], "s1": x[:, 1], "action": action})

session = (
    Session.ingest(frame)
    .set_roles({"s0": "feature", "s1": "feature", "action": "target"})
    .split(test_size=0.2, validation_size=0.2, random_state=0, stratify=True)
    .scale(method="standard")
)

fit = session.rl.fit_imitation()
print(fit.task, fit.train_score)

pred = session.rl.predict_imitation(partition="test")
print(pred.actions[:5])

ev = session.rl.evaluate_imitation(partition="validation")
print(ev.metrics)

session.rl.save_imitation_bundle("artifacts/imitation_bundle")
```

`predict_imitation` defaults to test. `evaluate_imitation` defaults to
validation. This is not inverse RL and not DAgger.

## Contextual bandit (what `fit()` actually runs)

`mode="contextual_bandit"` on sklearn. Algorithms:

| Algorithm | Behavior |
| --- | --- |
| `linucb` (default) | Disjoint LinUCB: per-arm linear model plus UCB bonus (`alpha=1.0`) |
| `epsilon_greedy` | Per-arm Ridge reward models plus ε exploration (`epsilon=0.1`) |
| `softmax` | Softmax over predicted rewards (`temperature=1.0`) |

`action_column` defaults to the Session target. Reward is
`reward_column=` if you pass it, else a column literally named
`reward`, else the target when the action already consumed a different
column. If the target is the action, you must pass `reward_column`.
Nulls in the reward column are refused.

`evaluate` for bandits is **offline**: direct method, IPS, action match
rate, mean logged reward on match. Those are not live A/B lifts.
`act` defaults to test, `deterministic=True`.

```python
import numpy as np
import pandas as pd

from buildml import Session

rng = np.random.default_rng(1)
n = 240
ctx = rng.normal(size=(n, 2))
arm = (ctx[:, 0] > 0).astype(int)
reward = np.where(arm == (ctx[:, 0] > 0).astype(int), 1.0, 0.0)
frame = pd.DataFrame(
    {"c0": ctx[:, 0], "c1": ctx[:, 1], "arm": arm, "reward": reward}
)

session = (
    Session.ingest(frame)
    .set_roles(
        {
            "c0": "feature",
            "c1": "feature",
            "arm": "target",
            "reward": "feature",
        }
    )
    .split(test_size=0.2, validation_size=0.2, random_state=0)
    .scale(method="standard", columns=["c0", "c1"])
)

fit = session.rl.fit(
    mode="contextual_bandit",
    algorithm="linucb",
    action_column="arm",
    reward_column="reward",
)
print(fit.n_arms, fit.train_metrics)

act = session.rl.act(partition="test", deterministic=True)
print(act.actions[:5])

ev = session.rl.evaluate(partition="validation")
print(ev.offline, ev.metrics)
```

## Gymnasium REINFORCE (`buildml[rl]`)

`backend="native"`, `mode="gym_reinforce"`. Linear softmax REINFORCE on
small discrete envs. Default `env_id` is `"CartPole-v1"`, `n_episodes=200`,
`max_steps=500`, `learning_rate=0.01`, `gamma=0.99`. Passing
`mode="gym_reinforce"` (or a native policy-gradient algorithm) is what
selects this path: the mixin `algorithm="linucb"` default would otherwise
keep you on the bandit.

`evaluate` rolls out episodes (`offline=False`) and reports mean/std
return.

```python
from buildml import Session
import pandas as pd

session = (
    Session.ingest(pd.DataFrame({"a": [0.0, 1.0], "y": [0, 1]}))
    .set_roles({"a": "feature", "y": "target"})
    .split(test_size=0.5, random_state=0)
)

fit = session.rl.fit(
    mode="gym_reinforce",
    env_id="CartPole-v1",
    n_episodes=200,
    learning_rate=0.01,
)
ev = session.rl.evaluate(n_episodes=20)
print(ev.offline, ev.metrics["mean_return"])
```

## Tabular TD control (`buildml[rl]`)

`mode="tabular_q"`: an explicit `Q[s, a]` table, no network. Passing
only `algorithm="sarsa"` (no `mode=`) routes here. If you pass
`mode="tabular_q"` and leave the mixin `algorithm="linucb"` alone, the
resolver treats linucb as unset and uses `q_learning`.

| Algorithm | Target | Family |
| --- | --- | --- |
| `q_learning` (mode default) | `r + γ max_a' Q(s', a')` | Off-policy TD control |
| `sarsa` | `r + γ Q(s', a')` with the behaviour policy's `a'` | On-policy |
| `expected_sarsa` | `r + γ Σ_a' π(a'\|s') Q(s', a')` | On-policy, lower variance |
| `double_q_learning` | Cross-evaluated `Q_A` / `Q_B` | Off-policy, no max bias |

```python
session.rl.fit(
    mode="tabular_q",
    algorithm="q_learning",
    env_id="FrozenLake-v1",
    n_episodes=3_000,
    learning_rate=0.2,
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.999,
)
ev = session.rl.evaluate(n_episodes=100)
print(ev.metrics["mean_return"], ev.metrics["unseen_state_rate"])
print(session.rl.act(observations=[0, 1, 2]).scores)
```

`Discrete` observation spaces index directly. `Box` spaces are binned
uniformly into `n_bins` (default 8) per dimension. Bounds come from the
declared space where finite, else from a seeded random-policy probe.
`session.rl.plan.config["discretizer"]` records `n_states`, per-dimension
`low`/`high`, and `bound_sources`. Tables above 500k states are refused.
`MultiDiscrete` observation spaces are refused. Continuous action spaces
are refused on the lite paths.

This is an **online** env loop. Off-policy TD control is not batch
offline RL. `state_coverage` (fit) and `unseen_state_rate` (eval) say
how much of the table was actually visited.

## Stable-Baselines3 (`buildml[rl-industry]`)

`backend="industry"`, `mode="gym_sb3"`. Algorithms: `ppo` (default when
this mode is selected), `dqn`, `a2c`. Default `total_timesteps` is
20_000. Small discrete sims (CartPole-class). Not multi-agent, not AV,
not Ray RLlib.

```python
session.rl.fit(
    backend="industry",
    mode="gym_sb3",
    algorithm="ppo",
    env_id="CartPole-v1",
    total_timesteps=25_000,
)
ev = session.rl.evaluate(n_episodes=20)
print(ev.metrics["mean_return"])
```

Benchmark: `python benchmarks/rl/policy_return.py`.

## Bundles

Imitation and RL are different artifacts. Neither is inside a Session
checkpoint.

| Artifact | Contains |
| --- | --- |
| `buildml.imitation_bundle.v1` | `ImitationPlan` |
| `buildml.rl_bundle.v1` | `RlPlan` (bandit, Q-table, REINFORCE, or SB3) |

`save_imitation_bundle` / `load_imitation_bundle` for cloning.
`save_bundle` / `load_bundle` for `fit`. `trusted=True` only for a file
you made.

## When it refuses

| What you see | What happened |
| --- | --- |
| No split | BC or bandit `fit` before `split` (Gym paths still need a split) |
| Bandit needs `reward_column` | Target is already the action and there is no `reward` column |
| Reward has nulls | Impute or drop before `fit` |
| `MissingExtraError` for `rl` | You asked for Gymnasium native modes without `buildml[rl]` |
| `MissingExtraError` for `rl-industry` | You asked for SB3 or industry imitation without that extra |
| Industry imitation on regression | `bc_mlp` / `gail_lite` need discrete actions |
| `gail_lite` obs dim mismatch | Demo features do not match `env_id` |
| sklearn with a gym mode | `backend="sklearn"` is bandit-only |
| `gym_reinforce` plus a tabular algorithm | Use `mode="tabular_q"` instead |
| More than 500k tabular states | Lower `n_bins` or pick a function-approx path |
| Continuous / MultiDiscrete actions or obs | Lite Gym paths do not take them |

[Imitation + RL quickstart](quickstart-imitation-rl.md) ·
[imitation-cartpole-control](../proofs/imitation-cartpole-control/) ·
[Artifacts](artifacts-checkpoints-bundles.md)
