# Quickstart: Imitation learning + Reinforcement learning

> **Install (GitHub 2.x):**
> `pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"`
> Optional Gymnasium path: `pip install "buildml[rl]"`
> Industry SB3 + imitation: `pip install "buildml[rl-industry]"`
> See [installation](../docs/installation.rst).

Behavioral cloning from demonstration tables, contextual bandits on logged
`(context, action, reward)` rows, and optional small Gymnasium loops: tabular
TD control (Q-learning / SARSA / Expected SARSA / Double Q-learning) and
REINFORCE-lite. **Not** a MuJoCo / robotics / multi-agent platform.

**Proof:** [imitation-cartpole-control](../proofs/imitation-cartpole-control/) (+ Tier C sklearn BC twin; Gymnasium optional via `buildml[rl]`).

Runnable mirror: [`examples/imitation_rl_loop.py`](../examples/imitation_rl_loop.py).
Deep guide: [imitation-rl-deep.md](imitation-rl-deep.md).

---

## Behavioral cloning (core)

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

fit = session.rl.fit_imitation()  # train-only BC
print(fit.task, fit.train_score)

pred = session.rl.predict_imitation(partition="test")
print(pred.actions[:5])

ev = session.rl.evaluate_imitation(partition="validation")
print(ev.metrics)

session.rl.save_imitation_bundle("artifacts/imitation_demo_bundle")
```

---

## Contextual bandit (core)

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
print(ev.offline, ev.metrics)  # DM / IPS: offline, not live A/B

session.rl.save_bundle("artifacts/rl_bandit_demo_bundle")
```

---

## Optional Gymnasium REINFORCE (`buildml[rl]`)

```python
# pip install "buildml[rl]"
from buildml import Session
import pandas as pd

# Session hosts the checkpointed env policy (tabular rows are not the training signal).
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
print(fit.train_metrics)

ev = session.rl.evaluate(n_episodes=20)
print(ev.offline, ev.metrics["mean_return"])

session.rl.save_bundle("artifacts/rl_gym_demo_bundle")
```

---

## Optional tabular Q-learning / SARSA (`buildml[rl]`)

The classical value-based starting point: an explicit `Q[s, a]` table, no
neural network. `q_learning`, `sarsa`, `expected_sarsa`, and
`double_q_learning` all run through `mode="tabular_q"`.

```python
# pip install "buildml[rl]"
from buildml import Session
import pandas as pd

session = (
    Session.ingest(pd.DataFrame({"a": [0.0, 1.0], "y": [0, 1]}))
    .set_roles({"a": "feature", "y": "target"})
    .split(test_size=0.5, random_state=0)
)

fit = session.rl.fit(
    mode="tabular_q",
    algorithm="q_learning",
    env_id="FrozenLake-v1",
    n_episodes=3_000,
    learning_rate=0.2,   # TD step size alpha
    gamma=0.99,
    epsilon=1.0,         # exploration start, decayed each episode
    epsilon_min=0.05,
    epsilon_decay=0.999,
)
print(fit.train_metrics["state_coverage"])

ev = session.rl.evaluate(n_episodes=100)
print(ev.offline, ev.metrics["mean_return"], ev.metrics["unseen_state_rate"])

# Q(s, a) per action for a few states
print(session.rl.act(observations=[0, 1, 2]).scores)

# Inspect what was actually learned
q_table = session.rl.plan.policy_.q_table
print(q_table.shape, session.rl.plan.policy_.greedy_policy_table())
```

Continuous (Box) observations such as CartPole are binned automatically:
`session.rl.fit(mode="tabular_q", env_id="CartPole-v1", n_bins=6, n_episodes=3_000)`.
Inspect `session.rl.plan.config["discretizer"]` to see the bounds used.

---

## Optional SB3 PPO (`buildml[rl-industry]`)

```python
# pip install "buildml[rl-industry]"
from buildml import Session
import pandas as pd

session = (
    Session.ingest(pd.DataFrame({"a": [0.0, 1.0], "y": [0, 1]}))
    .set_roles({"a": "feature", "y": "target"})
    .split(test_size=0.5, random_state=0)
)

fit = session.rl.fit(
    backend="industry",
    mode="gym_sb3",
    algorithm="ppo",
    env_id="CartPole-v1",
    total_timesteps=20_000,
)
print(fit.train_metrics)

ev = session.rl.evaluate(n_episodes=15)
print(ev.metrics["mean_return"])

session.rl.save_bundle("artifacts/rl_sb3_demo_bundle")
```

---

## Honesty checklist

| Claim | Reality in BuildML |
| --- | --- |
| BC / bandit fit | Train partition only |
| Bandit holdout scores | Offline DM / IPS (disclosed) |
| Gym path | Optional `buildml[rl]`; small discrete envs |
| SB3 industry | Optional `buildml[rl-industry]`; PPO/DQN/A2C |
| Robotics / MuJoCo | **Out of scope** |

---

## Next

IL+RL industry depth is shipped (Gymnasium / SB3 extras when installed).
