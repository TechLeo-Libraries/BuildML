# Imitation + RL: deep guide

> **Install:** core BC/bandit need no extra. Gymnasium REINFORCE + tabular
> Q-learning/SARSA: `pip install "buildml[rl]"`.
> Industry SB3 + imitation BC/GAIL: `pip install "buildml[rl-industry]"`.

This guide covers leakage, offline metrics, bundles, capability matrix, and honesty
boundaries for `buildml.rl` / Session IL+RL APIs. Pair with
[quickstart-imitation-rl.md](quickstart-imitation-rl.md).

---

## Capability matrix

```python
from buildml.rl import rl_capability_matrix

print(rl_capability_matrix())
```

Backends default when installed (`backend=None`):

| Surface | sklearn (core) | native (`buildml[rl]`) | industry (`buildml[rl-industry]`) |
| --- | --- | --- | --- |
| Imitation | BC estimators |: | `bc_mlp`, `gail_lite` |
| RL | contextual bandit | `gym_reinforce`, `tabular_q` | `gym_sb3` (PPO/DQN/A2C) |

`list_rl_algorithms(backend="native", mode="tabular_q")` returns the tabular
TD-control family; `mode="gym_reinforce"` returns the policy-gradient entry.

---

## Surfaces

| API | Role |
| --- | --- |
| `fit_imitation` / `predict_imitation_action` / `evaluate_imitation` | Behavioral cloning |
| `save_imitation_bundle` / `load_imitation_bundle` | `buildml.imitation_bundle.v1` |
| `fit_rl` / `act_rl` / `evaluate_rl` | Contextual bandit, tabular TD control, REINFORCE-lite, or SB3 |
| `save_rl_bundle` / `load_rl_bundle` | `buildml.rl_bundle.v1` |

Package: `buildml.rl` (lazy imports). Core stays numpy/pandas/sklearn.

---

## Behavioral cloning

- `backend="sklearn"` (default when industry absent): demonstration tables on train.
- `backend="industry"`: MLP BC via imitation+SB3 (`method="bc_mlp"`) or small-budget
  GAIL (`method="gail_lite"`, requires `env_id=` with compatible demo obs dim).
- Classification estimators (sklearn): `logistic_regression`, `hist_gradient_boosting`.
- Regression estimators (sklearn): `ridge`, `hist_gradient_boosting_regressor`.
- Holdout metrics compare predicted actions to demonstration actions.
- **Not** inverse RL; **not** DAgger by default; **not** batch offline RL.

Leakage: `assert_can_fit("train")` / `assert_fit_partition`. Validation/test never
update the cloning policy.

---

## Contextual bandits

Algorithms (`backend="sklearn"`, `mode="contextual_bandit"`):

| Algorithm | Behavior |
| --- | --- |
| `linucb` | Disjoint LinUCB (per-arm linear + UCB bonus) |
| `epsilon_greedy` | Per-arm Ridge reward models + ε exploration |
| `softmax` | Softmax over predicted rewards |

### Offline evaluation (disclosed)

`evaluate_rl` for bandits sets `offline=True` and reports DM / IPS / action_match_rate.
These are **not** online A/B lifts.

---

## Gymnasium REINFORCE-lite (`buildml[rl]`)

- `backend="native"`, `mode="gym_reinforce"`
- Linear softmax REINFORCE teaching loop on discrete envs
- `evaluate_rl` rolls out episodes (`offline=False`)

---

## Tabular TD control: Q-learning family (`buildml[rl]`)

`backend="native"`, `mode="tabular_q"`. The foundational value-based methods:
an explicit `Q[s, a]` table updated by bootstrapping, with no neural network.

| Algorithm | Target | Family |
| --- | --- | --- |
| `q_learning` | `r + γ max_a' Q(s', a')` | Off-policy TD control |
| `sarsa` | `r + γ Q(s', a')` with the behaviour policy's `a'` | On-policy |
| `expected_sarsa` | `r + γ Σ_a' π(a'\|s') Q(s', a')` | On-policy, lower variance |
| `double_q_learning` | Cross-evaluated `Q_A` / `Q_B` | Off-policy, no max bias |

```python
session.fit_rl(
    mode="tabular_q",
    algorithm="q_learning",
    env_id="FrozenLake-v1",
    n_episodes=3_000,
    learning_rate=0.2,      # TD step size alpha
    gamma=0.99,
    epsilon=1.0,            # exploration start
    epsilon_min=0.05,
    epsilon_decay=0.999,
)
ev = session.evaluate_rl(n_episodes=100)
print(ev.metrics["mean_return"], ev.metrics["unseen_state_rate"])
```

Passing only `algorithm="sarsa"` (no `mode=`) routes to `tabular_q`
automatically. `act_rl(observations=...)` returns per-action `Q(s, a)` as scores.

**State discretization.** `Discrete` observation spaces (FrozenLake, Taxi,
CliffWalking) index directly. `Box` spaces (CartPole) are binned uniformly into
`n_bins` per dimension; bounds come from the declared space where finite and
from a seeded random-policy probe where not. `RlPlan.config["discretizer"]`
records `n_states`, per-dimension `low`/`high`, and `bound_sources`. Tables
above 500k states are refused with a pointer to function approximation.

**Honesty.** `tabular_q` is an *online* env loop. Off-policy TD control is not
the same thing as batch offline RL: CQL / IQL / Decision Transformer remain out
of scope. `state_coverage` (fit) and `unseen_state_rate` (eval) disclose how
much of the table was actually learned.

---

## Stable-Baselines3 industry path (`buildml[rl-industry]`)

- `backend="industry"`, `mode="gym_sb3"`
- Algorithms: `ppo`, `dqn`, `a2c`
- DQN is `tabular_q` scaled up: the Q-table becomes a network, plus a replay
  buffer and a target network
- Honest small-env sim (CartPole-class): **not** MuJoCo, robotics, AV, multi-agent
- Ray RLlib intentionally omitted (prefer clean SB3 adapter)
- **Offline RL** (CQL/IQL/batch RL) is out of scope and disclosed in the capability matrix

```python
session.fit_rl(
    backend="industry",
    mode="gym_sb3",
    algorithm="ppo",
    env_id="CartPole-v1",
    total_timesteps=25_000,
)
ev = session.evaluate_rl(n_episodes=20)
print(ev.metrics["mean_return"])
```

Benchmark: `python benchmarks/rl/policy_return.py` (BC baseline vs SB3 return).

---

## Bundles vs checkpoints

| Artifact | Contains | Does not contain |
| --- | --- | --- |
| Session checkpoint | data, roles, splits, history | IL/RL policies |
| `buildml.imitation_bundle.v1` | `ImitationPlan` | dataset / splits |
| `buildml.rl_bundle.v1` | `RlPlan` (bandit, Q-table, REINFORCE, or SB3) | dataset / splits |

---

## Walkthrough / AI / catalog

- Walkthrough exposes `imitation_status` and `rl_status` with `capability_matrix`.
- AI allowlist includes fit/predict/evaluate/act + bundle tools with `backend=` knobs.
- Concept keys: `imitation-behavioral-cloning`, `rl-contextual-bandit`,
  `rl-offline-metrics`, `rl-gym-reinforce`, `rl-tabular-q-learning`,
  `rl-sarsa-on-policy`, `rl-state-discretization`, `rl-sb3-industry`, bundle
  boundaries.

---

## Failure modes

- Fitting BC/bandit without a split → leakage error
- Bandit without `reward_column` → validation error
- Gym/SB3 without extras → `MissingExtraError`
- `gail_lite` without matching `env_id` obs dim → validation error
- Continuous-action Gym envs → unsupported in lite paths (including `tabular_q`)
- `tabular_q` with `n_bins ** obs_dim` above the 500k state cap → validation error
- `tabular_q` on a MultiDiscrete observation space → validation error

---

