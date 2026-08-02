# Imitation + RL — deep guide

> **Install:** core BC/bandit need no extra. Gymnasium REINFORCE: `pip install "buildml[rl]"`.
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
| Imitation | BC estimators | — | `bc_mlp`, `gail_lite` |
| RL | contextual bandit | `gym_reinforce` | `gym_sb3` (PPO/DQN/A2C) |

---

## Surfaces

| API | Role |
| --- | --- |
| `fit_imitation` / `predict_imitation_action` / `evaluate_imitation` | Behavioral cloning |
| `save_imitation_bundle` / `load_imitation_bundle` | `buildml.imitation_bundle.v1` |
| `fit_rl` / `act_rl` / `evaluate_rl` | Contextual bandit, REINFORCE-lite, or SB3 |
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

## Stable-Baselines3 industry path (`buildml[rl-industry]`)

- `backend="industry"`, `mode="gym_sb3"`
- Algorithms: `ppo`, `dqn`, `a2c`
- Honest small-env sim (CartPole-class) — **not** MuJoCo, robotics, AV, multi-agent
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
| `buildml.rl_bundle.v1` | `RlPlan` (bandit, REINFORCE, or SB3) | dataset / splits |

---

## Walkthrough / AI / catalog

- Walkthrough exposes `imitation_status` and `rl_status` with `capability_matrix`.
- AI allowlist includes fit/predict/evaluate/act + bundle tools with `backend=` knobs.
- Concept keys: `imitation-behavioral-cloning`, `rl-contextual-bandit`,
  `rl-offline-metrics`, `rl-gym-reinforce`, `rl-sb3-industry`, bundle boundaries.

---

## Failure modes

- Fitting BC/bandit without a split → leakage error
- Bandit without `reward_column` → validation error
- Gym/SB3 without extras → `MissingExtraError`
- `gail_lite` without matching `env_id` obs dim → validation error
- Continuous-action Gym envs → unsupported in lite paths

---

## Tracker

IL+RL industry depth is **PASS** (R6.11). R6 sweep complete.
