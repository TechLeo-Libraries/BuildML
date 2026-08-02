# Imitation + RL — deep guide

> **Install:** core BC/bandit need no extra. Gymnasium path: `pip install "buildml[rl]"`.

This guide covers leakage, offline metrics, bundles, and honesty boundaries for
`buildml.rl` / Session IL+RL APIs. Pair with [quickstart-imitation-rl.md](quickstart-imitation-rl.md).

---

## Surfaces

| API | Role |
| --- | --- |
| `fit_imitation` / `predict_imitation_action` / `evaluate_imitation` | Behavioral cloning |
| `save_imitation_bundle` / `load_imitation_bundle` | `buildml.imitation_bundle.v1` |
| `fit_rl` / `act_rl` / `evaluate_rl` | Contextual bandit or `gym_reinforce` |
| `save_rl_bundle` / `load_rl_bundle` | `buildml.rl_bundle.v1` |

Package: `buildml.rl` (lazy imports). Core stays numpy/pandas/sklearn.

---

## Behavioral cloning

- Demonstrations = train rows: state features → action column (default: Dataset target).
- Classification estimators: `logistic_regression`, `hist_gradient_boosting`.
- Regression estimators: `ridge`, `hist_gradient_boosting_regressor`.
- Holdout metrics compare predicted actions to demonstration actions
  (`accuracy`/`macro_f1` or `rmse`/`mae`/`r2`).
- **Not** inverse RL; **not** DAgger (compounding-error mitigation) by default.

Leakage: `assert_can_fit("train")` / `assert_fit_partition`. Validation/test never
update the cloning policy.

---

## Contextual bandits

Algorithms:

| Algorithm | Behavior |
| --- | --- |
| `linucb` | Disjoint LinUCB (per-arm linear + UCB bonus) |
| `epsilon_greedy` | Per-arm Ridge reward models + ε exploration |
| `softmax` | Softmax over predicted rewards |

Required columns:

- Context features (numeric; reduce components honored when present)
- `action_column` (discrete arms; defaults to target)
- `reward_column` (numeric). If omitted, a column named `reward` is used when present.

### Offline evaluation (disclosed)

`evaluate_rl` for bandits sets `offline=True` and reports:

- **direct_method** — mean predicted reward under π(x)
- **ips** — inverse propensity scoring using a train-fitted π_b(a\|x)
- **action_match_rate** / **mean_logged_reward_on_match**

These are **not** online A/B lifts. Confounding and positivity failures can bias IPS.

---

## Gymnasium REINFORCE-lite

- Extra: `buildml[rl]` → `gymnasium`
- Policy: linear softmax; update: REINFORCE with returns-to-go + mean baseline
- Requires discrete `action_space.n` and Box-like observations
- Session hosts the checkpointed policy; tabular partitions are not the gradient signal
- `evaluate_rl` rolls out episodes (`offline=False`)

Honesty: teaching / small-env path — **not** MuJoCo, robotics stacks, or multi-agent sims.

---

## Bundles vs checkpoints

| Artifact | Contains | Does not contain |
| --- | --- | --- |
| Session checkpoint | data, roles, splits, history | IL/RL policies |
| `buildml.imitation_bundle.v1` | `ImitationPlan` | dataset / splits |
| `buildml.rl_bundle.v1` | `RlPlan` | dataset / splits |

Reload workflow with `checkpoint_load`; reload policies with
`load_imitation_bundle` / `load_rl_bundle`.

---

## Walkthrough / AI / catalog

- Walkthrough exposes `imitation_status` and `rl_status`.
- AI allowlist includes fit/predict/evaluate/act + bundle tools.
- Concept keys: `imitation-behavioral-cloning`, `rl-contextual-bandit`,
  `rl-offline-metrics`, `rl-gym-reinforce`, bundle boundaries.

---

## Failure modes

- Fitting BC/bandit without a split → leakage error
- Bandit without `reward_column` (and no `reward` column) → validation error
- Gym without `buildml[rl]` → `MissingExtraError`
- Continuous-action Gym envs → unsupported in this lite path

---

## Tracker

IL+RL is **PASS** vs the Phase-1 bar. Next depth item:
[Topological Data Analysis](tda-deep.md) → then application systems
(recommenders first).
