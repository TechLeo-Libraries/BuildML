# Online / continual learning (deep)

> **Install (GitHub 2.x):**
> `pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"`
> Core sklearn only — no optional extra.

## What this is

Session-facing incremental learning with the sklearn `partial_fit` family:

1. `fit_online` — warm-start on an initial **train** chunk
2. `partial_fit_online` — update on subsequent train chunks (or role-aligned frames)
3. `evaluate_online` / `predict_online` — holdout inference (never for updates)
4. `save_online_bundle` / `load_online_bundle` — `buildml.online_bundle.v1`

| In scope | Out of scope (next / never-as-product) |
| --- | --- |
| SGD / PassiveAggressive / Perceptron / NB `partial_fit` | Deep MTL / streaming platforms |
| Train-cursor chunk carving + external frames | Meta / federated / Bayesian / causal / graph |
| Class vocabulary contract on first fit | Distributed streaming platforms |
| Disclosed refit fallback (opt-in) | Lifelong research suites (EWC, replay zoos) |
| Lite chunk mean-shift disclosure | Full drift product (use `Session.eda()`) |

## Leakage discipline

- Updates use **train** rows (or user frames with matching feature/target columns).
- Validation/test indices are refused.
- Evaluation never feeds `partial_fit`.
- Classifiers: `classes=` on first fit — explicit or discovered from the **full train target column** (labels only; features from unseen chunks wait until their update).

## Estimators

| Name | Task | Notes |
| --- | --- | --- |
| `sgd_classifier` | classification | Default; `log_loss` |
| `sgd_regressor` | regression | |
| `passive_aggressive_classifier` | classification | |
| `passive_aggressive_regressor` | regression | |
| `perceptron` | classification | |
| `multinomial_nb` | classification | Non-negative features required |
| `bernoulli_nb` | classification | |

Estimators without `partial_fit` raise unless `allow_refit_fallback=True`, which **always** discloses a full cumulative `.fit` (never silent).

## Bundle boundary

`buildml.online_bundle.v1` stores `OnlinePlan` (estimator + cursor + seen indices + update history + classes). Session checkpoints do **not** embed it. See [Artifacts](artifacts-checkpoints-bundles.md).

## Teaching surfaces

- Concepts: `online-partial-fit`, `online-class-discovery`, `online-drift-disclose`, `online-bundle-boundary`
- Overlays for all Session ops; AI allowlist: `fit_online`, `partial_fit_online`, `evaluate_online`, save/load bundle
- Walkthrough / audit include online status

## Phase tracker

Phase 2 items 1–5 (semi / self / active / online / multi-task) are done. **Next:** meta-learning.
