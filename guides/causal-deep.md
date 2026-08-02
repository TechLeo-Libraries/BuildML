# Causal ML (deep)

BuildML’s causal path is a **separate, assumption-declared** Session surface for
backdoor average treatment effect (ATE) estimation. It is intentionally not
bolted onto EDA.

## Mental model

1. `declare_causal_assumptions` records treatment, outcome, confounders,
   estimand=`ATE`, identification=`backdoor`, and requires
   `acknowledge_unconfoundedness=True` plus `acknowledge_positivity=True`.
2. `fit_causal` fits nuisance models on Session **train only** and estimates
   ATE (optional full retrain bootstrap CI).
3. `estimate_causal` / `evaluate_causal` score a partition with **fixed** train
   nuisances. Holdout metrics check predictive quality — they do **not** prove
   identification.
4. `refute_causal` runs a simple placebo-treatment or random-confounder
   sensitivity disclosure (not a full DoWhy suite).
5. `save_causal_bundle` / `load_causal_bundle` persist `CausalPlan` separately
   from Session checkpoints.

## Why EDA cannot call this

EDA reports, mutual information, permutation importance, clustering labels, and
anomaly scores remain **associational**. Their teaching prose continues to
refuse causal claims. There is no bridge that turns an EDA finding into an ATE
without a new, explicit `CausalAssumptions` object.

## Estimators

| Method | Nuisances | Notes |
| --- | --- | --- |
| `t_learner` | μ₀(W), μ₁(W) | Outcome regression ATE |
| `ipw` | e(W)=P(T=1\|W) | Inverse propensity; clips extremes |
| `aipw` (default) | μ₀, μ₁, e | Doubly robust score |

Treatment must be **binary**. Outcome may be continuous or binary. Confounders
are numeric (after Session preprocess). Empty confounders require
`allow_empty_confounders=True` (extremely strong assumption).

## Leakage discipline

- Nuisance fitting never sees validation/test.
- Bootstrap in `fit_causal` resamples **train**.
- Partition bootstrap in `estimate_causal` keeps nuisances fixed.
- Instruments, if supplied, are **refused** until an IV path exists — unused
  instruments must not silently count as identification.

## Bundle boundary

`buildml.causal_bundle.v1` stores assumptions, fitted nuisances, train ATE/CI,
and disclosures. Session checkpoints do **not** embed `CausalPlan`.

## Anti-patterns

- Inferring causality from `eda()` / importance ranks
- Skipping acknowledgements
- Advertising evaluate metrics as proof of unconfoundedness
- Calling this DoWhy / EconML / causal discovery
- Claiming IV identification while instruments are refused

## Relation to probabilistic / classical paths

Probabilistic ML quantifies predictive uncertainty. Classical fit/evaluate
optimize predictive risk. Neither identifies counterfactual effects. Causal ML
is a third path with a harder entry gate: declared assumptions.

## Tracker

Known residuals (accepted): no IV/front-door; binary treatment; ATE not CATE.
Next Phase 2 depth item: [Graph ML](quickstart-graph.md) → then evolutionary
algorithms as search/HPO backend.
