# Causal ML (deep)

BuildML’s causal path is a **separate, assumption-declared** Session surface for
backdoor average treatment effect (ATE) estimation. It is intentionally not
bolted onto EDA.

## Mental model

1. `declare_causal_assumptions` records treatment, outcome, confounders,
   estimand=`ATE`, identification=`backdoor`, and requires
   `acknowledge_unconfoundedness=True` plus `acknowledge_positivity=True`.
2. `fit_causal` fits on Session **train only** and estimates ATE. Backends:
   - **native** (default): T-learner / IPW / AIPW + bootstrap CI
   - **dowhy** (`buildml[causal-industry]`): causal graph from declared
     confounders, DoWhy identification + backdoor estimation
   - **econml** (`buildml[causal-industry]`): DML, CausalForestDML, PolicyTree
3. `estimate_causal` / `evaluate_causal` score a partition with **fixed** train
   models. Holdout metrics check predictive quality: they do **not** prove
   identification.
4. `refute_causal` runs sensitivity checks. Native: placebo / random confounder.
   DoWhy: full refutation suite (placebo, random common cause, unobserved
   confounder, data subset, placebo outcome).
5. `save_causal_bundle` / `load_causal_bundle` persist `CausalPlan` separately
   from Session checkpoints.

Inspect installed backends with `buildml.causal.causal_capability_matrix()`.

## Why EDA cannot call this

EDA reports, mutual information, permutation importance, clustering labels, and
anomaly scores remain **associational**. Their teaching prose continues to
refuse causal claims. There is no bridge that turns an EDA finding into an ATE
without a new, explicit `CausalAssumptions` object.

## Backends and methods

| Backend | Extra | Methods | Refutation |
| --- | --- | --- | --- |
| `native` | core | `t_learner`, `ipw`, `aipw` | placebo, random confounder |
| `dowhy` | causal-industry | `backdoor_linear`, `backdoor_propensity_score`, `backdoor_propensity_weighting` | DoWhy suite |
| `econml` | causal-industry | `dml`, `causal_forest`, `policy_tree` | native refuters |

Treatment must be **binary**. Outcome may be continuous or binary. Confounders
are numeric (after Session preprocess). Empty confounders require
`allow_empty_confounders=True` (extremely strong assumption).

## Leakage discipline

- Nuisance fitting never sees validation/test.
- Bootstrap in `fit_causal` resamples **train** (native/econml).
- Partition bootstrap in `estimate_causal` keeps nuisances fixed (native).
- Instruments, if supplied, are **refused** until an IV path exists: unused
  instruments must not silently count as identification.

## Bundle boundary

`buildml.causal_bundle.v1` stores assumptions, fitted models, train ATE/CI,
and disclosures. Session checkpoints do **not** embed `CausalPlan`.

## Anti-patterns

- Inferring causality from `eda()` / importance ranks
- Skipping acknowledgements
- Advertising evaluate metrics as proof of unconfoundedness
- Calling DoWhy refutation proof of identification
- Treating EconML policy_tree as a deployment-ready policy product
- Claiming IV identification while instruments are refused

## Relation to probabilistic / classical paths

Probabilistic ML quantifies predictive uncertainty. Classical fit/evaluate
optimize predictive risk. Neither identifies counterfactual effects. Causal ML
is a third path with a harder entry gate: declared assumptions.

