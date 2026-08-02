# Probabilistic / Bayesian ML (deep)

BuildML’s probabilistic path is a **Session-facing uncertainty** stack on
sklearn estimators, not a probabilistic-programming product.

## Mental model

1. `fit_probabilistic` fits on Session **train**.
2. If `conformal=True`, a calibration subset is carved from **train only**
   (stratified for classification). The estimator fits on the remainder;
   nonconformity scores on the carve yield a finite-sample quantile.
3. `predict_interval` builds regression bands (`posterior_std`,
   `split_conformal`, or `both`) or classification prediction sets.
4. `evaluate_probabilistic` reports point metrics plus NLL, coverage / mean
   width (or set coverage / mean set size), and binary Brier/ECE when
   applicable.
5. `save_probabilistic_bundle` / `load_probabilistic_bundle` persist the
   `ProbabilisticPlan` separately from Session checkpoints.

## Estimators

| Key | Task | Uncertainty |
| --- | --- | --- |
| `bayesian_ridge` | regression | `return_std` + optional conformal |
| `gaussian_process_regressor` | regression | `return_std` + optional conformal |
| `gaussian_process_classifier` | classification | `predict_proba` + conformal sets |
| `gaussian_nb` | classification | `predict_proba` + conformal sets |

GP `n_restarts_optimizer` defaults to `0` for cheap/deterministic runs.

## Leakage discipline

- Fit and conformal calibration never see validation/test.
- Holdout is for evaluation and interval *scoring* only.
- Do not retune `alpha` / conformal fraction against the locked test set
  without a declared protocol.

## Relationship to `Session.calibration`

Classical `Session.calibration()` diagnoses reliability for classical
`fit(...)` classifiers (`FitResult`). The probabilistic path does **not**
replace it; `evaluate_probabilistic` reports NLL/Brier/ECE for its own
plan. Both can coexist on one Session.

## Why no MAPIE / PyMC extra

Split conformal (absolute residual / `1 − p(y)`) is implemented in-tree for
this Session-scoped path, so core stays sklearn-only. PyMC/Stan would be a
different product surface (MCMC / probabilistic programming) and is an
explicit non-goal here.

## Bundle boundary

`buildml.probabilistic_bundle.v1` stores the estimator, conformal quantile,
train carve indices, and disclosures. Session checkpoints do **not** embed
`ProbabilisticPlan`.

## Anti-patterns

- Calling this a PyMC/Stan platform
- Calibrating conformal on Session test
- Treating Gaussian `return_std` bands as distribution-free without conformal
- Expecting Bayesian deep nets or hierarchical MCMC samples
