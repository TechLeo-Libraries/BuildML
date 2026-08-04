# Probabilistic / Bayesian ML (deep)

BuildML’s probabilistic path is a **Session-facing uncertainty** stack on
tabular estimators, not a probabilistic-programming product.

## Mental model

1. `session.probabilistic.fit` fits on Session **train** (choose a **backend**).
2. If `conformal=True` on **native** / **ngboost**, a calibration subset is
   carved from **train only** (stratified for classification). **MAPIE** owns
   conformal calibration internally (split carve, CV+, or jackknife+).
3. `session.probabilistic.predict_interval` builds regression bands or classification prediction sets.
4. `session.probabilistic.evaluate` reports point metrics plus NLL, CRPS (when
   feasible), coverage / mean width (or set coverage / mean set size).
5. `session.probabilistic.save_bundle` / `session.probabilistic.load_bundle` persist the
   `ProbabilisticPlan` separately from Session checkpoints.

## Backends

| Backend | Extra | Role |
| --- | --- | --- |
| `native` | core | sklearn BayesianRidge / GP / GaussianNB + in-tree split conformal |
| `mapie` | `probabilistic-industry` | MAPIE conformal regression/classification (split, CV+, jackknife+) |
| `ngboost` | `probabilistic-industry` | NGBoost predictive distributions + optional in-tree conformal overlay |

Install industry backends:

```bash
pip install 'buildml[probabilistic-industry]'
```

Inspect honest defaults:

```python
from buildml.probabilistic import probabilistic_capability_matrix
probabilistic_capability_matrix()
```

## Native estimators

| Key | Task | Uncertainty |
| --- | --- | --- |
| `bayesian_ridge` | regression | `return_std` + optional conformal |
| `gaussian_process_regressor` | regression | `return_std` + optional conformal |
| `gaussian_process_classifier` | classification | `predict_proba` + conformal sets |
| `gaussian_nb` | classification | `predict_proba` + conformal sets |

## MAPIE methods (`backend='mapie'`)

| Key | Description |
| --- | --- |
| `split` | Prefit base estimator on train fit-carve; calibrate on train calib-carve |
| `cv_plus` | Cross-validation+ on Session train |
| `jackknife_plus` | Jackknife+ on Session train |

Set `task='regression'` or `task='classification'` explicitly for MAPIE.

## NGBoost estimators (`backend='ngboost'`)

| Key | Task |
| --- | --- |
| `ngboost_regressor` | regression (NLL / CRPS from `pred_dist`) |
| `ngboost_classifier` | classification (`predict_proba` + optional conformal sets) |

GP `n_restarts_optimizer` defaults to `0` for cheap/deterministic runs.

## Leakage discipline

- Fit and conformal calibration never see validation/test.
- Holdout is for evaluation and interval *scoring* only.
- Do not retune `alpha` / conformal fraction against the locked test set
  without a declared protocol.

## Relationship to `Session.calibration`

Classical `Session.calibration()` diagnoses reliability for classical
`fit(...)` classifiers (`FitResult`). The probabilistic path does **not**
replace it; `session.probabilistic.evaluate` reports NLL/Brier/ECE/CRPS for its own
plan. Both can coexist on one Session.

## Bundle boundary

`buildml.probabilistic_bundle.v1` stores the estimator, backend, conformal
quantile, train carve indices, and disclosures. Session checkpoints do **not**
embed `ProbabilisticPlan`.

## Anti-patterns

- Calling this a PyMC/Stan platform
- Calibrating conformal on Session test
- Treating Gaussian `return_std` bands as distribution-free without conformal
- Expecting Bayesian deep nets or hierarchical MCMC samples
