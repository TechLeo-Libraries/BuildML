# Probabilistic / Bayesian ML

```bash
pip install buildml
# MAPIE conformal + NGBoost: pip install "buildml[probabilistic-industry]"
```

You want a point prediction plus an interval or a set, on a tabular
Session, without standing up PyMC or Stan.

`session.probabilistic.fit` defaults to estimator `bayesian_ridge` with
`conformal=True` and `conformal_calibration_fraction=0.2`. That
estimator is native, so `backend=None` stays native even if MAPIE is
installed. Pass `estimator="split"` (or `cv_plus` / `jackknife_plus`)
with `backend=None` to take MAPIE when the extra imported cleanly.
Pass `estimator="ngboost_regressor"` with `backend=None` to take
NGBoost the same way. `backend="mapie"` with the default
`bayesian_ridge` estimator is refused: name a MAPIE method.

The API refuses a fit without a split and conformal calibration on
validation/test. You decide `alpha` (default 0.1, so 90% intervals),
whether conformal stays on, and whether Gaussian `return_std` bands
are enough or you need the conformal overlay.

Short on-ramp: [probabilistic quickstart](quickstart-probabilistic.md).
Proof: [prob-interval-risk](../proofs/prob-interval-risk/).
Classical `session.calibration()` stays on the classical `fit` path.
This plan does not replace it.

## A first interval

```python
import numpy as np
import pandas as pd

from buildml import Session

rng = np.random.default_rng(0)
x = rng.normal(size=(200, 2))
y = 1.5 * x[:, 0] - 0.7 * x[:, 1] + rng.normal(scale=0.4, size=200)
frame = pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "y": y})

session = (
    Session.ingest(frame)
    .set_roles({"a": "feature", "b": "feature", "y": "target"})
    .split(test_size=0.2, validation_size=0.2, random_state=0)
    .scale(method="standard")
)

fit = session.probabilistic.fit(
    estimator="bayesian_ridge",
    alpha=0.1,
    conformal=True,
)
print(fit.n_fit_rows, fit.n_conformal_calib_rows, fit.conformal_quantile)

intervals = session.probabilistic.predict_interval(partition="test")
print(intervals.method, intervals.lower[:3], intervals.upper[:3])

ev = session.probabilistic.evaluate(partition="validation")
print(ev.metrics)

session.probabilistic.save_bundle("artifacts/probabilistic_bundle")
```

On native and NGBoost, conformal carves a calibration subset from
**train only** (stratified for classification). MAPIE owns conformal
calibration internally: split carve, CV+, or jackknife+. Holdout is
for scoring intervals, not for fitting the quantile.

`evaluate` defaults to validation. `predict` and `predict_interval`
default to test. `predict` defaults to `return_std=True` and
`return_proba=True`. Do not retune `alpha` or the conformal fraction
against a locked test set unless you declared that protocol.

## Backends

| Backend | Extra | Role |
| --- | --- | --- |
| `native` | none (`bayesian_ridge` default) | sklearn BayesianRidge / GP / GaussianNB plus in-tree split conformal |
| `mapie` | `buildml[probabilistic-industry]` | MAPIE conformal regression/classification |
| `ngboost` | `buildml[probabilistic-industry]` | NGBoost predictive distributions, optional in-tree conformal overlay |

MAPIE and NGBoost need to import. A broken extra reports unavailable.

```python
session.probabilistic.capability_matrix()
```

### Native estimators

| Key | Task | Uncertainty |
| --- | --- | --- |
| `bayesian_ridge` | regression | `return_std` plus optional conformal |
| `gaussian_process_regressor` | regression | `return_std` plus optional conformal |
| `gaussian_process_classifier` | classification | `predict_proba` plus conformal sets |
| `gaussian_nb` | classification | `predict_proba` plus conformal sets |

GP `n_restarts_optimizer` defaults to 0 so runs stay cheap and
deterministic.

### MAPIE (`backend="mapie"`)

| Key | Description |
| --- | --- |
| `split` | Prefit base estimator on a train fit-carve; calibrate on a train calib-carve |
| `cv_plus` | Cross-validation+ on Session train |
| `jackknife_plus` | Jackknife+ on Session train |

Set `task="regression"` or `task="classification"` explicitly for
MAPIE. If you omit task, the resolver infers regression.

### NGBoost (`backend="ngboost"`)

| Key | Task |
| --- | --- |
| `ngboost_regressor` | regression (NLL / CRPS from `pred_dist`) |
| `ngboost_classifier` | classification (`predict_proba` plus optional conformal sets) |

## Metrics

Regression eval can report MAE, RMSE, R², NLL, CRPS, interval
coverage, mean interval width, interval score.

Classification eval can report accuracy, macro/weighted F1, NLL,
Brier, ECE, set coverage, mean set size.

Gaussian posterior bands are model-based. Distribution-free coverage
is the conformal overlay. Do not quote `return_std` as if it were
split-conformal.

## Relation to classical calibration

`session.calibration()` diagnoses reliability for a classical
`session.fit(...)` classifier. `session.probabilistic.evaluate`
reports NLL / Brier / ECE / CRPS for **this** plan. Both can sit on
one Session. They are not the same object.

## Bundles

`buildml.probabilistic_bundle.v1` stores the estimator, backend,
conformal quantile, train carve indices, and disclosures. A Session
checkpoint does not embed the `ProbabilisticPlan`. `trusted=True` only
for a file you made.

[Artifacts](artifacts-checkpoints-bundles.md)

## When it refuses

| What you see | What happened |
| --- | --- |
| Fit before a split | Train-only fit and train-only conformal carve |
| Classifier estimator with `task="regression"` | Task must match the estimator |
| `MissingExtraError` | MAPIE / NGBoost without `buildml[probabilistic-industry]` |
| Unknown estimator | See the tables above |

This is not a PyMC / Stan / NumPyro platform, not Bayesian deep nets,
and not hierarchical MCMC.

[Probabilistic quickstart](quickstart-probabilistic.md)
