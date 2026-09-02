# Causal ML

```bash
pip install buildml
# DoWhy + EconML: pip install "buildml[causal-industry]"
```

Backdoor average treatment effect on a Session, after you declare the
identification story. EDA, mutual information, and permutation
importance stay associational. Nothing in those surfaces fills in a
`CausalAssumptions` object for you.

`session.causal.fit` defaults to method `aipw`. That name is native, so
`backend=None` stays native even if EconML is installed. Pass
`method="dml"` (or `causal_forest` / `policy_tree`) with `backend=None`
to take EconML when it imported cleanly. Pass
`method="backdoor_linear"` with `backend=None` to take DoWhy the same
way. `backend="econml"` with the default `aipw` method is refused:
name a DML / forest / policy method.

The API refuses fit without declared assumptions, skipped
acknowledgements, instruments (IV is not implemented), empty
confounders unless you set `allow_empty_confounders=True`, and
estimands other than ATE under backdoor. You decide the treatment,
outcome, confounder list, and whether those acknowledgements are
honest for your data.

Short on-ramp: [causal quickstart](quickstart-causal.md).
Proof: [causal-treatment-effect](../proofs/causal-treatment-effect/).

## Declare, then fit

`declare_assumptions` needs `treatment`, `outcome`, `confounders`,
`acknowledge_unconfoundedness=True`, and `acknowledge_positivity=True`.
`estimand` defaults to `ATE`. `identification` defaults to `backdoor`.
Treatment must be binary. Outcome may be continuous or binary.
Confounders should be numeric after Session preprocess.

```python
import numpy as np
import pandas as pd

from buildml import Session

rng = np.random.default_rng(0)
n = 400
w = rng.normal(size=(n, 2))
logit = 0.8 * w[:, 0] - 0.5 * w[:, 1]
t = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
y = 1.5 * t + 0.7 * w[:, 0] - 0.4 * w[:, 1] + rng.normal(scale=0.5, size=n)
frame = pd.DataFrame({"x1": w[:, 0], "x2": w[:, 1], "t": t, "y": y})

session = (
    Session.ingest(frame)
    .set_roles({"x1": "feature", "x2": "feature", "t": "feature", "y": "target"})
    .split(test_size=0.2, validation_size=0.2, random_state=0)
    .scale(method="standard")
)

session.causal.declare_assumptions(
    treatment="t",
    outcome="y",
    confounders=["x1", "x2"],
    estimand="ATE",
    acknowledge_unconfoundedness=True,
    acknowledge_positivity=True,
)

fit = session.causal.fit(backend="native", method="aipw", bootstrap_samples=50)
print(fit.ate, fit.ate_ci_low, fit.ate_ci_high)

ev = session.causal.evaluate(partition="validation")
print(ev.metrics, ev.ate)

ref = session.causal.refute(kind="placebo_treatment")
print(ref.original_ate, ref.refute_ate)

session.causal.save_bundle("artifacts/causal_bundle")
```

Nuisance models fit on **train only**. `bootstrap_samples` defaults to
200 on `fit` and resamples train (native/EconML). `clip_propensity`
defaults to `(0.01, 0.99)`. Outcome model defaults to `ridge`,
propensity to `logistic_regression`.

`estimate` defaults to `partition="train"` and keeps nuisances fixed
when it bootstraps a partition. `evaluate` defaults to validation: it
checks predictive quality of nuisances plus a partition ATE under the
declared assumptions. That is not proof of unconfoundedness.

## Backends and methods

| Backend | Extra | Methods | Refutation |
| --- | --- | --- | --- |
| `native` | none (`aipw` default) | `t_learner`, `ipw`, `aipw` | `placebo_treatment`, `random_confounder` |
| `dowhy` | `buildml[causal-industry]` | `backdoor_linear`, `backdoor_propensity_score`, `backdoor_propensity_weighting` | DoWhy suite (placebo, random common cause, unobserved confounder, data subset, placebo outcome) |
| `econml` | `buildml[causal-industry]` | `dml`, `causal_forest`, `policy_tree` | native placebo / random confounder |

DoWhy builds a graph from the confounders you declared, then identifies
and estimates. That is not causal discovery. EconML DML /
CausalForestDML estimate ATE with optional CATE spread.
`policy_tree` learns a treatment assignment rule on train. It is not a
deployment-ready policy product.

DoWhy and EconML need to import, not just be listed as installed.

```python
session.causal.capability_matrix()
```

Industry examples when the extra is installed:

```python
# fit = session.causal.fit(backend="dowhy", method="backdoor_linear")
# ref = session.causal.refute(kind="random_common_cause")
# fit = session.causal.fit(backend="econml", method="dml", bootstrap_samples=50)
```

## Why EDA cannot call this

EDA reports, clustering labels, and anomaly scores do not identify an
ATE. There is no bridge that turns an association into an effect
without a new, explicit assumptions object. If you skip
`declare_assumptions`, `fit` refuses.

Empty confounders are allowed only with `allow_empty_confounders=True`.
That is a strong claim: unconfoundedness with no covariates.

Instruments, if supplied, are refused. Unused instruments must not
silently count as identification. IV and front-door are not on this
surface.

## Refutation

`refute` defaults to `kind="placebo_treatment"`. Native and EconML
support placebo treatment and random confounder. DoWhy adds the
broader suite. A refutation that "fails to reject" is not proof of
identification. It is a sensitivity disclosure.

## Bundles

`buildml.causal_bundle.v1` stores assumptions, fitted models, train
ATE/CI, and disclosures. A Session checkpoint does not embed the
`CausalPlan`. `trusted=True` only for a file you made.

[Artifacts](artifacts-checkpoints-bundles.md)

## When it refuses

| What you see | What happened |
| --- | --- |
| No `CausalAssumptions` | Call `declare_assumptions` (or pass `assumptions=`) |
| Acknowledgements false | You must set both flags yourself |
| `instruments` non-empty | IV is not implemented |
| `estimand` not `ATE` / identification not `backdoor` | This surface is backdoor ATE |
| Empty confounders | Pass a list, or `allow_empty_confounders=True` |
| Treatment equals outcome, or either listed as a confounder | Distinct columns |
| `MissingExtraError` | DoWhy / EconML method without `buildml[causal-industry]` |

Holdout metrics are predictive calibration of nuisances plus an
out-of-sample ATE under **your** assumptions. Probabilistic ML
quantifies predictive uncertainty. Classical fit/evaluate optimize
predictive risk. Neither identifies a counterfactual effect.

[Causal quickstart](quickstart-causal.md)
