# Quickstart: Bayesian / probabilistic ML

Session path for uncertainty quantification: fit sklearn `BayesianRidge` /
Gaussian Process / `GaussianNB`, get predictive std or probabilities, and
optionally calibrate **split conformal** intervals/sets on a **train-only**
carve. Persist via `buildml.probabilistic_bundle.v1`.

Honesty: **not** a PyMC / Stan / NumPyro MCMC platform and **not** Bayesian
deep nets. Classical `Session.calibration()` remains for classical
`fit(...)` classifiers and is complementary.

**Go deeper:** [Probabilistic deep](probabilistic-deep.md) ·

**Proof:** [prob-interval-risk](../proofs/prob-interval-risk/) (+ Tier C BayesianRidge+quantile). Cross-domain: [harbor-demand-desk](../proofs/harbor-demand-desk/).
[Artifacts](artifacts-checkpoints-bundles.md)

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

fit = session.fit_probabilistic(
    estimator="bayesian_ridge",
    alpha=0.1,
    conformal=True,
)
print(fit.n_fit_rows, fit.n_conformal_calib_rows, fit.conformal_quantile)

intervals = session.predict_interval(partition="test")
print(intervals.method, intervals.lower[:3], intervals.upper[:3])

ev = session.evaluate_probabilistic(partition="validation")
print(ev.metrics)

session.save_probabilistic_bundle("artifacts/probabilistic_bundle")
```

| In scope | Out of scope |
| --- | --- |
| BayesianRidge / GP / GaussianNB | PyMC / Stan / NumPyro MCMC |
| Predictive std / proba + NLL | Bayesian deep nets |
| Train-only split conformal | Conformal calibration on Session test |
| Distinct `buildml.probabilistic_bundle.v1` | Session checkpoint embedding the plan |

See also: [causal ML quickstart](quickstart-causal.md).
