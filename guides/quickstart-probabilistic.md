# Probabilistic quickstart

```bash
pip install buildml
```

BayesianRidge / GP / Naive Bayes, plus train-only split conformal
(default on, 20% of train). Default estimator is BayesianRidge. Not
PyMC / Stan / NumPyro. Classical `session.calibration()` stays on the
classical fit path.

[Probabilistic deep](probabilistic-deep.md) ·
[prob-interval-risk](../proofs/prob-interval-risk/)

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

| In scope | Out of scope |
| --- | --- |
| BayesianRidge / GP / GaussianNB | PyMC / Stan / NumPyro MCMC |
| Predictive std / proba + NLL | Bayesian deep nets |
| Train-only split conformal | Conformal calibration on Session test |
| Distinct `buildml.probabilistic_bundle.v1` | Session checkpoint embedding the plan |

Related next: [causal ML](quickstart-causal.md).
