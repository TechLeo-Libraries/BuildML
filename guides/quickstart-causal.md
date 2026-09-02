# Causal ML quickstart

```bash
pip install buildml
# DoWhy / EconML: pip install "buildml[causal-industry]"
```

Backdoor ATE only, and only after you declare it.
`session.causal.declare_assumptions` needs `treatment`, `outcome`,
`confounders`, and explicit unconfoundedness plus positivity
acknowledgements. Without that declaration the fit refuses. EDA never
fills this in. Instruments are refused: IV and front-door are not
implemented. Default method is native AIPW.

[Causal deep](causal-deep.md) ·
Paste: [`examples/causal_aipw_ate.py`](../examples/causal_aipw_ate.py) ·
Evidence: [causal-treatment-effect](../proofs/causal-treatment-effect/)

```python
import numpy as np
import pandas as pd
from buildml import Session
from buildml.causal import causal_capability_matrix

print(causal_capability_matrix()["default_backend_when_installed"])

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

# Native (always available)
fit = session.causal.fit(backend="native", method="aipw", bootstrap_samples=50)
print(fit.ate, fit.ate_ci_low, fit.ate_ci_high)

# DoWhy when buildml[causal-industry] is installed:
# fit = session.causal.fit(backend="dowhy", method="backdoor_linear")
# ref = session.causal.refute(kind="random_common_cause")

# EconML when installed:
# fit = session.causal.fit(backend="econml", method="dml", bootstrap_samples=50)

ev = session.causal.evaluate(partition="validation")
print(ev.metrics, ev.ate)

ref = session.causal.refute(kind="placebo_treatment")
print(ref.original_ate, ref.refute_ate)

session.causal.save_bundle("artifacts/causal_bundle")
# Roundtrip: other.causal.load_bundle(..., trusted=True) then evaluate again
```

| In scope | Out of scope |
| --- | --- |
| Declared backdoor ATE | Causal discovery / graph learning |
| native + optional DoWhy/EconML | IV / front-door (instruments refused) |
| Train-only fit + bootstrap | Causality from EDA alone |
| DoWhy refutation when installed | Proof of unconfoundedness from holdout |
| Distinct `buildml.causal_bundle.v1` | Multi-valued / continuous treatment |

Related next: federated learning.
