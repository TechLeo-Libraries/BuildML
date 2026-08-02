# Quickstart: Causal ML

Session path for **assumption-declared** backdoor ATE estimation: declare
`CausalAssumptions`, fit train-only models (native / DoWhy / EconML when
installed), estimate effects with optional bootstrap CIs, run sensitivity
checks, and persist via `buildml.causal_bundle.v1`.

**Critical boundary:** EDA / associations / feature importance remain
**associational**. They never identify causal effects and never populate
`CausalAssumptions`. Estimation **refuses** without an explicit declaration
(including unconfoundedness + positivity acknowledgements).

Honesty: native sklearn nuisances ship with core BuildML. Install
`buildml[causal-industry]` for DoWhy refutation and EconML DML/CATE/policy
paths. **Not** causal discovery, **not** IV / front-door (instruments refused).

**Go deeper:** [Causal deep](causal-deep.md) ·

**Proof:** [causal-treatment-effect](../proofs/causal-treatment-effect/) (+ Tier C sklearn AIPW twin).
[EDA / Teaching Studio](eda-teaching-studio.md) (still non-causal) ·
[Artifacts](artifacts-checkpoints-bundles.md)

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

session.declare_causal_assumptions(
    treatment="t",
    outcome="y",
    confounders=["x1", "x2"],
    estimand="ATE",
    acknowledge_unconfoundedness=True,
    acknowledge_positivity=True,
)

# Native (always available)
fit = session.fit_causal(backend="native", method="aipw", bootstrap_samples=50)
print(fit.ate, fit.ate_ci_low, fit.ate_ci_high)

# DoWhy when buildml[causal-industry] is installed:
# fit = session.fit_causal(backend="dowhy", method="backdoor_linear")
# ref = session.refute_causal(kind="random_common_cause")

# EconML when installed:
# fit = session.fit_causal(backend="econml", method="dml", bootstrap_samples=50)

ev = session.evaluate_causal(partition="validation")
print(ev.metrics, ev.ate)

ref = session.refute_causal(kind="placebo_treatment")
print(ref.original_ate, ref.refute_ate)

session.save_causal_bundle("artifacts/causal_bundle")
```

| In scope | Out of scope |
| --- | --- |
| Declared backdoor ATE | Causal discovery / graph learning |
| native + optional DoWhy/EconML | IV / front-door (instruments refused) |
| Train-only fit + bootstrap | Causality from EDA alone |
| DoWhy refutation when installed | Proof of unconfoundedness from holdout |
| Distinct `buildml.causal_bundle.v1` | Multi-valued / continuous treatment |

Next in R5 tracker: **Federated** (R5.5).
