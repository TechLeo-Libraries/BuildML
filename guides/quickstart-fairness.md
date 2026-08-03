# Fairness (observational) quickstart

> **Install first (GitHub):** PyPI `buildml` is still legacy 1.x and does **not**
> install Session 2.x. Install 2.x from GitHub (or an editable checkout).
> Fairness metrics ship in core — no optional extra.
> See [installation](../docs/installation.rst).

Holdout group-disparity reporting on a fitted binary classifier: selection
rates, demographic parity, disparate impact, and equalized odds gaps.

**Boundary:** this is **observational analysis**, not a legal audit, not causal
fairness, and not automatic bias mitigation. Sensitive columns must be declared
by the caller. `positive_label` is hard-validated against observed labels so
string targets with a default `1` raise instead of silent zero rates.

**Proof:** [loan-fairness-observational](../proofs/loan-fairness-observational/).
[Classical quickstart](quickstart-classical.md) ·
[stability](../docs/stability.md)

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from buildml import Session

rng = np.random.default_rng(0)
n = 400
group = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
x = rng.normal(size=n)
logits = x + np.where(group == "B", -0.7, 0.0)
y = np.where(logits > 0, "approved", "denied")
frame = pd.DataFrame({"x": x, "group": group, "decision": y})

session = (
    Session.ingest(frame)
    .set_roles({"x": "feature", "group": "ignore", "decision": "target"})
    .split(test_size=0.25, validation_size=0.2, stratify=True, random_state=0)
    .fit(LogisticRegression(max_iter=500), task="classification")
)

print(Session.fairness_capability_matrix()["non_goals"][:2])

# String labels require an explicit positive_label — default 1 would raise.
report = session.evaluate_fairness(
    sensitive_column="group",
    partition="test",
    positive_label="approved",
)
print(report.demographic_parity_difference)
print(report.selection_rate_by_group)
```

Discoverability helpers on Session:

```python
caps = Session.list_capabilities()
print([d["domain"] for d in caps["domains"] if d["domain"] == "fairness"])
print(Session.describe_method("evaluate_fairness")["summary"][:120])
```

## Non-goals

- Legal disparate-impact certification
- Inferring protected class membership
- Multi-class / regression fairness suites
- Automatic reweighing / mitigation products
