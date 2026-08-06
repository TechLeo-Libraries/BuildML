# Fairness (observational) quickstart

> **Install:** Install Session 2.x with `pip install buildml` (2.5.x). Legacy 1.x remains available as `pip install "buildml==1.0.9"`. Install 2.x from GitHub (or an editable checkout).
> Fairness metrics ship in core — no optional extra.
> See [installation](../docs/installation.rst).

Holdout group-disparity reporting on a fitted binary classifier: selection
rates, demographic parity, disparate impact, equalized odds gaps, per-group
classical metrics, and optional stability bands.

**Boundary:** this is **observational analysis**, not a legal audit, not causal
fairness, and not automatic bias mitigation. Sensitive columns must be declared
by the caller. `positive_label` is hard-validated against observed labels so
string targets with a default `1` raise instead of silent zero rates.

**Deep guide:** [fairness-deep.md](fairness-deep.md) ·
**Proof:** [loan-fairness-observational](../proofs/loan-fairness-observational/) ·
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

print(session.fairness.capability_matrix()["non_goals"][:2])

# String labels require an explicit positive_label — default 1 would raise.
report = session.fairness.evaluate(
    sensitive_column="group",
    partition="test",
    positive_label="approved",
    bootstrap_samples=50,  # optional stability bands
)
print(report.demographic_parity_difference)
print(report.selection_rate_by_group)
print(report.classical_metrics_by_group["A"]["f1"])
print(report.to_markdown().splitlines()[0])
```

Bridge after classical evaluate:

```python
session.evaluate(partition="test")
report = session.fairness.attach_to_last_eval(
    sensitive_column="group",
    positive_label="approved",
)
```

Intersectional keys (composite `group|…`):

```python
# report = session.fairness.evaluate(
#     sensitive_column=["group", "region"],
#     positive_label="approved",
# )
```

Opt-in mitigation **suggestions** (not auto-applied, not certification):

```python
thr = session.fairness.suggest_thresholds(
    sensitive_column="group",
    partition="validation",
    positive_label="approved",
)
weights = session.fairness.suggest_reweighing(
    sensitive_column="group",
    partition="train",
    positive_label="approved",
)
```

Discoverability helpers on Session:

```python
caps = Session.list_capabilities()
print([d["domain"] for d in caps["domains"] if d["domain"] == "fairness"])
# describe_method still keys flat names; preferred call path is session.fairness.evaluate(...)
print(Session.describe_method("evaluate_fairness")["summary"][:120])
```

## Non-goals

- Legal disparate-impact certification
- Inferring protected class membership
- Multi-class / regression fairness suites
- Automatic / silent reweighing or fairness washing
