# Fairness (observational) quickstart

```bash
pip install buildml
```

Holdout disparity on a fitted classifier. You name the sensitive
column; BuildML will not infer protected class. `suggest_thresholds` and
`suggest_reweighing` return suggestions only. They are not applied.
Default evaluate partition is test. This is observational reporting, not
a legal audit and not causal fairness.

String labels need an explicit `positive_label`. Default `1` raises
instead of inventing zero rates.

[Fairness deep](fairness-deep.md) ·
Paste: [`examples/fairness_observational_loop.py`](../examples/fairness_observational_loop.py) ·
Evidence: [loan-fairness-observational](../proofs/loan-fairness-observational/)

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

# String labels require an explicit positive_label - default 1 would raise.
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
