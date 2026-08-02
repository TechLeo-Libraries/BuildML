# Quickstart: Symbolic + Neuro-symbolic AI

Session path for **tabular if-then rules** and a **sklearn + rules hybrid**.
Induce rules from train (`decision_tree` / `decision_list`) or compile
expert-declared rules; predict with explanation traces; persist via
`buildml.symbolic_bundle.v1`.

Honesty: **not** an AGI symbolic reasoner, Prolog engine, or Z3 SMT solver.
Core stays light (numpy / pandas / sklearn). Fuzzy logic and full expert-system
products remain out of scope.

**Go deeper:** [Symbolic deep](symbolic-deep.md) ·
[Artifacts](artifacts-checkpoints-bundles.md)

```python
import numpy as np
import pandas as pd
from buildml import Session

rng = np.random.default_rng(0)
x = rng.normal(size=(220, 2))
y = (x[:, 0] + 0.3 * x[:, 1] > 0).astype(int)
frame = pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "y": y})

session = (
    Session.ingest(frame)
    .set_roles({"a": "feature", "b": "feature", "y": "target"})
    .split(test_size=0.2, validation_size=0.2, random_state=0, stratify=True)
    .scale(method="standard")
)

# Symbolic: induce a decision-tree rule list on train only
fit = session.fit_symbolic(source="decision_tree", task="classification")
print(fit.n_rules, fit.provenance)

pred = session.predict_symbolic(partition="test", return_traces=True)
print(pred.traces[0].fired_rule_ids, pred.traces[0].chosen_rule_id)

ev = session.evaluate_symbolic(partition="validation")
print(ev.metrics, ev.rule_coverage)

session.save_symbolic_bundle("artifacts/symbolic_bundle")

# Neuro-symbolic: sklearn base + hard constraint overlay
constraints = [
    {
        "rule_id": "high_a",
        "if": [{"column": "a", "op": ">", "value": 1.5}],
        "then": 1,
        "hardness": "hard",
        "kind": "constraint",
        "priority": 100,
    }
]
neuro = session.fit_neuro_symbolic(
    mode="constraint_overlay",
    base_estimator="logistic_regression",
    task="classification",
    rules=constraints,
    rule_source="declared",
)
print(neuro.mode, neuro.rule_provenance, neuro.n_rules)
print(session.evaluate_neuro_symbolic(partition="test").metrics)
```

| In scope | Out of scope |
| --- | --- |
| Declared / tree / decision-list rules | Prolog / Z3 / AGI reasoners |
| Rule-firing traces | Full expert-system product |
| Neuro-symbolic overlay / features / repair | Fuzzy logic product; LTN research stack |
| `buildml.symbolic_bundle.v1` | Session checkpoint embedding the plan |

Next Phase 2 item after symbolic: **Case-based reasoning**
([quickstart-cbr](quickstart-cbr.md)).
