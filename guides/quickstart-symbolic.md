# Quickstart: Symbolic + Neuro-symbolic AI

Session path for **tabular if-then rules** and a **base-model + rules hybrid**.
Induce rules from train (sklearn tree/list or industry skope-rules/imodels) or
compile expert-declared rules; predict with explanation traces; persist via
`buildml.symbolic_bundle.v1`.

Honesty: **not** an AGI symbolic reasoner, Prolog engine, or full Z3 SMT product.
Core stays light (numpy / pandas / sklearn). Optional industry depth via
`buildml[symbolic-industry]`; torch neuro-symbolic via `buildml[torch]`.

**Proof:** [policy-rules-neuro-symbolic](../proofs/policy-rules-neuro-symbolic/) (+ Tier C DecisionTree twin). Cross-domain: [pulse-support-copilot](../proofs/pulse-support-copilot/).

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

# Inspect honest defaults (industry when installed, else sklearn)
print(session.symbolic.capability_matrix()["default_symbolic_backend_when_installed"])

# Symbolic: sklearn decision-tree rules (always available)
fit = session.symbolic.fit(backend="sklearn", source="decision_tree", task="classification")
print(fit.backend, fit.n_rules, fit.provenance)

# Industry rule export when buildml[symbolic-industry] is installed:
# fit = session.symbolic.fit(backend="industry", method="skope_rules")

pred = session.symbolic.predict(partition="test", return_traces=True)
print(pred.traces[0].fired_rule_ids, pred.traces[0].chosen_rule_id)

ev = session.symbolic.evaluate(partition="validation")
print(ev.metrics, ev.rule_coverage)

session.symbolic.save_bundle("artifacts/symbolic_bundle")

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
neuro = session.symbolic.fit_neuro(
    backend="sklearn",
    mode="constraint_overlay",
    base_estimator="logistic_regression",
    task="classification",
    rules=constraints,
    rule_source="declared",
)
print(neuro.backend, neuro.mode, neuro.rule_provenance, neuro.n_rules)
print(session.symbolic.evaluate_neuro(partition="test").metrics)

# Torch lite concept-bottleneck when buildml[torch] is installed:
# session.symbolic.fit_neuro(backend="torch", base_estimator="concept_bottleneck_lite")
```

| In scope | Out of scope |
| --- | --- |
| sklearn + industry rule induction | Prolog / full Z3 product / AGI reasoners |
| Rule-firing traces | Full expert-system product |
| Neuro-symbolic overlay / features / repair | Fuzzy logic product; LTN research stack |
| Optional Z3 lite constraint check | Complete SMT verification product |
| `buildml.symbolic_bundle.v1` | Session checkpoint embedding the plan |

Benchmark: `python benchmarks/symbolic/rule_fidelity.py` (rule accuracy vs
black-box RandomForest on tabular reference data).

Related next: case-based reasoning
([quickstart-cbr](quickstart-cbr.md)).
