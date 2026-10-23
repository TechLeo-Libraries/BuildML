# Symbolic / neuro-symbolic quickstart

```bash
pip install buildml
```

Tabular if-then rules, declared or induced from train. Default source is
a decision tree. `fit_neuro` is the sklearn hybrid overlay. General-purpose logic programming is outside this API. `verify_constraints=True` is a lite check only.

New classification fits preserve raw class labels: `"001"` stays a string,
and boolean/numeric classes retain their types through rules and trusted bundles.
Rule consequents must use the intended label representation; string `"1.0"`
does not count as numeric target `1.0` in new-fit accuracy or repair reporting.
Incompatible mixed training label types are refused. Existing bundles keep their
original class ordering and comparison behavior. Legacy neural encoders also keep
their historical numeric decoding and emit a refit notice; refit older plans to
obtain raw class identity. Loading cannot recover types lost by an older fit.

Evaluation scores the requested partition without refitting. `train` and `all`
include training rows and carry an explicit warning; those metrics are not
independent holdout evidence. Regression evaluation refuses non-finite, complex
or unscorable targets/predictions with a validation error instead of returning
partial metrics or raw numerical-library errors.

[Symbolic deep](symbolic-deep.md) ·
Runnable example: [`examples/symbolic_rules_loop.py`](../examples/symbolic_rules_loop.py) ·
Evidence: [policy-rules-neuro-symbolic](https://github.com/TechLeo-Libraries/BuildML/tree/main/proofs/policy-rules-neuro-symbolic)

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

# Inspect backend selection (optional backend when installed, otherwise sklearn)
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
| sklearn + industry rule induction | General-purpose logic programming |
| Rule-firing traces | Knowledge-base management |
| Neuro-symbolic overlay / features / repair | General fuzzy-logic inference |
| Optional Z3 lite constraint check | Complete SMT verification |
| `buildml.symbolic_bundle.v1` | Session checkpoint embedding the plan |

Benchmark: `python benchmarks/symbolic/rule_fidelity.py` (rule accuracy vs
black-box RandomForest on tabular reference data).
