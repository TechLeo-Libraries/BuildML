# Symbolic and neuro-symbolic deep

```bash
pip install buildml
# skope-rules / imodels / Z3 lite: pip install "buildml[symbolic-industry]"
# concept-bottleneck / neural-additive bases: pip install "buildml[torch]"
```

You want if-then rules over columns, either ones you wrote or ones
induced from train, with a trace of which rule fired. That is
`session.symbolic`. The hybrid that wraps a sklearn (or lite Torch) model
with the same rules is `session.symbolic.fit_neuro`.

`session.symbolic.fit()` with no extra knobs uses `source="decision_tree"`
on the **sklearn** backend. That stays sklearn even when
`buildml[symbolic-industry]` is installed. Industry export is something
you ask for with `backend="industry"` or `method="skope_rules"` (or
`rulefit` / `boosted_rules`). Neuro-symbolic defaults to
`mode="constraint_overlay"` and `base_estimator="logistic_regression"`:
sklearn, not Torch, until you name a torch method.

This is tabular rules. It is not Prolog, not a Z3 product, and not an
expert-system suite. `verify_constraints=True` is a lite SAT check on
hard antecedents when z3-solver is present, not a proof that the rule
set is globally consistent.

Short on-ramp: [symbolic quickstart](quickstart-symbolic.md). Proof:
[policy-rules-neuro-symbolic](../proofs/policy-rules-neuro-symbolic/).

## Induce, predict, evaluate

Fit needs a split and exactly one target. Induction sees **train only**.
`predict` defaults to test and, with `return_traces=True` (the default),
returns fired rule ids and the chosen rule. `evaluate` defaults to
validation: accuracy / F1 for classification, RMSE / R² for regression,
plus rule coverage.

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

fit = session.symbolic.fit(source="decision_tree", task="classification")
print(fit.backend, fit.n_rules, fit.provenance)

pred = session.symbolic.predict(partition="test", return_traces=True)
print(pred.traces[0].fired_rule_ids, pred.traces[0].chosen_rule_id)

ev = session.symbolic.evaluate(partition="validation")
print(ev.metrics, ev.rule_coverage)

session.symbolic.save_bundle("artifacts/symbolic_bundle")
```

Tree induction uses `max_depth=4`, `min_samples_leaf=5`, `max_rules=32`
unless you change them. Declared rules are never relabeled as induced.

## Backends and sources

| Backend | Extra | How you get there | What it induces |
| --- | --- | --- | --- |
| `sklearn` (what `fit()` actually runs) | core | `source="declared"`, `"decision_tree"`, or `"decision_list"` | Your rules, sklearn tree paths, or sequential covering |
| `industry` | `symbolic-industry` | `backend="industry"` or `method=` one of the industry names | `skope_rules` (default industry method when skope-rules imports), else `rulefit`, then `boosted_rules` |

| `source` / `method` | Provenance stored on the plan | Learns from train? |
| --- | --- | --- |
| `declared` | caller / expert | No |
| `decision_tree` | `induced_tree` | Yes |
| `decision_list` | `induced_list` | Yes |
| `skope_rules` | `induced_skope` | Yes |
| `rulefit` / `boosted_rules` | `induced_*` | Yes |

`source="decision_tree"` and `source="decision_list"` stay on sklearn
even if industry extras are present. That is deliberate: auto-preferring
industry used to drop those sources silently. Pass `method="skope_rules"`
when you want the industry path.

skope-rules is skipped on Python 3.13 (broken `collections.Iterable`).
imodels and z3-solver still install from `symbolic-industry`.

```python
# When buildml[symbolic-industry] is installed:
# session.symbolic.fit(backend="industry", method="skope_rules")
```

## Neuro-symbolic hybrid

Same split, same train-only rule of the game. `fit_neuro` fits a base
estimator and binds rules in one of three modes you pick:

| Mode | What happens at predict |
| --- | --- |
| `constraint_overlay` (default) | Base model predicts; hard/soft rules overlay |
| `rules_as_features` | Rules fire as binary columns; the base fits on `[X ‖ R]` |
| `constraint_repair` | Base predicts; hard constraints repair violations |

Soft rules scale by `soft_strength` (default 0.5) times `rule.strength`.
Hard rules override in overlay and repair in repair mode. Traces expose
`neural_prediction`, `chosen_rule_id`, and `repaired`.

Sklearn bases: `logistic_regression` (default), `ridge`,
`random_forest`, `decision_tree`. Torch methods, when you ask:
`concept_bottleneck_lite`, `neural_additive_lite`. Naming one of those
as `base_estimator` (or `torch_method`) is what selects `backend="torch"`.
`backend=None` with the default logistic base stays sklearn even if
Torch is installed.

```python
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
print(neuro.mode, neuro.n_rules)
print(session.symbolic.evaluate_neuro(partition="test").metrics)
```

`evaluate_neuro` / `predict_neuro` are the hybrid twins. They do not
update the pure-symbolic plan.

## Z3 lite check

`verify_constraints=True` on `session.symbolic.fit` runs a SAT check on
hard constraint antecedents when z3-solver is installed via
`symbolic-industry`. Missing Z3 raises `MissingExtraError`. A passing
check does not mean the whole knowledge base is consistent, complete, or
causal.

## Bundles

`session.symbolic.save_bundle` writes `buildml.symbolic_bundle.v1`
(`meta.json` + `symbolic_plan.joblib`). `meta.kind` is `symbolic` or
`neuro_symbolic`. Session checkpoints do not embed either plan. Load
with `trusted=True` only for a file you made.

Runnable mirror: [`examples/symbolic_rules_loop.py`](../examples/symbolic_rules_loop.py).
Benchmark: `python benchmarks/symbolic/rule_fidelity.py`.

## When it refuses

| What you see | What happened |
| --- | --- |
| No split | `fit` / `fit_neuro` before `split` |
| No target | Symbolic fit needs exactly one target |
| `MissingExtraError` for `symbolic-industry` | You asked for industry methods or Z3 without the extra |
| `MissingExtraError` for `torch` | You asked for a torch neuro backend without Torch |
| Invalid source for sklearn | Something other than `declared` / `decision_tree` / `decision_list` |
| Invalid industry method | Name not in the methods that actually imported |

[Symbolic quickstart](quickstart-symbolic.md) ·
[policy-rules-neuro-symbolic](../proofs/policy-rules-neuro-symbolic/) ·
[Artifacts](artifacts-checkpoints-bundles.md) ·
[Leakage and recipes](leakage-cv-recipes.md)
