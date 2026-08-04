# Symbolic + Neuro-symbolic deep guide

## Scope

BuildML’s symbolic path is a **Session-native tabular rule engine** plus a
**neuro-symbolic hybrid** that binds a sklearn or lite torch base estimator to
those rules.

| Surface | Role |
| --- | --- |
| `session.symbolic.capability_matrix()` | Honest backend / method availability |
| `session.symbolic.fit` | Compile declared rules or induce from train |
| `session.symbolic.predict` | Decision-list inference + `RuleTrace` |
| `session.symbolic.evaluate` | Holdout accuracy/RMSE + rule coverage |
| `session.symbolic.fit_neuro` | sklearn/torch + rules in one API |
| `session.symbolic.predict_neuro` / `session.symbolic.evaluate_neuro` | Hybrid score / metrics |
| `session.symbolic.save_bundle` / `session.symbolic.load_bundle` | `buildml.symbolic_bundle.v1` |

## Backends

| Backend | Extra | Symbolic induction | Neuro-symbolic base |
| --- | --- | --- | --- |
| `sklearn` | none (core) | `declared`, `decision_tree`, `decision_list` | LR / Ridge / RF / DT |
| `industry` | `symbolic-industry` | `skope_rules`, `rulefit`, `boosted_rules` |: |
| `torch` | `torch` |: | `concept_bottleneck_lite`, `neural_additive_lite` |

Defaults when installed: industry symbolic backend when skope-rules/imodels
present; torch neuro-symbolic when torch present; otherwise sklearn.

## Rule sources (disclose provenance)

| `source` / `method` | Provenance | Learning? |
| --- | --- | --- |
| `declared` | Expert / caller | No |
| `decision_tree` | `induced_tree` | Yes: train only |
| `decision_list` | `induced_list` | Yes: sequential covering, train only |
| `skope_rules` | `induced_skope` | Yes: skope-rules on train |
| `rulefit` / `boosted_rules` | `induced_*` | Yes: imodels export on train |

Induction never uses Session validation/test. Declared rules are never silently
relabeled as induced.

## Neuro-symbolic modes

| Mode | Behavior |
| --- | --- |
| `constraint_overlay` | Predict with base model; apply hard/soft rules |
| `rules_as_features` | Fire rules as binary features; fit on `[X ‖ R]` |
| `constraint_repair` | Predict; hard constraints repair violations |

Soft rules use `soft_strength × rule.strength`. Hard rules override (overlay)
or repair (repair mode). Traces expose `neural_prediction`, `chosen_rule_id`,
and `repaired`.

## Optional Z3 constraint verification

Set `verify_constraints=True` on `session.symbolic.fit` when `z3-solver` is installed
(via `buildml[symbolic-industry]`). This runs a **lite SAT check** on hard
constraint antecedents: not a complete rule-set consistency prover or SMT
product.

## Leakage discipline

- Require `SplitPlan` before fit.
- Fit / induction / conformal-style carves: **train only**.
- Holdout partitions: evaluate / predict only.
- Bundles store the plan; Session checkpoints do **not**.

## Honesty boundary

- Structured if-then rules over columns: readable, auditable.
- Industry backends export interpretable models as rules: not Prolog products.
- **Not** an AGI symbolic reasoner.
- **Not** Prolog, ASP, or a full Z3 SMT product.
- **Not** a fuzzy-logic product or full expert-system suite.
- Neuro-symbolic here means sklearn/torch lite + rules hybrid: **not** Logic
  Tensor Networks / differentiable theorem proving.

## Anti-patterns

- Inducing rules on the full frame before `split`.
- Calling `fit()` then hand-applying rules outside Session (no shared plan,
  no traces, no bundle).
- Treating induced rules as causal laws.
- Expecting `checkpoint_load` to restore `SymbolicPlan`.
- Claiming Z3 lite verification proves global rule-set correctness.

## Bundle boundary

See [Artifacts](artifacts-checkpoints-bundles.md). Format:
`buildml.symbolic_bundle.v1` (`meta.json` + `symbolic_plan.joblib`).
`meta.kind` is `symbolic` or `neuro_symbolic`.

## Benchmark

`benchmarks/symbolic/rule_fidelity.py` compares symbolic holdout accuracy to a
black-box RandomForest baseline and reports a fidelity ratio plus rule coverage.

## Related

- [Quickstart](quickstart-symbolic.md)
- [Leakage / CV](leakage-cv-recipes.md)
- Example: `examples/symbolic_rules_loop.py`
