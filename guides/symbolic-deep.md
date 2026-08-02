# Symbolic + Neuro-symbolic deep guide

## What shipped

BuildML’s symbolic path is a **Session-native tabular rule engine** plus a
**neuro-symbolic hybrid** that binds a sklearn estimator to those rules.

| Surface | Role |
| --- | --- |
| `fit_symbolic` | Compile declared rules or induce from train |
| `predict_symbolic` | Decision-list inference + `RuleTrace` |
| `evaluate_symbolic` | Holdout accuracy/RMSE + rule coverage |
| `fit_neuro_symbolic` | sklearn + rules in one API |
| `predict_neuro_symbolic` / `evaluate_neuro_symbolic` | Hybrid score / metrics |
| `save_symbolic_bundle` / `load_symbolic_bundle` | `buildml.symbolic_bundle.v1` |

## Rule sources (disclose provenance)

| `source` / `rule_source` | Provenance | Learning? |
| --- | --- | --- |
| `declared` | Expert / caller | No |
| `decision_tree` | `induced_tree` | Yes — train only |
| `decision_list` | `induced_list` | Yes — sequential covering, train only |

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

## Leakage discipline

- Require `SplitPlan` before fit.
- Fit / induction / conformal-style carves: **train only**.
- Holdout partitions: evaluate / predict only.
- Bundles store the plan; Session checkpoints do **not**.

## Honesty boundary

- Structured if-then rules over columns — readable, auditable.
- **Not** an AGI symbolic reasoner.
- **Not** Prolog, ASP, or Z3 (and none are required in core).
- **Not** a fuzzy-logic product or full expert-system suite.
- Neuro-symbolic here means sklearn + rules hybrid — **not** Logic Tensor
  Networks / differentiable theorem proving.

## Anti-patterns

- Inducing rules on the full frame before `split`.
- Calling `fit()` then hand-applying rules outside Session (no shared plan,
  no traces, no bundle).
- Treating induced rules as causal laws.
- Expecting `checkpoint_load` to restore `SymbolicPlan`.

## Bundle boundary

See [Artifacts](artifacts-checkpoints-bundles.md). Format:
`buildml.symbolic_bundle.v1` (`meta.json` + `symbolic_plan.joblib`).
`meta.kind` is `symbolic` or `neuro_symbolic`.

## Related

- [Quickstart](quickstart-symbolic.md)
- [Leakage / CV](leakage-cv-recipes.md)
- Example: `examples/symbolic_rules_loop.py`
