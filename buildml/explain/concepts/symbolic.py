# ruff: noqa: E501
"""Symbolic / neuro-symbolic concept notes."""

from __future__ import annotations

from buildml.explain.concepts._builder import _note
from buildml.explain.schemas import ConceptNote

SYMBOLIC_NOTES: dict[str, ConceptNote] = {
    note.key: note
    for note in (
        _note(
            key="symbolic-rules",
            title="Tabular if-then rule knowledge bases",
            summary="Symbolic AI here means explicit predicates over columns compiled into an ordered rule list with explanation traces: not an AGI reasoner.",
            definition=(
                "A RuleKnowledgeBase is an ordered list of if-then rules. Each "
                "rule has AND-ed antecedents (column predicates) and a "
                "consequent (class label, regression value, or constraint "
                "action). Decision-list semantics: first matching rule wins; "
                "else default_consequent."
            ),
            intuition=(
                "Readable rules like 'if age > 60 and risk == high then deny' "
                "that you can audit: not a black-box neural net and not a "
                "full Prolog theorem prover."
            ),
            formal_idea=(
                "Decision list: for rules r1..rk, predict consequent of the "
                "first ri whose antecedents hold; else default."
            ),
            why_it_matters=(
                "Rules give explanation traces (which rules fired).",
                "Provenance must disclose declared vs train-induced.",
            ),
            how_buildml_uses=(
                "Session.fit_symbolic → predict_symbolic / evaluate_symbolic.",
            ),
            interpretation_rules=(
                "Read provenance on the knowledge base (declared / induced_tree / induced_list).",
                "Inspect RuleTrace.fired_rule_ids and chosen_rule_id.",
            ),
            assumptions=("Predicates reference columns present after preprocess.",),
            failure_modes=(
                "Rules referencing missing columns; empty rule lists without default.",
            ),
            anti_patterns=(
                "Calling this an AGI symbolic reasoner.",
                "Requiring Prolog/Z3 in core for basic tabular rules.",
            ),
            worked_example_pattern=(
                "fit_symbolic(source='decision_tree') → "
                "predict_symbolic(return_traces=True).",
            ),
            related_concepts=(
                "symbolic-induction",
                "symbolic-traces",
                "neuro-symbolic-hybrid",
                "symbolic-bundle-boundary",
                "leakage-boundary",
            ),
        ),
        _note(
            key="symbolic-induction",
            title="Declared vs train-induced rules",
            summary="Expert-declared rules are not learned; decision_tree and decision_list induce rules from Session train only.",
            definition=(
                "source='declared' compiles caller rules. source='decision_tree' "
                "fits a shallow sklearn DecisionTree on train and exports "
                "root-to-leaf paths. source='decision_list' uses sequential "
                "covering with shallow stumps (RIPPER-/CN2-lite)."
            ),
            intuition=(
                "Either you write the rules, or the train set writes them: "
                "and BuildML tells you which."
            ),
            formal_idea=(
                "Tree path export: each leaf path ∧(feature ⋈ threshold) → label. "
                "Covering: greedily add high-support rules, remove covered rows."
            ),
            why_it_matters=(
                "Mixing expert and induced rules without disclosure is dishonest.",
                "Induction on validation/test would leak.",
            ),
            how_buildml_uses=(
                "fit_symbolic(source=...); knowledge_base.provenance disclosure.",
            ),
            interpretation_rules=(
                "induced_* means train-only learning; declared means expert.",
            ),
            assumptions=("Enough train rows for tree / covering min_samples_leaf.",),
            failure_modes=("Overfitting shallow trees; unstable covering on tiny data.",),
            anti_patterns=(
                "Treating induced rules as causal laws.",
                "Inducing on the full dataset before split.",
            ),
            worked_example_pattern=(
                "fit_symbolic(source='decision_list', max_rules=16).",
            ),
            related_concepts=(
                "symbolic-rules",
                "leakage-boundary",
                "neuro-symbolic-hybrid",
            ),
        ),
        _note(
            key="symbolic-traces",
            title="Rule-firing explanation traces",
            summary="predict_symbolic returns which rules fired and which rule chose the prediction.",
            definition=(
                "A RuleTrace records fired_rule_ids, chosen_rule_id, prediction, "
                "and optional neural_prediction / repaired flags for hybrids."
            ),
            intuition=(
                "For each row you can answer: which rules matched, and which "
                "one decided the output?"
            ),
            formal_idea=("Trace τ(x) = ({r : r(x)}, argmin priority among firings)."),
            why_it_matters=("Auditability is the point of shipping rules.",),
            how_buildml_uses=(
                "predict_symbolic(..., return_traces=True); evaluate reports rule_coverage.",
            ),
            interpretation_rules=(
                "chosen_rule_id=None means default_consequent was used.",
                "Multiple firings are normal; only the first by priority chooses.",
            ),
            assumptions=("Rules are priority-ordered in the knowledge base.",),
            failure_modes=("Empty traces when return_traces=False.",),
            anti_patterns=("Ignoring traces and treating rules as opaque.",),
            worked_example_pattern=("pred.traces[0].fired_rule_ids",),
            related_concepts=("symbolic-rules", "neuro-symbolic-hybrid"),
        ),
        _note(
            key="neuro-symbolic-hybrid",
            title="Sklearn + symbolic constraints in one Session API",
            summary="Neuro-symbolic modes combine a probabilistic/sklearn model with rule overlay, rules-as-features, or constraint repair: not disconnected ad-hoc calls.",
            definition=(
                "fit_neuro_symbolic trains a base estimator and attaches a "
                "RuleKnowledgeBase. Modes: constraint_overlay (predict then "
                "hard/soft rules), rules_as_features (binary rule firings "
                "concatenated into X), constraint_repair (fix hard violations)."
            ),
            intuition=(
                "Let the model guess, then let domain rules veto or reshape "
                "the answer: or let rules become extra features the model sees."
            ),
            formal_idea=(
                "ŷ = overlay(f_θ(x), R) or f_θ([x ‖ fire(R,x)]) or repair(f_θ(x), R)."
            ),
            why_it_matters=(
                "A real hybrid needs shared Session state, leakage discipline, "
                "and provenance: not 'fit() then manually if-then'.",
            ),
            how_buildml_uses=(
                "Session.fit_neuro_symbolic → predict_neuro_symbolic / evaluate_neuro_symbolic.",
            ),
            interpretation_rules=(
                "Read mode, rule_provenance, and repair_rate on eval.",
                "Soft rules blend/prefer using soft_strength × rule.strength.",
            ),
            assumptions=("Base estimator family matches task (clf vs reg).",),
            failure_modes=(
                "Hard rules that never fire; soft_strength=0 (no effect).",
            ),
            anti_patterns=(
                "Calling fit then applying rules outside Session without traces.",
                "Claiming deep neuro-symbolic research depth (LTN/logic tensors).",
            ),
            worked_example_pattern=(
                "fit_neuro_symbolic(mode='constraint_overlay', rules=[...]).",
            ),
            related_concepts=(
                "symbolic-rules",
                "symbolic-induction",
                "symbolic-traces",
                "symbolic-bundle-boundary",
                "leakage-boundary",
            ),
        ),
        _note(
            key="symbolic-bundle-boundary",
            title="Symbolic bundles vs Session checkpoints",
            summary="buildml.symbolic_bundle.v1 stores SymbolicPlan / NeuroSymbolicPlan; Session checkpoints do not embed them.",
            definition=(
                "save_symbolic_bundle writes meta.json + symbolic_plan.joblib. "
                "Reload via load_symbolic_bundle. Checkpoints keep data/roles/"
                "splits/history only."
            ),
            intuition=(
                "Checkpoint the workflow; bundle the rule base / hybrid learner."
            ),
            formal_idea=("Artifact separation: workflow state ⊥ learner state."),
            why_it_matters=("Prevents silent loss of rules across reattach.",),
            how_buildml_uses=(
                "save_symbolic_bundle / load_symbolic_bundle.",
            ),
            interpretation_rules=(
                "meta.json kind is symbolic or neuro_symbolic.",
            ),
            assumptions=("joblib can pickle the sklearn base estimator.",),
            failure_modes=("Loading wrong format; incomplete bundle directory.",),
            anti_patterns=(
                "Expecting checkpoint_load to restore SymbolicPlan.",
            ),
            worked_example_pattern=(
                "save_symbolic_bundle('artifacts/symbolic_bundle').",
            ),
            related_concepts=(
                "symbolic-rules",
                "neuro-symbolic-hybrid",
                "leakage-boundary",
            ),
        ),
    )
}
