# ruff: noqa: E501
"""Decision / optimisation concept notes."""

from __future__ import annotations

from buildml.explain.concepts._builder import _note
from buildml.explain.schemas import ConceptNote

OPTIMIZE_NOTES: dict[str, ConceptNote] = {
    note.key: note
    for note in (
        _note(
            key="decision-operating-point",
            title="Cost-sensitive operating points",
            summary=(
                "session.decision.fit(method='threshold') selects a binary "
                "cutoff on validation using the same engine as tune_threshold."
            ),
            definition=(
                "An operating point maps scores to actions (e.g. proba ≥ t → "
                "positive). With FP/FN costs, choose t minimizing expected cost "
                "on the tuning partition."
            ),
            intuition=(
                "F1 picks a balanced cutoff; when false negatives are 10× more "
                "expensive, the cost-optimal threshold usually moves lower."
            ),
            formal_idea=(
                "min_t  fp_cost·FP(t) + fn_cost·FN(t) − tp_benefit·TP(t) − tn_benefit·TN(t)."
            ),
            why_it_matters=(
                "Selecting t on test leaks holdout feedback into the policy.",
                "Persisting a DecisionPlan separates diagnostics from deployment.",
            ),
            how_buildml_uses=(
                "session.decision.fit(method='threshold', partition='validation', "
                "fp_cost=..., fn_cost=...); classical Session.tune_threshold remains.",
            ),
            interpretation_rules=(
                "Prefer validation for selection; session.decision.evaluate on test once.",
                "allow_test_tuning=True is a dangerous opt-in with disclosure.",
            ),
            assumptions=("Binary probabilistic classifier; split present.",),
            failure_modes=("Tuning on test; omitting costs then over-trusting F1.",),
            anti_patterns=(
                "Calling this a general OR solver.",
                "Replacing tune_threshold diagnostics with silent test peeks.",
            ),
            worked_example_pattern=(
                "fit → session.decision.fit(method='threshold', partition='validation', "
                "fp_cost=1, fn_cost=5) → session.decision.evaluate(partition='test').",
            ),
            related_concepts=(
                "decision-cost-matrix",
                "decision-allocation",
                "decision-bundle-boundary",
                "leakage-boundary",
            ),
        ),
        _note(
            key="decision-cost-matrix",
            title="Multiclass Bayes decisions under a cost matrix",
            summary=(
                "method='cost_matrix' chooses action a minimizing "
                "Σ_y P(y|x) C[y,a] from predict_proba."
            ),
            definition=(
                "Given a square cost matrix C indexed by true class and action, "
                "the Bayes action for x is argmin_a Σ_y P(y|x) C[y,a]."
            ),
            intuition=(
                "If confusing class A for B is cheap but B for A is costly, "
                "the policy becomes asymmetric even with equal probabilities."
            ),
            formal_idea=("a*(x) = argmin_a  p(x)^T C_{·,a}."),
            why_it_matters=(
                "User supplies C: BuildML does not invent business costs from test labels.",
            ),
            how_buildml_uses=(
                "session.decision.fit(method='cost_matrix', cost_matrix=...).",
            ),
            interpretation_rules=(
                "Rows = true class, columns = action; align class_labels with estimator.classes_.",
            ),
            assumptions=("Classification with predict_proba; square cost_matrix.",),
            failure_modes=("Mismatched label order; estimating C from the test set.",),
            anti_patterns=("Treating this as causal decision analysis.",),
            worked_example_pattern=(
                "fit multiclass → session.decision.fit(method='cost_matrix', "
                "cost_matrix=[[0,1,5],[2,0,1],[5,2,0]]) → session.decision.evaluate.",
            ),
            related_concepts=(
                "decision-operating-point",
                "decision-allocation",
                "decision-bundle-boundary",
            ),
        ),
        _note(
            key="decision-allocation",
            title="Top-K, knapsack-lite, and LP allocation",
            summary=(
                "Constrained selection over model or column scores: top-K capacity, "
                "0-1 knapsack-lite, or continuous LP budget shares."
            ),
            definition=(
                "topk: select ≤K highest scores. knapsack: maximize value under "
                "budget (DP when costs near-integral, else greedy). lp_allocate: "
                "fractional x_i via scipy.optimize.linprog."
            ),
            intuition=(
                "Review the top 50 fraud alerts under analyst capacity; or spend "
                "a fixed outreach budget on the highest-scoring leads."
            ),
            formal_idea=(
                "max Σ v_i x_i  s.t. Σ c_i x_i ≤ B, x_i∈{0,1} or x_i∈[0,f_max]."
            ),
            why_it_matters=(
                "Capacity/budget constraints turn scores into decisions without a MIP suite.",
            ),
            how_buildml_uses=(
                "session.decision.fit(method='topk'|'knapsack'|'lp_allocate', ...).",
            ),
            interpretation_rules=(
                "LP is continuous: not integer MIP.",
                "Greedy knapsack is approximate; DP discloses when used.",
            ),
            assumptions=("Non-negative costs; scores/values finite.",),
            failure_modes=("Huge DP state → greedy fallback; zero budget.",),
            anti_patterns=(
                "Calling this PuLP/OR-Tools or a digital twin.",
                "Fitting allocations on test without allow_test_tuning.",
            ),
            worked_example_pattern=(
                "fit → session.decision.fit(method='knapsack', partition='validation', "
                "budget=100, cost_column='cost') → session.decision.apply(partition='test').",
            ),
            related_concepts=(
                "decision-operating-point",
                "decision-cost-matrix",
                "decision-bundle-boundary",
            ),
        ),
        _note(
            key="decision-bundle-boundary",
            title="Decision bundles vs Session checkpoints",
            summary=(
                "buildml.decision_bundle.v1 stores a DecisionPlan; Session "
                "checkpoints do not embed it."
            ),
            definition=(
                "A decision bundle persists thresholds, cost matrices, and "
                "allocation rules. Reload via session.decision.load_bundle after "
                "restoring data/fit as needed."
            ),
            intuition=(
                "Ship the operating policy separately from the raw Session "
                "history: like a recommender bundle vs a checkpoint."
            ),
            formal_idea=("DecisionPlan ⊄ SessionCheckpoint."),
            why_it_matters=(
                "Prevents silent loss of tuned thresholds across restarts.",
            ),
            how_buildml_uses=(
                "session.decision.save_bundle / session.decision.load_bundle.",
            ),
            interpretation_rules=(
                "Bundles do not replace FitResult; threshold/cost_matrix apply "
                "still needs Session.fit on compatible features.",
            ),
            assumptions=("Compatible feature columns for model-score methods.",),
            failure_modes=("Loading a policy onto an unfitted Session for threshold apply.",),
            anti_patterns=("Assuming checkpoint_save embeds the DecisionPlan.",),
            worked_example_pattern=(
                "session.decision.fit → session.decision.save_bundle(path) → "
                "session.decision.load_bundle(path) → session.decision.apply.",
            ),
            related_concepts=(
                "decision-operating-point",
                "decision-allocation",
                "leakage-boundary",
            ),
        ),
    )
}
