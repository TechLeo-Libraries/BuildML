# ruff: noqa: E501, F401
"""Decision / optimisation Session operation overlays."""

from __future__ import annotations

from buildml.explain.overlays._common import (
    DATASET,
    ROLES,
    SPLIT,
    OperationKind,
    _operation,
    _p,
)
from buildml.explain.schemas import OperationSpec, Prerequisite

DECISION_PLAN = Prerequisite(
    "decision-plan",
    "A fitted DecisionPlan is attached.",
    check_hint="Session.decision_plan is not None.",
)
FITTED = Prerequisite(
    "fitted-estimator",
    "A classical FitResult is attached (for model-score methods).",
    check_hint="Session has a fitted estimator.",
)

_OPERATIONS: tuple[OperationSpec, ...] = (
    _operation(
        "fit_decision_policy",
        OperationKind.MODEL,
        "Fit a decision policy (threshold, cost matrix, or allocation).",
        "Select operating point / allocation rules on train or validation.",
        "Decision-policy fit step.",
        (
            "Require SplitPlan.",
            "Refuse partition='test' unless allow_test_tuning=True.",
            "threshold: wrap classical threshold_report; persist DecisionPlan.",
            "cost_matrix: Bayes action under user C.",
                "topk/knapsack/lp_allocate: constrained selection over scores.",
                "backend= routes to PuLP/OR-Tools MIP, CVXPY LP, or XGB/calibrated thresholds.",
            ),
        parameters=(
            _p(
                "backend",
                "native | pulp | ortools | cvxpy | calibrated | xgb",
                "Solver/scorer backend (see decision_capability_matrix).",
                None,
            ),
            _p(
                "method",
                "threshold | cost_matrix | topk | knapsack | lp_allocate",
                "Decision helper family.",
                "threshold",
            ),
            _p(
                "partition",
                "train | validation | test",
                "Tuning partition (default validation).",
                "validation",
            ),
            _p(
                "allow_test_tuning",
                "bool",
                "Dangerous opt-in to tune on Session test.",
                False,
            ),
            _p("fp_cost", "float | None", "False-positive cost (threshold)."),
            _p("fn_cost", "float | None", "False-negative cost (threshold)."),
            _p("tp_benefit", "float", "True-positive benefit.", 0.0),
            _p("tn_benefit", "float", "True-negative benefit.", 0.0),
            _p("cost_matrix", "list[list[float]] | None", "C[true, action]."),
            _p("class_labels", "list[str] | None", "Labels aligning cost_matrix."),
            _p("capacity", "int | None", "Top-K capacity."),
            _p("budget", "float | None", "Knapsack / LP budget."),
            _p(
                "score_source",
                "model_proba | model_decision_function | column",
                "Score origin for allocation.",
                "model_proba",
            ),
            _p("score_column", "str | None", "Candidate score column."),
            _p("cost_column", "str | None", "Per-row selection cost."),
            _p("value_column", "str | None", "Knapsack value column."),
            _p("id_column", "str | None", "Candidate id column."),
            _p("knapsack_solver", "dp | greedy", "0-1 knapsack solver.", "dp"),
            _p(
                "objective",
                "maximize_score | maximize_value | minimize_cost",
                "Allocation objective.",
                "maximize_score",
            ),
            _p("min_score", "float | None", "Optional score floor."),
            _p("lp_max_fraction", "float", "Max fraction per item in LP.", 1.0),
        ),
        inputs=("Split + fitted estimator and/or score/cost columns.",),
        outputs=("DecisionPlan + DecisionFitResult.",),
        prerequisites=(DATASET, ROLES, SPLIT),
        ordering=(
            "After split (+ fit for model-score methods); before apply/evaluate."
        ,),
        alternatives=(
            "Session.tune_threshold for diagnostic threshold sweeps without a persisted plan.",
        ),
        rationale=(
            "Unify cost-sensitive thresholds and constrained allocations as a Session path."
        ,),
        assumptions=("Split present; costs non-negative when provided.",),
        failures=(
            "No split; test tuning without opt-in; missing capacity/budget; no proba.",
        ),
        leakage=(
            "Default validation tuning; test requires allow_test_tuning=True + disclosure.",
        ),
        anti_patterns=(
            "Calling this a general OR / MIP / digital-twin platform.",
            "Silently selecting thresholds on test.",
            "Assuming this replaces Optuna HPO.",
        ),
        state_changes=(
            "Stores decision_plan; clears prior apply/eval; may set last_diagnostic "
            "for method='threshold'."
        ,),
        result_reading=(
            "Inspect threshold / expected_cost / n_selected and disclosures."
        ,),
        next_steps=("apply_decisions; evaluate_decisions; save_decision_bundle.",),
        concepts=(
            "decision-operating-point",
            "decision-cost-matrix",
            "decision-allocation",
            "decision-bundle-boundary",
            "leakage-boundary",
        ),
    ),
    _operation(
        "apply_decisions",
        OperationKind.MODEL,
        "Apply a frozen DecisionPlan to a partition without retuning it.",
        "Produce the actions the tuned policy implies: labels, selected candidate ids, or allocations.",
        "Decision apply step.",
        (
            "Require DecisionPlan.",
            "threshold/cost_matrix: score partition with fitted estimator.",
            "allocation: select under capacity/budget on partition or candidates.",
        ),
        parameters=(
            _p(
                "partition",
                "train | validation | test | all | None",
                "Rows to score when candidates omitted.",
                "test",
            ),
            _p(
                "candidates",
                "DataFrame | None",
                "Explicit candidate table for allocation methods.",
            ),
        ),
        inputs=("Frozen DecisionPlan + partition or candidates.",),
        outputs=("ApplyDecisionsResult.",),
        prerequisites=(DECISION_PLAN,),
        ordering=("After fit_decision_policy.",),
        alternatives=("evaluate_decisions for labeled metrics.",),
        rationale=("Deploy the frozen operating policy.",),
        assumptions=("Compatible features for model-score methods.",),
        failures=("No DecisionPlan; missing FitResult for threshold/cost_matrix.",),
        leakage=("Applying is not fitting; iterating on test still peeks.",),
        anti_patterns=("Refitting the policy on test apply metrics.",),
        state_changes=("Stores decision_apply_result.",),
        result_reading=("Inspect decisions / selected_ids / selected_value.",),
        next_steps=("evaluate_decisions; save_decision_bundle.",),
        concepts=("decision-operating-point", "decision-allocation"),
    ),
    _operation(
        "evaluate_decisions",
        OperationKind.DIAGNOSTIC,
        "Evaluate a frozen DecisionPlan on a holdout partition.",
        "Realized cost / classification / allocation utilization metrics.",
        "Decision eval step.",
        (
            "Require DecisionPlan.",
            "Apply frozen policy; compute metrics on the named partition.",
        ),
        parameters=(
            _p(
                "partition",
                "train | validation | test",
                "Evaluation partition.",
                "test",
            ),
        ),
        inputs=("Frozen DecisionPlan + labeled/scoreable partition.",),
        outputs=("DecisionEvalResult.",),
        prerequisites=(DECISION_PLAN,),
        ordering=("After fit_decision_policy.",),
        alternatives=("apply_decisions when labels are unavailable.",),
        rationale=("Confirm the frozen policy once on untouched test.",),
        assumptions=("Split present; labels available for threshold/cost_matrix.",),
        failures=("No DecisionPlan; evaluating without labels where required.",),
        leakage=(
            "Repeated test evaluation can still overfit human iteration; "
            "prefer one confirmation."
        ,),
        anti_patterns=("Retuning the policy after peeking at test evaluate_decisions.",),
        state_changes=("Stores decision_eval_result.",),
        result_reading=("Inspect metrics and realized_cost.",),
        next_steps=("save_decision_bundle.",),
        concepts=(
            "decision-operating-point",
            "decision-cost-matrix",
            "decision-allocation",
            "leakage-boundary",
        ),
    ),
    _operation(
        "save_decision_bundle",
        OperationKind.PERSIST,
        "Persist DecisionPlan as buildml.decision_bundle.v1.",
        "Write meta.json + decision_plan.joblib.",
        "Decision bundle save.",
        ("Require DecisionPlan.", "Write bundle directory."),
        parameters=(_p("path", "str | Path", "Destination directory."),),
        inputs=("DecisionPlan.",),
        outputs=("Path.",),
        prerequisites=(DECISION_PLAN,),
        ordering=("After fit_decision_policy.",),
        alternatives=("Session checkpoint for data/history (does not embed plan).",),
        rationale=("Ship the operating policy separately from Session state.",),
        assumptions=("Writable path.",),
        failures=("No DecisionPlan.",),
        leakage=("Bundles do not embed Session test labels.",),
        anti_patterns=("Assuming checkpoint_save includes the DecisionPlan.",),
        state_changes=("Filesystem write; history record.",),
        result_reading=("Confirm format buildml.decision_bundle.v1.",),
        next_steps=("load_decision_bundle on a restored Session.",),
        concepts=("decision-bundle-boundary",),
    ),
    _operation(
        "load_decision_bundle",
        OperationKind.MODEL,
        "Load a buildml.decision_bundle.v1 plan into the Session.",
        "Restore DecisionPlan for apply/evaluate.",
        "Decision bundle load.",
        ("Read meta.json + decision_plan.joblib.", "Attach DecisionPlan."),
        parameters=(
            _p("path", "str | Path", "Bundle directory.", required=True),
            _p(
                "trusted",
                "bool",
                "Must be True to deserialize pickle/joblib/torch payloads (default False).",
                False,
            ),
        ),
        inputs=("Decision bundle directory.",),
        outputs=("Session with decision_plan.",),
        prerequisites=(DATASET,),
        ordering=("Anytime a bundle exists; model-score apply still needs fit.",),
        alternatives=("fit_decision_policy to create a new plan.",),
        rationale=("Reload a previously selected operating policy.",),
        assumptions=("Compatible bundle format.",),
        failures=("Incomplete or wrong-format bundle.",),
        leakage=("Loading does not re-open Session test for retuning.",),
        anti_patterns=("Loading then retuning on test without disclosure.",),
        state_changes=("Stores decision_plan; clears fit/apply/eval decision results.",),
        result_reading=("Inspect decision_plan.method / threshold.",),
        next_steps=("apply_decisions; evaluate_decisions.",),
        concepts=("decision-bundle-boundary",),
    ),
)
