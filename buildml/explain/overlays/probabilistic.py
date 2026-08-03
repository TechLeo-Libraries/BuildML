# ruff: noqa: E501, F401
"""Probabilistic / Bayesian Session operation overlays (human teaching prose)."""

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

PROBABILISTIC_PLAN = Prerequisite(
    "probabilistic-plan",
    "A train-fitted ProbabilisticPlan is attached to the Session.",
    check_hint="Session.probabilistic_plan is not None.",
)

_OPERATIONS: tuple[OperationSpec, ...] = (
    _operation(
        "fit_probabilistic",
        OperationKind.MODEL,
        "Fit a Bayesian / probabilistic estimator with uncertainty.",
        "Train sklearn BayesianRidge / GP / GaussianNB, MAPIE conformal, or NGBoost; optional train-only conformal.",
        "Probabilistic fit step.",
        (
            "Require a SplitPlan and exactly one target.",
            "Optionally carve a conformal calibration subset from train only.",
            "Fit the estimator on the fit carve (or full train).",
            "Store predictive uncertainty contract + conformal quantile when enabled.",
            "Never use validation/test for fit or conformal calibration.",
        ),
        parameters=(
            _p(
                "backend",
                "native | mapie | ngboost | None",
                "Probabilistic backend; inferred from estimator when omitted.",
                "native",
            ),
            _p(
                "estimator",
                "bayesian_ridge | gaussian_process_regressor | "
                "gaussian_process_classifier | gaussian_nb | split | cv_plus | "
                "jackknife_plus | ngboost_regressor | ngboost_classifier",
                "Estimator or MAPIE conformal method.",
                "bayesian_ridge",
            ),
            _p(
                "task",
                "classification | regression | None",
                "Inferred from estimator when omitted.",
            ),
            _p("columns", "list[str] | None", "Optional explicit numeric feature columns."),
            _p("random_state", "int | None", "RNG seed for conformal carve / GP.", 0),
            _p("alpha", "float", "Miscoverage level for intervals/sets.", 0.1),
            _p(
                "conformal",
                "bool",
                "Enable train-only split conformal calibration.",
                True,
            ),
            _p(
                "conformal_calibration_fraction",
                "float",
                "Fraction of train carved for conformal calibration.",
                0.2,
            ),
            _p(
                "interval_method",
                "posterior_std | split_conformal | both | none | None",
                "Interval construction policy (inferred when omitted).",
            ),
            _p(
                "prefer_reduce_components",
                "bool",
                "Prefer reduce components when a ReducePlan is attached.",
                True,
            ),
            _p(
                "n_restarts_optimizer",
                "int",
                "GP kernel optimizer restarts (0 keeps runs cheap/deterministic).",
                0,
            ),
            _p(
                "n_estimators",
                "int",
                "NGBoost boosting rounds when backend='ngboost'.",
                100,
            ),
            _p(
                "learning_rate",
                "float",
                "NGBoost learning rate when backend='ngboost'.",
                0.05,
            ),
        ),
        inputs=("Session dataset with split and target.",),
        outputs=("ProbabilisticFitResult; Session.probabilistic_plan attached.",),
        prerequisites=(DATASET, ROLES, SPLIT),
        ordering=("After ingest → set_roles → split → optional scale/reduce.",),
        alternatives=(
            "fit for classical point estimators; calibration for classical reliability diagnostics.",
        ),
        rationale=(
            "Use when you need predictive intervals / NLL beyond point metrics."
        ,),
        assumptions=(
            "Numeric non-null features; enough train rows for optional conformal.",
        ),
        failures=(
            "No split; null features; too few train rows for conformal carve.",
        ),
        leakage=(
            "Using validation/test rows for fit or conformal calibration.",
        ),
        anti_patterns=(
            "Claiming PyMC/Stan MCMC or Bayesian deep nets from this path.",
            "Silent conformal calibration on Session test.",
        ),
        state_changes=(
            "Stores probabilistic_plan and fit result; clears prior eval/predict/interval slots.",
        ),
        result_reading=(
            "Read n_fit_rows, n_conformal_calib_rows, conformal_quantile, disclosures.",
        ),
        next_steps=(
            "predict_interval → evaluate_probabilistic; optionally save_probabilistic_bundle.",
        ),
        concepts=(
            "probabilistic-uncertainty",
            "probabilistic-bayesian-ridge",
            "probabilistic-gaussian-process",
            "probabilistic-split-conformal",
            "probabilistic-mapie",
            "probabilistic-ngboost",
            "leakage-boundary",
        ),
    ),
    _operation(
        "evaluate_probabilistic",
        OperationKind.DIAGNOSTIC,
        "Evaluate probabilistic predictions with proper scoring rules.",
        "Holdout NLL / coverage / Brier; never for fit or conformal calibration.",
        "Probabilistic holdout evaluation.",
        (
            "Require an attached ProbabilisticPlan.",
            "Score point metrics plus NLL / interval or set coverage.",
        ),
        parameters=(
            _p(
                "partition",
                "train | validation | test | all",
                "Evaluation partition (validation falls back to test if absent).",
                "validation",
            ),
            _p("alpha", "float | None", "Override miscoverage for interval metrics."),
        ),
        inputs=("ProbabilisticPlan + holdout partition.",),
        outputs=("ProbabilisticEvalResult with metrics and coverage.",),
        prerequisites=(DATASET, PROBABILISTIC_PLAN),
        ordering=("After fit_probabilistic or load_probabilistic_bundle.",),
        alternatives=(
            "Session.calibration for classical fit(...) reliability curves.",
        ),
        rationale=("Use to quantify calibration and interval honesty on holdout.",),
        assumptions=("Feature/target columns match the plan contract.",),
        failures=("No plan; empty partition; missing columns.",),
        leakage=(
            "Tuning alpha / conformal fraction against the reported test coverage repeatedly without a locked protocol.",
        ),
        anti_patterns=(
            "Reporting train coverage as generalization.",
        ),
        state_changes=("Stores probabilistic_eval_result.",),
        result_reading=(
            "Read metrics (nll, interval_coverage / set_coverage, brier/ece), disclosures.",
        ),
        next_steps=("predict_interval; save_probabilistic_bundle.",),
        concepts=(
            "probabilistic-uncertainty",
            "probabilistic-split-conformal",
            "evaluation-partitions",
            "diagnostic-uncertainty",
        ),
    ),
    _operation(
        "predict_probabilistic",
        OperationKind.DIAGNOSTIC,
        "Predict with the probabilistic estimator (optional std / proba).",
        "Point predictions without mutating the plan.",
        "Probabilistic predict step.",
        (
            "Require an attached ProbabilisticPlan.",
            "Run predict / predict_proba / return_std as requested.",
        ),
        parameters=(
            _p(
                "partition",
                "train | validation | test | all",
                "Partition to score.",
                "test",
            ),
            _p("return_std", "bool", "Include posterior std when supported.", True),
            _p("return_proba", "bool", "Include class probabilities when supported.", True),
        ),
        inputs=("Active ProbabilisticPlan.",),
        outputs=("ProbabilisticPredictResult.",),
        prerequisites=(DATASET, PROBABILISTIC_PLAN),
        ordering=("After fit_probabilistic.",),
        alternatives=("predict_interval when you need bands/sets.",),
        rationale=("Use for inference snapshots with optional uncertainty columns.",),
        assumptions=("Feature columns match the plan contract.",),
        failures=("No plan; missing columns; null features.",),
        leakage=("None inherent — still do not train on predictions.",),
        anti_patterns=("Treating std as conformal coverage without predict_interval.",),
        state_changes=("Stores probabilistic_predict_result.",),
        result_reading=("Read predictions / std / probabilities and disclosures.",),
        next_steps=("predict_interval or evaluate_probabilistic.",),
        concepts=("probabilistic-uncertainty", "probabilistic-bayesian-ridge"),
    ),
    _operation(
        "predict_interval",
        OperationKind.DIAGNOSTIC,
        "Predictive intervals or conformal prediction sets.",
        "posterior_std and/or train-calibrated split conformal bands/sets.",
        "Probabilistic interval step.",
        (
            "Require an attached ProbabilisticPlan.",
            "Build regression intervals or classification prediction sets.",
        ),
        parameters=(
            _p(
                "partition",
                "train | validation | test | all",
                "Partition to score.",
                "test",
            ),
            _p("alpha", "float | None", "Miscoverage override (prefer re-fit to change)."),
            _p(
                "method",
                "posterior_std | split_conformal | both | None",
                "Interval construction override.",
            ),
        ),
        inputs=("Active ProbabilisticPlan.",),
        outputs=("ProbabilisticIntervalResult.",),
        prerequisites=(DATASET, PROBABILISTIC_PLAN),
        ordering=("After fit_probabilistic.",),
        alternatives=("evaluate_probabilistic for coverage metrics.",),
        rationale=("Use when decisions need bands or prediction sets.",),
        assumptions=("Conformal quantile present when method needs it.",),
        failures=("No plan; conformal requested but not calibrated.",),
        leakage=("Recalibrating intervals on the scored holdout partition.",),
        anti_patterns=("Calling Gaussian std bands distribution-free without conformal.",),
        state_changes=("Stores probabilistic_interval_result.",),
        result_reading=("Read lower/upper or prediction_sets, method, disclosures.",),
        next_steps=("evaluate_probabilistic; save_probabilistic_bundle.",),
        concepts=(
            "probabilistic-split-conformal",
            "probabilistic-uncertainty",
        ),
    ),
    _operation(
        "save_probabilistic_bundle",
        OperationKind.PERSIST,
        "Persist the active ProbabilisticPlan as buildml.probabilistic_bundle.v1.",
        "Write a domain bundle distinct from Session checkpoints.",
        "Probabilistic bundle save.",
        ("Require an attached plan.", "Write meta.json + probabilistic_plan.joblib."),
        parameters=(_p("path", "str | Path", "Destination directory.", required=True),),
        inputs=("Active ProbabilisticPlan.",),
        outputs=("Bundle directory path.",),
        prerequisites=(PROBABILISTIC_PLAN,),
        ordering=("After a successful fit_probabilistic.",),
        alternatives=("Session.checkpoint_save for workflow resume without the learner.",),
        rationale=("Use when the uncertainty model must travel separately.",),
        assumptions=("Destination is writable.",),
        failures=("No plan attached.",),
        leakage=("Bundles do not embed holdout rows.",),
        anti_patterns=("Assuming a Session checkpoint embeds the ProbabilisticPlan.",),
        state_changes=("History records save_probabilistic_bundle.",),
        result_reading=("Confirm meta.json format buildml.probabilistic_bundle.v1.",),
        next_steps=("load_probabilistic_bundle in another Session.",),
        concepts=("probabilistic-bundle-boundary",),
    ),
    _operation(
        "load_probabilistic_bundle",
        OperationKind.PERSIST,
        "Load a buildml.probabilistic_bundle.v1 ProbabilisticPlan into the Session.",
        "Restore a probabilistic model without re-fitting.",
        "Probabilistic bundle load.",
        (
            "Validate bundle format.",
            "Attach ProbabilisticPlan; clear fit/eval/predict/interval slots.",
        ),
        parameters=(
            _p("path", "str | Path", "Bundle directory.", required=True),
            _p(
                "trusted",
                "bool",
                "Must be True to deserialize pickle/joblib/torch payloads (default False).",
                False,
            ),
        ),
        inputs=("Bundle directory with meta.json + probabilistic_plan.joblib.",),
        outputs=("Session with probabilistic_plan attached.",),
        prerequisites=(DATASET,),
        ordering=("After ingest/roles/split aligned with the plan feature contract.",),
        alternatives=("fit_probabilistic to learn a new plan.",),
        rationale=("Use to resume an uncertainty model.",),
        assumptions=("Feature/target columns still match the plan contract.",),
        failures=("Incomplete or wrong-format bundle.",),
        leakage=("Do not treat load as permission to train on holdout rows.",),
        anti_patterns=("Loading into a Session whose features drifted from the plan.",),
        state_changes=(
            "Sets probabilistic_plan; clears fit/eval/predict/interval result slots.",
        ),
        result_reading=("Inspect Session.probabilistic_plan.to_dict().",),
        next_steps=("evaluate_probabilistic / predict_interval.",),
        concepts=("probabilistic-bundle-boundary",),
    ),
)
