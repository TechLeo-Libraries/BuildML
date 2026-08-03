# ruff: noqa: E501, F401
"""Synthetic-data Session operation overlays."""

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

SYNTHESIZER_PLAN = Prerequisite(
    "synthesizer-plan",
    "A fitted SynthesizerPlan is attached.",
    check_hint="Session.synthesizer_plan is not None.",
)

_OPERATIONS: tuple[OperationSpec, ...] = (
    _operation(
        "fit_synthesizer",
        OperationKind.MODEL,
        "Fit a train-only tabular synthesizer.",
        "Estimate schema + generator (bootstrap / Gaussian copula / SMOTE) on train.",
        "Synthetic fit step.",
        (
            "Require SplitPlan.",
            "Fit generator on train partition only (refuse val/test fit).",
            "bootstrap: row resample (+ optional smooth_sigma noise).",
            "gaussian_copula: mixed-type empirical CDF + correlation latent.",
            "smote: reusable imblearn SMOTE wrap (extra imbalanced).",
            "sdv: CTGAN/TVAE/CopulaGAN when buildml[synthetic-industry] installed.",
        ),
        parameters=(
            _p(
                "backend",
                "native | sdv | None",
                "Synthesizer backend (see synthetic_capability_matrix).",
                None,
            ),
            _p(
                "method",
                "bootstrap | gaussian_copula | smote | ctgan | tvae | copulagan",
                "Generator family.",
                "gaussian_copula",
            ),
            _p("columns", "list[str] | None", "Columns to model (default: features+target)."),
            _p("random_state", "int", "RNG seed.", 42),
            _p(
                "smooth_sigma",
                "float",
                "Bootstrap Gaussian noise scale × column std (0 = plain).",
                0.0,
            ),
            _p(
                "correlation_ridge",
                "float",
                "Ridge added to copula correlation for PSD.",
                1e-3,
            ),
            _p("target_column", "str | None", "Target for SMOTE / TSTR."),
            _p("k_neighbors", "int", "SMOTE k_neighbors.", 5),
            _p(
                "sampling_strategy",
                "str | float | dict",
                "Forwarded to imblearn SMOTE.",
                "auto",
            ),
            _p("epochs", "int", "SDV training epochs.", 300),
            _p("batch_size", "int", "SDV batch size.", 500),
        ),
        inputs=("Split train rows + column roles.",),
        outputs=("SynthesizerPlan + SynthesizerFitResult.",),
        prerequisites=(DATASET, ROLES, SPLIT),
        ordering=("After split; before sample_synthetic / evaluate_synthetic.",),
        alternatives=(
            "Session.resample for class-balance preprocess (mutates train; not a reusable generator).",
        ),
        rationale=(
            "Provide a Session synthetic-data product distinct from imbalance resampling."
        ,),
        assumptions=("Non-empty train; SMOTE needs numeric features + ≥2 classes.",),
        failures=(
            "No split; SMOTE without imbalanced extra; insufficient minority for SMOTE.",
        ),
        leakage=(
            "Always fits on train only; never estimates joints from validation/test.",
        ),
        anti_patterns=(
            "Calling this differential privacy.",
            "Fitting generators on test.",
            "Confusing with Session.resample class balancing.",
            "Requiring heavy SDV/CTGAN stacks for core workflows.",
        ),
        state_changes=(
            "Stores synthesizer_plan; clears prior sample/eval synthetic results."
        ,),
        result_reading=("Inspect method, column_kinds, and privacy disclosures.",),
        next_steps=("sample_synthetic; evaluate_synthetic; save_synthetic_bundle.",),
        concepts=(
            "synthetic-train-only-generator",
            "synthetic-vs-resample",
            "synthetic-privacy-limits",
            "synthetic-bundle-boundary",
            "leakage-boundary",
        ),
    ),
    _operation(
        "sample_synthetic",
        OperationKind.MODEL,
        "Sample rows from a frozen synthesizer.",
        "Return a Frame or explicitly extend train with provenance.",
        "Synthetic sample step.",
        (
            "Require SynthesizerPlan.",
            "Draw n rows from frozen generator.",
            "merge_mode='none' (default): return Frame; do not mutate roles.",
            "merge_mode='extend_train': append to train; mark provenance role=ignore.",
        ),
        parameters=(
            _p("n", "int | None", "Rows to sample (default: train fit size)."),
            _p("random_state", "int | None", "Sample RNG seed."),
            _p(
                "condition",
                "dict | None",
                "Optional categorical condition (gaussian_copula rejection).",
            ),
            _p(
                "merge_mode",
                "none | extend_train",
                "Whether to append into Session train.",
                "none",
            ),
            _p(
                "provenance_column",
                "str",
                "Column marking synthetic rows when merging.",
                "_synthetic",
            ),
            _p(
                "validate",
                "bool",
                "Run built-in validate_synthetic on the sample.",
                False,
            ),
        ),
        inputs=("Frozen SynthesizerPlan.",),
        outputs=("SyntheticSampleResult (+ optional Session train mutation).",),
        prerequisites=(SYNTHESIZER_PLAN, SPLIT),
        ordering=("After fit_synthesizer.",),
        alternatives=("evaluate_synthetic without merging.",),
        rationale=("Keep merge explicit so roles/splits are not silently poisoned.",),
        assumptions=("Compatible generator state.",),
        failures=("No SynthesizerPlan; condition on non-copula methods.",),
        leakage=("Sampling does not refit; merging only extends train, never holdouts.",),
        anti_patterns=(
            "Silently concatenating synthetics into the full frame without provenance.",
            "Treating synthetic rows as real labeled observations without disclosure.",
        ),
        state_changes=(
            "Stores synthetic_sample_result; extend_train rebuilds dataset/split "
            "and clears classical FitResult."
        ,),
        result_reading=("Inspect n_rows, merged, provenance_column, frame.",),
        next_steps=("evaluate_synthetic; save_synthetic_bundle.",),
        concepts=(
            "synthetic-train-only-generator",
            "synthetic-merge-provenance",
            "synthetic-vs-resample",
        ),
    ),
    _operation(
        "evaluate_synthetic",
        OperationKind.DIAGNOSTIC,
        "Evaluate a frozen synthesizer (fidelity or TSTR).",
        "Column fidelity metrics or train-on-synthetic test-on-real utility.",
        "Synthetic eval step.",
        (
            "Require SynthesizerPlan.",
            "Sample synthetic rows from the frozen generator.",
            "fidelity: KS / TV / correlation L1 vs real partition.",
            "tstr: train sklearn estimator on synthetic; score on real holdout.",
        ),
        parameters=(
            _p("mode", "fidelity | tstr", "Evaluation family.", "fidelity"),
            _p(
                "eval_backend",
                "auto | builtin | sdmetrics",
                "Fidelity eval backend (SDMetrics when installed).",
                "auto",
            ),
            _p(
                "partition",
                "train | validation | test",
                "Real partition to compare / score on.",
                "test",
            ),
            _p("n_synthetic", "int | None", "Synthetic draw size."),
            _p("random_state", "int", "Eval sampling seed.", 0),
            _p(
                "estimator",
                "auto | logistic | ridge",
                "TSTR estimator family.",
                "auto",
            ),
        ),
        inputs=("Frozen SynthesizerPlan + real holdout partition.",),
        outputs=("SyntheticEvalResult.",),
        prerequisites=(SYNTHESIZER_PLAN, SPLIT),
        ordering=("After fit_synthesizer.",),
        alternatives=("Manual downstream model training on sample_synthetic frames.",),
        rationale=("Disclose utility vs fidelity so users know what was measured.",),
        assumptions=("Overlapping columns; tstr needs a target.",),
        failures=("No plan; tstr without target; empty partitions.",),
        leakage=(
            "Never refits the generator on the eval partition; prefer test for holdout utility."
        ,),
        anti_patterns=(
            "Calling fidelity a privacy guarantee.",
            "Tuning generator hyperparameters against test evaluate_synthetic repeatedly.",
        ),
        state_changes=("Stores synthetic_eval_result.",),
        result_reading=("Inspect mode, metrics (mean_ks / score), disclosures.",),
        next_steps=("save_synthetic_bundle.",),
        concepts=(
            "synthetic-fidelity-vs-tstr",
            "synthetic-privacy-limits",
            "leakage-boundary",
        ),
    ),
    _operation(
        "save_synthetic_bundle",
        OperationKind.PERSIST,
        "Persist SynthesizerPlan as buildml.synthetic_bundle.v1.",
        "Write meta.json + synthetic_plan.joblib.",
        "Synthetic bundle save.",
        ("Require SynthesizerPlan.", "Write bundle directory."),
        parameters=(_p("path", "str | Path", "Destination directory."),),
        inputs=("SynthesizerPlan.",),
        outputs=("Path.",),
        prerequisites=(SYNTHESIZER_PLAN,),
        ordering=("After fit_synthesizer.",),
        alternatives=("Session checkpoint for data/history (does not embed plan).",),
        rationale=("Ship the fitted generator separately from Session state.",),
        assumptions=("Writable path.",),
        failures=("No SynthesizerPlan.",),
        leakage=("Bundles do not embed Session test labels.",),
        anti_patterns=("Assuming checkpoint_save includes the SynthesizerPlan.",),
        state_changes=("Filesystem write; history record.",),
        result_reading=("Confirm format buildml.synthetic_bundle.v1.",),
        next_steps=("load_synthetic_bundle on a restored Session.",),
        concepts=("synthetic-bundle-boundary",),
    ),
    _operation(
        "load_synthetic_bundle",
        OperationKind.MODEL,
        "Load a buildml.synthetic_bundle.v1 plan into the Session.",
        "Restore SynthesizerPlan for sample/evaluate.",
        "Synthetic bundle load.",
        ("Read meta.json + synthetic_plan.joblib.", "Attach SynthesizerPlan."),
        parameters=(
            _p("path", "str | Path", "Bundle directory.", required=True),
            _p(
                "trusted",
                "bool",
                "Must be True to deserialize pickle/joblib/torch payloads (default False).",
                False,
            ),
        ),
        inputs=("Synthetic bundle directory.",),
        outputs=("Session with synthesizer_plan.",),
        prerequisites=(DATASET,),
        ordering=("Anytime a bundle exists.",),
        alternatives=("fit_synthesizer to create a new plan.",),
        rationale=("Reload a previously fitted generator.",),
        assumptions=("Compatible bundle format.",),
        failures=("Incomplete or wrong-format bundle.",),
        leakage=("Loading does not re-open holdouts for generator refit.",),
        anti_patterns=("Loading then refitting on test.",),
        state_changes=("Stores synthesizer_plan; clears fit/sample/eval synthetic results.",),
        result_reading=("Inspect synthesizer_plan.method / columns.",),
        next_steps=("sample_synthetic; evaluate_synthetic.",),
        concepts=("synthetic-bundle-boundary",),
    ),
)
