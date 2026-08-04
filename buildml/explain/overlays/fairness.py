# ruff: noqa: E501, F401
"""Fairness / disparity Session operation overlays."""

from __future__ import annotations

from buildml.explain.overlays._common import (
    DATASET,
    FIT,
    SPLIT,
    OperationKind,
    _operation,
    _p,
)
from buildml.explain.schemas import OperationSpec

_OPERATIONS: tuple[OperationSpec, ...] = (
    _operation(
        "evaluate_fairness",
        OperationKind.DIAGNOSTIC,
        "Report observational group disparity metrics on a holdout partition.",
        "Inspect demographic parity, disparate impact, equalized-odds gaps, "
        "per-group classical metrics, and optional stability bands after a fit.",
        "Post-fit fairness audit overlay (not automatic mitigation).",
        (
            "Require a fitted classifier and caller-declared sensitive column(s).",
            "Compose intersectional group keys when multiple columns are supplied.",
            "Compute selection rates, disparity gaps, classical per-group metrics.",
            "Optionally resample for disclosed stability bands.",
            "Store the FairnessReport on the Session for later inspection.",
        ),
        parameters=(
            _p(
                "sensitive_column",
                "str | Sequence[str]",
                "Column (or ordered columns for intersectional keys) defining groups.",
                required=True,
            ),
            _p(
                "partition",
                "train | validation | test",
                "Rows used for the audit.",
                "test",
            ),
            _p(
                "positive_label",
                "Any",
                "Label treated as the positive / selected class.",
                1,
            ),
            _p(
                "include_classical_metrics",
                "bool",
                "Attach per-group accuracy/precision/recall/F1 (and AUC when scores exist).",
                True,
            ),
            _p(
                "bootstrap_samples",
                "int",
                "Resample draws for stability bands; 0 disables.",
                0,
            ),
            _p(
                "stability_method",
                "bootstrap | stratified_subsample",
                "Resampling strategy when bootstrap_samples > 1.",
                "bootstrap",
            ),
            _p(
                "confidence_level",
                "float",
                "Central percentile interval for stability bands.",
                0.95,
            ),
            _p(
                "include_scores",
                "bool",
                "Fetch predict_proba scores for per-group ROC-AUC when available.",
                True,
            ),
        ),
        inputs=("Fitted model, labeled partition, and a sensitive attribute column.",),
        outputs=("FairnessReport with rates, gaps, classical bridge, optional stability.",),
        prerequisites=(DATASET, SPLIT, FIT),
        ordering=("After evaluate on a credible model; before claiming deployment readiness.",),
        alternatives=(
            "session.fairness.attach_to_last_eval; external fairness toolkits for legal review; "
            "slice diagnostics via error_slices.",
        ),
        rationale=(
            "Surface group gaps honestly without pretending metrics alone are compliance.",
        ),
        assumptions=(
            "The sensitive column(s) are correctly labeled and present on the chosen partition.",
            "Observational metrics are not causal bias proofs.",
        ),
        failures=(
            "Missing fit, missing sensitive column, empty groups, or invalid positive_label.",
        ),
        leakage=(
            "Auditing on test is fine for reporting; do not retune thresholds on the same test rows.",
        ),
        anti_patterns=(
            "Treating disparate impact as a legal green light.",
            "Hiding groups with tiny n without disclosure.",
            "Claiming intersectional keys with support < 30 are precise.",
        ),
        state_changes=("Stores session.fairness.last_report / Session fairness report; does not change the model.",),
        result_reading=(
            "Read selection rates per group, then DP / DI / equalized-odds gaps with sample sizes; "
            "inspect classical_metrics_by_group and stability bands when present.",
        ),
        next_steps=(
            "Prefer session.fairness.evaluate(...); document limitations; "
            "use suggest_thresholds / suggest_reweighing only as disclosed opt-in tools.",
        ),
        concepts=("evaluation-partitions", "leakage-boundary", "diagnostic-uncertainty"),
        plain="Check whether the model treats groups differently on a holdout set.",
        when_to_use=(
            "When a sensitive attribute is available and stakeholders need disparity numbers.",
        ),
        when_not_to_use=(
            "When you need causal fairness, silent mitigation, or a legal determination.",
        ),
        mini_example=(
            "session.fairness.evaluate(sensitive_column='group', bootstrap_samples=100)",
            "# intersectional: session.fairness.evaluate(sensitive_column=['group', 'region'])",
            "# flat alias (deprecated): session.evaluate_fairness(sensitive_column='group')",
        ),
    ),
    _operation(
        "attach_fairness_to_last_eval",
        OperationKind.DIAGNOSTIC,
        "Run fairness on the partition used by the latest classical evaluate.",
        "Bridge classical evaluate → fairness without shrinking Session evaluate APIs.",
        "Convenience after session.evaluate(...); same report contract as evaluate_fairness.",
        (
            "Resolve the latest evaluate partition from Session history (or override).",
            "Delegate to evaluate_fairness with the same sensitive column options.",
        ),
        parameters=(
            _p(
                "sensitive_column",
                "str | Sequence[str]",
                "Caller-declared sensitive column(s).",
                required=True,
            ),
            _p(
                "positive_label",
                "Any",
                "Positive class label.",
                1,
            ),
            _p(
                "partition",
                "str | None",
                "Optional override; default uses last evaluate partition or test.",
                None,
            ),
            _p(
                "bootstrap_samples",
                "int",
                "Resample draws for stability bands; 0 disables.",
                0,
            ),
        ),
        inputs=("Fitted Session with a recent evaluate (recommended) and sensitive columns.",),
        outputs=("FairnessReport stored on session.fairness.last_report.",),
        prerequisites=(DATASET, SPLIT, FIT),
        ordering=("After session.evaluate(...).",),
        alternatives=("session.fairness.evaluate(partition=...)",),
        rationale=(
            "Keep classical evaluate unchanged while offering an explicit fairness attach path.",
        ),
        assumptions=("Last evaluate partition is the intended audit slice.",),
        failures=("Missing fit/split/sensitive columns.",),
        leakage=("Same as evaluate_fairness.",),
        anti_patterns=("Assuming attach rewrites classical metrics.",),
        state_changes=("Stores fairness report only.",),
        result_reading=("Same FairnessReport fields as evaluate_fairness.",),
        next_steps=("session.fairness.last_report.to_markdown()",),
        concepts=("evaluation-partitions",),
        plain="Attach a fairness audit to the partition you just evaluated.",
        when_to_use=("Right after classical evaluate when sensitive columns are available.",),
        when_not_to_use=("When you need a different partition than last evaluate.",),
        mini_example=(
            "session.evaluate()",
            "session.fairness.attach_to_last_eval(sensitive_column='group')",
        ),
    ),
    _operation(
        "suggest_fairness_thresholds",
        OperationKind.DIAGNOSTIC,
        "Suggest per-group score thresholds for opt-in post-hoc equalization.",
        "Return thresholds that target demographic parity or equal opportunity — not auto-applied.",
        "Optional mitigation helper; not legal certification.",
        (
            "Require predict_proba scores on the chosen partition (default validation).",
            "Search a threshold grid per group toward a global reference rate/TPR.",
            "Return GroupThresholdSuggestion with disclosures; do not rewrite predictions.",
        ),
        parameters=(
            _p(
                "sensitive_column",
                "str | Sequence[str]",
                "Caller-declared sensitive column(s).",
                required=True,
            ),
            _p(
                "partition",
                "train | validation | test",
                "Rows used to suggest thresholds (prefer validation).",
                "validation",
            ),
            _p(
                "target",
                "demographic_parity | equal_opportunity",
                "Equalization objective.",
                "demographic_parity",
            ),
            _p("positive_label", "Any", "Positive class label.", 1),
        ),
        inputs=("Fitted classifier with probabilities and sensitive columns.",),
        outputs=("GroupThresholdSuggestion (thresholds + disclosures).",),
        prerequisites=(DATASET, SPLIT, FIT),
        ordering=("After fit; prefer validation before reporting on test.",),
        alternatives=("External decision-policy tooling; session.decision.fit for cost thresholds.",),
        rationale=("Make post-hoc equalization explicit and refuse silent washing.",),
        assumptions=("Scores are comparable across groups.",),
        failures=("No probabilities; missing columns; empty partition.",),
        leakage=("Do not select thresholds on the test partition you will headline.",),
        anti_patterns=(
            "Advertising threshold equalization as legal compliance.",
            "Silently applying thresholds without disclosure.",
        ),
        state_changes=("Stores last mitigation suggestion; does not change the model.",),
        result_reading=("Read thresholds_by_group and achieved rates; keep disclosures.",),
        next_steps=("Apply thresholds explicitly in your decision layer if stakeholders agree.",),
        concepts=("leakage-boundary", "diagnostic-uncertainty"),
        plain="Propose different cut-offs per group — you still choose whether to use them.",
        when_to_use=("Exploring post-hoc rate equalization with full disclosure.",),
        when_not_to_use=("When you need certification or automatic mitigation products.",),
        mini_example=(
            "session.fairness.suggest_thresholds(sensitive_column='group', partition='validation')",
        ),
    ),
    _operation(
        "suggest_fairness_reweighing",
        OperationKind.DIAGNOSTIC,
        "Suggest Kamiran–Calders sample weights for opt-in train rebalancing.",
        "Return per-row weights — BuildML does not auto-fit with them.",
        "Optional mitigation helper; not legal certification.",
        (
            "Compute (group, label) weights on the chosen partition (default train).",
            "Return ReweighingSuggestion for the caller to pass as sample_weight later.",
        ),
        parameters=(
            _p(
                "sensitive_column",
                "str | Sequence[str]",
                "Caller-declared sensitive column(s).",
                required=True,
            ),
            _p(
                "partition",
                "train | validation | test",
                "Rows used to compute weights (prefer train).",
                "train",
            ),
            _p("positive_label", "Any", "Recorded for disclosure / table labeling.", 1),
        ),
        inputs=("Labeled partition with sensitive columns.",),
        outputs=("ReweighingSuggestion (weights + weight_table + disclosures).",),
        prerequisites=(DATASET, SPLIT),
        ordering=("Before a disclosed re-fit that accepts sample_weight.",),
        alternatives=("External rebalancing pipelines; collect better data.",),
        rationale=("Surface reweighing as an explicit choice, not silent washing.",),
        assumptions=("Train labels and groups are correctly declared.",),
        failures=("Missing columns; empty partition.",),
        leakage=("Do not claim unweighted holdout metrics after a weighted re-fit without disclosure.",),
        anti_patterns=(
            "Calling reweighing a fairness certificate.",
            "Auto-applying weights without stakeholder consent.",
        ),
        state_changes=("Stores last mitigation suggestion; does not change the model.",),
        result_reading=("Inspect weight_table and disclosures before any re-fit.",),
        next_steps=("Pass weights into a future fit only under a declared protocol.",),
        concepts=("leakage-boundary",),
        plain="Compute sample weights that balance group–label mass — you decide whether to use them.",
        when_to_use=("Exploring train-time rebalancing with honest documentation.",),
        when_not_to_use=("When you need automatic mitigation or legal certification.",),
        mini_example=(
            "session.fairness.suggest_reweighing(sensitive_column='group', partition='train')",
        ),
    ),
)
