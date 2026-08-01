# ruff: noqa: E501
"""Evidence-linked editorial records for model diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from buildml.explain.schemas import (
    Action,
    ActionPriority,
    Evidence,
    EvidenceKind,
    Finding,
    FindingSeverity,
    Recommendation,
)


def diagnostic_finding(
    key: str,
    title: str,
    detail: str,
    *,
    evidence: Sequence[Evidence],
    severity: FindingSeverity = FindingSeverity.INFO,
) -> Finding:
    """Create a finding whose interpretation is backed by explicit evidence."""
    return Finding(
        key=key,
        title=title,
        detail=detail,
        severity=severity,
        evidence=tuple(evidence),
    )


def metric_evidence(
    key: str,
    summary: str,
    value: Any,
    *,
    source: str,
    limitations: Sequence[str] = (),
) -> Evidence:
    """Create JSON-safe metric evidence."""
    return Evidence(
        key=key,
        kind=EvidenceKind.METRIC,
        summary=summary,
        value=_json_value(value),
        source=source,
        limitations=tuple(limitations),
    )


def observation_evidence(
    key: str,
    summary: str,
    value: Any,
    *,
    source: str,
    limitations: Sequence[str] = (),
) -> Evidence:
    """Create JSON-safe observational evidence."""
    return Evidence(
        key=key,
        kind=EvidenceKind.OBSERVATION,
        summary=summary,
        value=_json_value(value),
        source=source,
        limitations=tuple(limitations),
    )


def diagnostic_recommendation(
    key: str,
    title: str,
    rationale: str,
    *,
    finding_keys: Sequence[str],
    operation: str,
    parameters: Mapping[str, Any] | None = None,
    priority: ActionPriority = ActionPriority.NEXT,
    caveats: Sequence[str] = (),
) -> Recommendation:
    """Create advice linked to both observed findings and a public API action."""
    if not finding_keys:
        raise ValueError("Diagnostic recommendations require observed finding keys")
    return Recommendation(
        key=key,
        title=title,
        rationale=rationale,
        priority=priority,
        action=Action(
            key=f"{key}-action",
            label=_action_label(operation, parameters or {}),
            operation=operation,
            parameters={str(k): _json_value(v) for k, v in (parameters or {}).items()},
        ),
        based_on=tuple(finding_keys),
        caveats=tuple(caveats),
    )


def compatibility_recommendations(
    recommendations: Sequence[Recommendation],
) -> list[str]:
    """Retain the original list-of-strings view for public compatibility."""
    return [
        (
            f"{item.title}: {item.rationale} "
            f"[evidence: {', '.join(item.based_on)}; "
            f"action: {item.action.label if item.action else 'none'}]"
        )
        for item in recommendations
    ]


def evidence_for_diagnostic(
    kind: str,
    payload: Mapping[str, Any],
) -> tuple[list[Finding], list[Recommendation], list[str], list[str]]:
    """Translate a diagnostic payload into auditable editorial records."""
    source = f"DiagnosticReport.payload ({kind})"
    partition = str(payload.get("partition", "training cross-validation"))
    limits = [f"Results describe {partition}; they may not transfer to another population."]
    findings: list[Finding] = []
    recommendations: list[Recommendation] = []

    if kind == "calibration":
        if "ece" in payload:
            ece = float(payload["ece"])
            findings.append(
                diagnostic_finding(
                    "calibration-reliability",
                    "Probability reliability was measured",
                    f"ECE was {ece:.4f}; reliability is {'materially uneven' if ece > 0.1 else 'within the report review threshold'}.",
                    severity=FindingSeverity.MEDIUM if ece > 0.1 else FindingSeverity.INFO,
                    evidence=(
                        metric_evidence("ece", "Expected calibration error", ece, source=source, limitations=("Bin-based ECE depends on binning and sample size.",)),
                        metric_evidence("brier-score", "Brier score", payload.get("brier_score"), source=source, limitations=("Brier score combines calibration and discrimination.",)),
                    ),
                )
            )
            recommendations.append(
                diagnostic_recommendation(
                    "review-calibration",
                    "Review probability calibration before probability-based decisions",
                    f"The observed ECE is {ece:.4f}; use validation data for any calibrator and retain test data for confirmation.",
                    finding_keys=("calibration-reliability",),
                    operation="calibration",
                    parameters={"partition": "validation"},
                    caveats=("CalibratedClassifierCV fits a new calibration layer; do not fit it on the reported holdout.",),
                )
            )
        else:
            rows = list(payload.get("per_class_brier") or [])
            findings.append(
                diagnostic_finding(
                    "multiclass-calibration",
                    "Multiclass calibration is limited to one-vs-rest scores",
                    "A binary reliability curve is not valid for this output shape; per-class Brier scores are retained instead.",
                    severity=FindingSeverity.LOW,
                    evidence=(metric_evidence("per-class-brier", "Per-class Brier scores", rows, source=source, limitations=("Scores do not show reliability-curve shape.",)),),
                )
            )
            recommendations.append(
                diagnostic_recommendation(
                    "inspect-worst-class",
                    "Inspect the weakest class separately",
                    "The observed per-class scores can hide class-specific reliability problems when averaged.",
                    finding_keys=("multiclass-calibration",),
                    operation="calibration",
                    parameters={"partition": partition},
                    caveats=("BuildML does not yet render multiclass reliability curves.",),
                )
            )
            limits.append("Binary calibration curves and a single threshold policy are not applicable to multiclass output.")
    elif kind == "threshold_sweep":
        best = dict(payload.get("best_f1_threshold") or {})
        recommended = dict(payload.get("recommended_threshold") or best)
        cost_model = payload.get("cost_model")
        basis = str(payload.get("recommendation_basis") or "best_f1")
        findings.append(
            diagnostic_finding(
                "threshold-tradeoff",
                "Candidate operating points were compared",
                (
                    f"Recommended threshold={_fmt(recommended.get('threshold'))} "
                    f"(basis={basis}) on partition={partition}; "
                    f"highest observed F1 was at {_fmt(best.get('threshold'))}."
                ),
                evidence=(
                    metric_evidence(
                        "recommended-threshold",
                        "Recommended operating point",
                        recommended,
                        source=source,
                    ),
                    metric_evidence("best-f1-point", "Highest observed F1 row", best, source=source),
                    metric_evidence(
                        "ranking-quality",
                        "ROC-AUC and average precision",
                        {
                            "roc_auc": payload.get("roc_auc"),
                            "average_precision": payload.get("average_precision"),
                        },
                        source=source,
                    ),
                ),
            )
        )
        if cost_model:
            findings.append(
                diagnostic_finding(
                    "threshold-expected-cost",
                    "Expected cost was minimized under supplied FP/FN costs",
                    (
                        f"Minimum observed expected cost total="
                        f"{_fmt(payload.get('expected_cost_at_recommended'))} "
                        f"at threshold {_fmt(recommended.get('threshold'))}."
                    ),
                    evidence=(
                        metric_evidence(
                            "cost-model",
                            "Cost/benefit coefficients",
                            dict(cost_model),
                            source=source,
                            limitations=(
                                "Costs are caller-supplied and assumed constant across the partition.",
                                "Expected cost on this partition is not a deployment guarantee.",
                            ),
                        ),
                    ),
                )
            )
            limits.append(
                "Cost-optimal thresholds inherit validation prevalence and the supplied cost ratios."
            )
        recommendations.append(
            diagnostic_recommendation(
                "validate-threshold-policy",
                "Validate a threshold against explicit error costs",
                (
                    "Use fp_cost/fn_cost on validation to select a policy, then confirm the fixed "
                    "cutoff once on untouched test."
                ),
                finding_keys=(
                    ("threshold-tradeoff", "threshold-expected-cost")
                    if cost_model
                    else ("threshold-tradeoff",)
                ),
                operation="tune_threshold",
                parameters={
                    "partition": "validation",
                    "fp_cost": (cost_model or {}).get("fp_cost"),
                    "fn_cost": (cost_model or {}).get("fn_cost"),
                },
                caveats=("Selecting on test biases the final estimate.",),
            )
        )
    elif kind == "learning_curve":
        gap = float(payload.get("final_gap", 0.0))
        gain = float(payload.get("valid_score_gain", 0.0))
        findings.append(
            diagnostic_finding(
                "learning-curve-shape",
                "Sample-size behavior was estimated",
                f"The final train-validation gap was {gap:.4f} and validation gain was {gain:.4f}.",
                severity=FindingSeverity.MEDIUM if gap > 0.1 else FindingSeverity.INFO,
                evidence=(metric_evidence("learning-curve-summary", "Final gap and validation gain", {"final_gap": gap, "valid_score_gain": gain}, source=source, limitations=("Ordinary CV may be invalid for grouped or temporal observations.",)),),
            )
        )
        recommendations.append(
            diagnostic_recommendation(
                "respond-to-learning-curve",
                "Choose the next experiment from the observed curve",
                "Use the measured gap and gain to compare regularization, feature changes, or additional labels.",
                finding_keys=("learning-curve-shape",),
                operation="learning_curve",
                parameters={"cv": payload.get("cv_folds")},
            )
        )
    elif kind == "permutation_importance":
        rows = list(payload.get("rows") or [])
        top = rows[0] if rows else {}
        findings.append(
            diagnostic_finding(
                "feature-reliance",
                "Holdout feature reliance was measured",
                f"The leading observed feature was {top.get('feature', 'not available')!r}; repeat spread is retained with every value.",
                evidence=(metric_evidence("importance-rows", "Permutation importance rows", rows, source=source, limitations=("Correlated substitutes can divide importance; importance is not causality.",)),),
            )
        )
        recommendations.append(
            diagnostic_recommendation(
                "validate-feature-reliance",
                "Audit leading reliance across relevant splits",
                "The observed ranking is specific to this fitted model, score, and partition.",
                finding_keys=("feature-reliance",),
                operation="feature_importance",
                parameters={"partition": partition, "n_repeats": payload.get("n_repeats")},
                caveats=("Do not automatically remove low-ranked correlated features.",),
            )
        )
    elif kind == "segment_errors":
        segments = list(payload.get("segments") or [])
        small = list(payload.get("small_segments") or [])
        worst = segments[0] if segments else (small[0] if small else {})
        by_columns = payload.get("by_columns") or payload.get("by")
        findings.append(
            diagnostic_finding(
                "segment-error-gaps",
                "Prediction errors were sliced by segment",
                (
                    f"Errors on {partition} were grouped by {by_columns!r}; "
                    f"the highest-error primary segment was {worst.get('segment', 'not available')!r} "
                    f"(n={worst.get('n', 0)}; primary={len(segments)}, small_n={len(small)})."
                ),
                severity=FindingSeverity.MEDIUM if segments else FindingSeverity.INFO,
                evidence=(
                    metric_evidence(
                        "segment-error-table",
                        "Primary per-segment error aggregates",
                        segments,
                        source=source,
                        limitations=(
                            "Only the most frequent segments are shown; small-n rates are unstable.",
                            "Segment gaps are observational, not fairness or causal proof.",
                        ),
                    ),
                    metric_evidence(
                        "small-segment-table",
                        "Segments below min_segment_n",
                        small,
                        source=source,
                        limitations=("Rates with tiny support are unstable review hints only.",),
                    ),
                ),
            )
        )
        recommendations.append(
            diagnostic_recommendation(
                "review-worst-segment",
                "Inspect the highest-error segment before changing the model",
                "Confirm sample size and label quality in the worst segment before treating the gap as a modeling failure.",
                finding_keys=("segment-error-gaps",),
                operation="error_slices",
                parameters={
                    "by": payload.get("by"),
                    "partition": partition,
                    "min_segment_n": payload.get("min_segment_n"),
                },
                caveats=("Do not tune on test solely to close a segment gap.",),
            )
        )
        limits.append("Segment analysis is not a fairness audit and does not adjust for confounders.")
        if not segments and small:
            limits.append(
                "No segment met min_segment_n; only small_segments were available for review."
            )

    methods = _methods(kind, payload)
    return findings, recommendations, limits, methods


def evidence_for_plot_board(
    task: str,
    partition: str,
    metrics: Mapping[str, Any],
    skipped: Sequence[Mapping[str, str]],
) -> tuple[list[Finding], list[Recommendation], list[str]]:
    """Create evidence-linked findings for a classification/regression board."""
    source = f"PlotBoardReport.metrics ({partition})"
    findings: list[Finding] = []
    recommendations: list[Recommendation] = []
    if task == "classification":
        findings.append(
            diagnostic_finding(
                "classification-errors",
                "Classification error structure was measured",
                f"The confusion matrix contains {metrics.get('n_labels', 'unknown')} labels and observed accuracy {metrics.get('accuracy_proxy', 'not available')}.",
                evidence=(metric_evidence("confusion-matrix-summary", "Confusion matrix summary", {"accuracy": metrics.get("accuracy_proxy"), "false_positives": metrics.get("false_positives"), "false_negatives": metrics.get("false_negatives")}, source=source),),
            )
        )
        if "roc_auc" in metrics:
            findings.append(
                diagnostic_finding(
                    "classification-ranking",
                    "Binary probability ranking was measured",
                    f"ROC-AUC={_fmt(metrics.get('roc_auc'))}; average precision={_fmt(metrics.get('average_precision'))}; ECE={_fmt(metrics.get('ece'))}.",
                    evidence=(metric_evidence("probability-metrics", "ROC-AUC, average precision, and ECE", {"roc_auc": metrics.get("roc_auc"), "average_precision": metrics.get("average_precision"), "ece": metrics.get("ece")}, source=source, limitations=("Binary-only probability panels do not describe multiclass behavior.",)),),
                )
            )
            recommendations.append(
                diagnostic_recommendation(
                    "review-decision-policy",
                    "Review calibration and the decision threshold together",
                    "The observed ranking, reliability, and threshold point answer different questions and should be reviewed jointly.",
                    finding_keys=("classification-ranking", "classification-errors"),
                    operation="tune_threshold",
                    parameters={"partition": "validation"},
                    caveats=("Use explicit error costs; peak F1 is only one candidate.",),
                )
            )
        else:
            findings.append(
                diagnostic_finding(
                    "probability-panels-unavailable",
                    "Probability-dependent evidence is unavailable",
                    "ROC, precision-recall, calibration, and threshold panels were not silently omitted; their skip reasons are retained.",
                    severity=FindingSeverity.LOW,
                    evidence=(observation_evidence("probability-panel-skips", "Probability panel skip reasons", [dict(item) for item in skipped if item.get("panel") in {"roc_curve", "pr_curve", "calibration", "threshold_tradeoff"}], source="PlotBoardReport.skipped"),),
                )
            )
            recommendations.append(
                diagnostic_recommendation(
                    "choose-probability-path",
                    "Choose a probability-capable evaluation path only if decisions require probabilities",
                    "The observed estimator capability or target shape does not support the binary probability panels.",
                    finding_keys=("probability-panels-unavailable",),
                    operation="eval_plots",
                    parameters={"partition": partition},
                    caveats=("Multiclass work needs class-specific methods; a binary threshold is not interchangeable.",),
                )
            )
    else:
        findings.append(
            diagnostic_finding(
                "regression-errors",
                "Regression residual behavior was measured",
                f"Observed RMSE={_fmt(metrics.get('rmse'))}, mean residual={_fmt(metrics.get('residual_bias'))}, and absolute-residual correlation={_fmt(metrics.get('heteroscedasticity_correlation'))}.",
                evidence=(metric_evidence("residual-summary", "Residual summary", dict(metrics), source=source, limitations=("Residual patterns on one partition may not persist after distribution shift.",)),),
            )
        )
        recommendations.append(
            diagnostic_recommendation(
                "review-regression-errors",
                "Inspect residual structure before changing the model",
                "The observed bias, spread, and variance pattern identify which regression assumptions need focused review.",
                finding_keys=("regression-errors",),
                operation="eval_plots",
                parameters={"partition": partition, "include_learning_curve": True},
            )
        )
    limitations = [
        f"Visual and numeric evidence describes partition={partition}.",
        "Plots are diagnostic estimates, not causal or deployment-validity claims.",
    ]
    return findings, recommendations, limitations


def _action_label(operation: str, parameters: Mapping[str, Any]) -> str:
    args = ", ".join(f"{key}={value!r}" for key, value in parameters.items())
    return f"Session.{operation}({args})"


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    return str(value)


def _methods(kind: str, payload: Mapping[str, Any]) -> list[str]:
    if kind == "calibration":
        return ["Quantile reliability bins; Brier score and bin-mean absolute calibration gap.", "Multiclass output uses one-vs-rest Brier scores."]
    if kind == "threshold_sweep":
        methods = [
            "Thresholds from 0.05 to 0.95; precision, recall, F1, confusion counts, ROC-AUC, and average precision."
        ]
        if payload.get("cost_model"):
            methods.append(
                "Expected cost = fp_cost*FP + fn_cost*FN - tp_benefit*TP - tn_benefit*TN on the scored partition."
            )
        return methods
    if kind == "learning_curve":
        return [f"scikit-learn learning_curve with {payload.get('cv_folds')} folds and scoring={payload.get('scoring')}."]
    if kind == "permutation_importance":
        return [f"Permutation importance with {payload.get('n_repeats')} repeats and scoring={payload.get('scoring')}."]
    if kind == "segment_errors":
        by_columns = payload.get("by_columns") or payload.get("by")
        return [
            (
                f"Group predictions by {by_columns!r} on partition={payload.get('partition')}; "
                "aggregate classification or regression error metrics for the most frequent segments; "
                f"exclude n < {payload.get('min_segment_n', 5)} from primary ranking."
            )
        ]
    return ["Task-appropriate diagnostic calculation."]


def _fmt(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)
