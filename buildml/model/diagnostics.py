"""Ask the questions a single score cannot answer.

Accuracy tells you how often the model is right. It does not tell you whether
its confidence means anything, where to set the decision boundary, whether more
data would help, which features are carrying the result, or which slice of your
users it quietly fails. Each of those is a different question, and each has its
own diagnostic here.

:func:`calibration_report` asks whether predicted probabilities are honest :
whether among the cases the model calls 70% likely, roughly 70% actually happen.
A model can rank perfectly and still be badly calibrated, which matters the
moment a probability feeds a decision rather than a sort order.

:func:`threshold_report` asks where to draw the line. The default 0.5 is a
convention with no claim to being right; the correct threshold depends on what a
false positive costs relative to a false negative, and this shows the trade-off
across the range.

:func:`learning_curve_report` asks whether the constraint is data or model.
Training and validation scores converging at a disappointing level means the
model is too simple; a persistent gap means it is overfitting and more data
would help.

:func:`permutation_importance_report` asks which features the model is actually
using, by measuring how much performance drops when each is shuffled.

:func:`segment_error_report` asks who the model fails. Aggregate metrics average
over everyone, and an overall accuracy of 0.90 is compatible with 0.95 for most
users and 0.55 for a subgroup: the kind of failure that only appears when you
look for it.

Every report returns findings and recommendations linked to the numbers behind
them, so the interpretation can be checked rather than trusted.

See Also
--------
buildml.model.supervised.evaluate_estimator : The summary these go beyond.
buildml.model.plot_boards : Visual counterparts.
buildml.model.evidence : How findings stay linked to measurements.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import learning_curve

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition
from buildml.explain.schemas import Finding, Recommendation
from buildml.model.evidence import (
    compatibility_recommendations,
    evidence_for_diagnostic,
)
from buildml.model.supervised import FitResult, _feature_target_frames


@dataclass(slots=True)
class DiagnosticReport:
    """A diagnostic's numbers, what they appear to mean, and what they cannot say.

    Every diagnostic returns this shape, so a report can be read, serialised, or
    exported without knowing which kind it is. The numbers live in ``payload``;
    the interpretation is derived from them automatically, which is what keeps
    the advice tied to the measurement rather than floating free of it.

    Attributes
    ----------
    kind:
        Which diagnostic this is: ``'calibration'``, ``'threshold'``,
        ``'learning_curve'``, ``'permutation_importance'``, or
        ``'segment_error'``.
    payload:
        The computed values: curve points, per-threshold metrics, importance
        scores, per-segment errors. The primary content.
    recommendations:
        Advice as plain strings, for display.
    interpretation:
        What the numbers appear to say.
    figure_dir, html_path, figure_paths:
        Where any exported figures and HTML landed.
    findings:
        Structured observations, each carrying its supporting evidence.
    recommendation_details:
        Structured advice, each naming the findings behind it and an operation
        to run.
    limitations:
        What this report cannot support: always including the partition it
        describes.
    methods:
        How the numbers were computed.

    Notes
    -----
    **Read ``limitations`` before acting on ``recommendations``.** Every
    diagnostic describes one partition, and a threshold tuned on validation is
    not automatically right for production traffic.

    **``findings`` and ``recommendations`` are two views of the same thing.** The
    strings are for reading; the structured records keep the evidence links, and
    are what a report or an agent should consume.

    See Also
    --------
    buildml.model.evidence.evidence_for_diagnostic : The interpretation layer.
    """

    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    interpretation: list[str] = field(default_factory=list)
    figure_dir: str | None = None
    html_path: str | None = None
    figure_paths: dict[str, str] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    recommendation_details: list[Recommendation] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.findings or self.recommendation_details:
            return
        findings, recommendations, limitations, methods = evidence_for_diagnostic(
            self.kind, self.payload
        )
        self.findings = findings
        self.recommendation_details = recommendations
        self.recommendations = compatibility_recommendations(recommendations)
        self.limitations = limitations
        self.methods = methods

    def to_dict(self) -> dict[str, Any]:
        """Convert the report to plain data for history, HTML, and logs.

        Findings and structured recommendations are converted too, so the
        evidence links survive into whatever consumes this.

        Returns
        -------
        dict
            Every field, with the structured records converted to dictionaries.

        Notes
        -----
        **``payload`` can be large.** A threshold report carries a row per
        candidate threshold and a segment report a row per segment; trim before
        writing this into a compact log.
        """
        return {
            "kind": self.kind,
            "payload": self.payload,
            "recommendations": list(self.recommendations),
            "interpretation": list(self.interpretation),
            "figure_dir": self.figure_dir,
            "html_path": self.html_path,
            "figure_paths": dict(self.figure_paths),
            "findings": [item.to_dict() for item in self.findings],
            "recommendation_details": [
                item.to_dict() for item in self.recommendation_details
            ],
            "limitations": list(self.limitations),
            "methods": list(self.methods),
        }

    def show(self) -> None:
        """Print the interpretation, the recommendations, and any export paths.

        For reading a report at a prompt. Interpretation lines are marked ``*``
        and recommendations ``-``.

        Notes
        -----
        **Limitations are not printed.** They are the part most worth reading
        before acting, so consult ``limitations`` directly rather than treating
        this digest as the whole report.
        """
        print(f"Diagnostic · {self.kind}")
        for tip in self.interpretation[:8]:
            print(f"* {tip}")
        for tip in self.recommendations[:8]:
            print(f"- {tip}")
        if self.figure_dir:
            print(f"Figures: {self.figure_dir}")
        if self.html_path:
            print(f"HTML: {self.html_path}")

    def export_html(self, path: str | Path) -> Path:
        """Write the report as a self-contained HTML dashboard.

        For sharing a diagnostic with someone who will not be running Python.
        The findings, evidence, recommendations, and limitations all carry over,
        so the reader gets the caveats along with the numbers.

        Parameters
        ----------
        path:
            Where to write. Parent directories are created as needed.

        Returns
        -------
        pathlib.Path
            The file written. Also recorded on ``html_path``.

        Notes
        -----
        **Any figures are referenced, not embedded.** Move the HTML without its
        figure directory and the images break.

        See Also
        --------
        buildml.model.html_diagnostics.export_diagnostics_html : The renderer.
        """
        from buildml.model.html_diagnostics import export_diagnostics_html

        destination = export_diagnostics_html(self.to_dict(), path)
        self.html_path = str(destination)
        return destination


def calibration_report(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    fit_result: FitResult,
    *,
    partition: Literal["train", "validation", "test"] = "test",
    n_bins: int = 10,
    export_figures: str | Path | None = None,
    export_html: str | Path | None = None,
) -> DiagnosticReport:
    """Check whether the model's stated confidence is worth anything.

    A well-calibrated model that says 70% is right about 70% of the time. Many
    models are not: tree ensembles push probabilities toward the extremes, and a
    model that ranks cases perfectly can still report confidences that are
    systematically too high or too low.

    Whether that matters depends on what you do with the number. If the
    probability only sorts a queue, calibration is irrelevant. If it feeds an
    expected-value calculation, a threshold with real costs, or a figure shown to
    a human, an uncalibrated probability is actively misleading.

    Predictions are bucketed by confidence and each bucket's predicted rate is
    compared with its observed rate. Perfect calibration is the diagonal;
    departures from it are the finding.

    Parameters
    ----------
    dataset:
        The data.
    split_plan:
        Partition membership.
    fit_result:
        A fitted classifier that provides probabilities.
    partition:
        Which partition to assess. Defaults to test.
    n_bins:
        Confidence buckets. More bins resolve the curve finely and put fewer
        rows in each, making every point noisier.
    export_figures:
        Directory for a reliability diagram, if wanted.
    export_html:
        Path for an HTML report, if wanted.

    Returns
    -------
    DiagnosticReport
        Curve points, Brier score, and an ECE estimate for binary problems;
        per-class Brier scores for multiclass.

    Raises
    ------
    ValidationError
        If the model is not a classifier, offers no ``predict_proba``, or if the
        split is missing.

    Notes
    -----
    **Calibration and discrimination are independent.** A model can rank cases
    perfectly and be badly calibrated, or be beautifully calibrated and rank
    poorly. AUC measures one; this measures the other.

    **ECE depends on the binning.** The number is not comparable across
    different ``n_bins`` values, so hold it fixed when tracking calibration over
    time. Above roughly 0.1 is worth investigating.

    **Assess on held-out data.** Calibration on training rows is meaningless :
    the model has seen the answers.

    **The multiclass path is per-class Brier only.** A single reliability
    diagram does not generalise beyond binary, so the report gives one score per
    class and points at the worst.

    **Fixing calibration does not require a new model.** Wrapping the existing
    one in Platt scaling or isotonic regression, fitted on held-out data, is
    usually enough.

    See Also
    --------
    threshold_report : Choosing a decision boundary from these probabilities.
    """
    if fit_result.task != "classification":
        raise ValidationError("Calibration report requires a classification model")
    if not hasattr(fit_result.estimator, "predict_proba"):
        raise ValidationError(
            "Estimator does not support predict_proba: "
            "use a probabilistic classifier or CalibratedClassifierCV"
        )
    if split_plan is None:
        raise ValidationError("Split required")

    x, y, _, _, _ = _feature_target_frames(dataset, split_plan, partition)
    x = x[list(fit_result.feature_columns)]
    proba = fit_result.estimator.predict_proba(x)
    tips: list[str] = []
    interpretation: list[str] = []
    payload: dict[str, Any] = {
        "partition": partition,
        "n_rows": int(len(y)),
        "n_classes": int(proba.shape[1]),
    }

    if proba.shape[1] == 2:
        y_bin = pd.Series(y).astype(str)
        positive = str(fit_result.estimator.classes_[1])
        y_true = (y_bin == positive).astype(int).to_numpy()
        prob_pos = proba[:, 1]
        n_bins_eff = min(n_bins, max(3, len(y_true) // 5))
        frac_pos, mean_pred = calibration_curve(
            y_true, prob_pos, n_bins=n_bins_eff, strategy="quantile"
        )
        brier = float(brier_score_loss(y_true, prob_pos))
        ece = float(np.mean(np.abs(np.asarray(frac_pos) - np.asarray(mean_pred))))
        max_gap = float(np.max(np.abs(np.asarray(frac_pos) - np.asarray(mean_pred))))
        payload["positive_class"] = positive
        payload["brier_score"] = brier
        payload["ece"] = ece
        payload["max_reliability_gap"] = max_gap
        payload["n_bins"] = int(n_bins_eff)
        payload["prevalence"] = float(y_true.mean())
        payload["calibration_curve"] = {
            "fraction_positives": [float(v) for v in frac_pos],
            "mean_predicted_value": [float(v) for v in mean_pred],
        }
        interpretation.append(
            f"Brier={brier:.4f} (0=perfect probabilistic accuracy); "
            f"ECE≈{ece:.4f}; max bin gap={max_gap:.4f}; "
            f"prevalence={payload['prevalence']:.3f}."
        )
        if brier > 0.25:
            tips.append(
                f"Brier score {brier:.3f} is high: probabilities are poorly "
                "calibrated/discriminative; consider better features or calibration."
            )
        elif ece > 0.1:
            tips.append(
                f"ECE≈{ece:.3f} indicates material miscalibration: "
                "wrap with CalibratedClassifierCV on train folds."
            )
        else:
            tips.append(
                f"Calibration looks usable (ECE≈{ece:.3f}); still validate "
                "threshold policy against business costs."
            )
        if abs(float(np.mean(prob_pos)) - float(y_true.mean())) > 0.08:
            tips.append(
                "Mean predicted probability diverges from prevalence: "
                "check class priors / sample weights."
            )
    else:
        # Multiclass: one-vs-rest Brier-style summary per class (micro)
        y_str = pd.Series(y).astype(str)
        classes = [str(c) for c in fit_result.estimator.classes_]
        per_class = []
        for idx, cls in enumerate(classes):
            y_bin = (y_str == cls).astype(int).to_numpy()
            prob_c = proba[:, idx]
            try:
                brier_c = float(brier_score_loss(y_bin, prob_c))
            except ValueError:
                brier_c = float("nan")
            per_class.append({"class": cls, "brier_score": brier_c})
        payload["per_class_brier"] = per_class
        payload["note"] = (
            "Multiclass: binary reliability diagram omitted; per-class Brier provided."
        )
        interpretation.append(
            "Multiclass probabilities scored with per-class one-vs-rest Brier; "
            "inspect the worst class first."
        )
        worst = max(
            (r for r in per_class if r["brier_score"] == r["brier_score"]),
            key=lambda r: r["brier_score"],
            default=None,
        )
        if worst:
            tips.append(
                f"Highest per-class Brier: '{worst['class']}'={worst['brier_score']:.3f}: "
                "consider class-specific calibration or rebalancing."
            )
        tips.append(
            "For multiclass production, prefer per-class reliability plots "
            "or temperature scaling."
        )

    report = DiagnosticReport(
        kind="calibration",
        payload=payload,
        recommendations=tips,
        interpretation=interpretation,
    )
    return _maybe_export_diagnostic_board(
        report,
        dataset=dataset,
        split_plan=split_plan,
        fit_result=fit_result,
        partition=partition,
        export_figures=export_figures,
        export_html=export_html,
        board_panels=("calibration",),
    )


def threshold_report(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    fit_result: FitResult,
    *,
    partition: Literal["train", "validation", "test"] = "test",
    fp_cost: float | None = None,
    fn_cost: float | None = None,
    tp_benefit: float = 0.0,
    tn_benefit: float = 0.0,
    export_figures: str | Path | None = None,
    export_html: str | Path | None = None,
) -> DiagnosticReport:
    """Find where to draw the line between predicting yes and predicting no.

    A classifier outputs a probability; turning that into a decision requires a
    threshold, and 0.5 is a convention rather than an answer. The right value
    depends entirely on what the two mistakes cost. Missing a fraudulent
    transaction and wrongly blocking a legitimate one are not equally bad, and
    no default can know which way.

    Every candidate threshold is swept and its precision, recall, and F1
    recorded, so the trade-off is visible rather than assumed. Raising the
    threshold buys precision at the cost of recall; lowering it does the
    reverse. There is no setting that improves both.

    When ``fp_cost`` and ``fn_cost`` are supplied the guesswork disappears: the
    threshold that minimises expected cost is computed directly. This is the
    right way to use the function whenever the costs can be estimated at all,
    even roughly: a rough ratio beats an arbitrary 0.5.

    Parameters
    ----------
    dataset:
        The data.
    split_plan:
        Partition membership.
    fit_result:
        A fitted binary classifier that provides probabilities.
    partition:
        Which partition to sweep on. Use validation: a threshold tuned on test
        is a threshold fitted to test.
    fp_cost:
        Cost of one false positive. Must be given together with ``fn_cost``.
    fn_cost:
        Cost of one false negative.
    tp_benefit:
        Benefit of a true positive, subtracted from cost.
    tn_benefit:
        Benefit of a true negative, subtracted from cost.
    export_figures:
        Directory for ROC, precision-recall, and trade-off plots.
    export_html:
        Path for an HTML report.

    Returns
    -------
    DiagnosticReport
        Per-threshold metrics, ROC and precision-recall samples, named operating
        points (best F1, high recall, high precision), and: in cost mode: the
        minimum-expected-cost threshold with its total and mean cost.

    Raises
    ------
    ValidationError
        If the model is not a probabilistic classifier, if the split is missing,
        if only one of ``fp_cost`` and ``fn_cost`` is given, or if a cost or
        benefit is negative or not finite.

    Notes
    -----
    **Only the ratio of the costs matters, not their units.** Setting them to 1
    and 10 gives the same threshold as 100 and 1000. That makes the input far
    easier to supply than it first appears: you need the relative severity, not
    a currency figure.

    **Peak F1 is the most tempting operating point and often the wrong one.** It
    treats both errors as equally costly, which is exactly the assumption you
    came here to avoid.

    **Tune on validation, confirm on test.** The best threshold on a partition
    is partly fitted to that partition's noise, and the effect is larger than
    people expect on small partitions.

    **A threshold does not fix a poorly ranked model.** If AUC is near 0.5, no
    threshold helps; the sweep only redistributes the errors.

    **Prevalence drift invalidates the choice.** A threshold tuned when 2% of
    cases were positive is wrong once that becomes 10%. Re-tune when the base
    rate moves.

    Examples
    --------
    Tune with explicit costs on validation::

        report = threshold_report(
            dataset, split_plan, fit,
            partition="validation",
            fp_cost=1.0,
            fn_cost=20.0,
        )
        print(report.payload["recommended_threshold"])

    See Also
    --------
    calibration_report : Whether the probabilities behind this are honest.
    """
    if fit_result.task != "classification" or not hasattr(fit_result.estimator, "predict_proba"):
        raise ValidationError(
            "Threshold sweep requires a probabilistic binary classifier "
            "(task=classification with predict_proba)"
        )
    if split_plan is None:
        raise ValidationError("Split required")
    cost_mode = fp_cost is not None or fn_cost is not None
    if cost_mode and (fp_cost is None or fn_cost is None):
        raise ValidationError(
            "Cost-sensitive threshold tuning requires both fp_cost and fn_cost "
            "(optional tp_benefit / tn_benefit default to 0)."
        )
    if cost_mode:
        for name, value in (
            ("fp_cost", fp_cost),
            ("fn_cost", fn_cost),
            ("tp_benefit", tp_benefit),
            ("tn_benefit", tn_benefit),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValidationError(f"{name} must be a finite number")
            if not np.isfinite(float(value)):
                raise ValidationError(f"{name} must be a finite number")
            if name in {"fp_cost", "fn_cost"} and float(value) < 0:
                raise ValidationError(f"{name} must be >= 0")

    x, y, _, _, _ = _feature_target_frames(dataset, split_plan, partition)
    x = x[list(fit_result.feature_columns)]
    if len(fit_result.estimator.classes_) != 2:
        raise ValidationError(
            "Threshold sweep currently supports binary targets; "
            "for multiclass use per-class one-vs-rest thresholds"
        )

    y_bin = pd.Series(y).astype(str)
    positive = str(fit_result.estimator.classes_[1])
    y_true = (y_bin == positive).astype(int).to_numpy()
    prob = fit_result.estimator.predict_proba(x)[:, 1]
    precision, recall, thr = precision_recall_curve(y_true, prob)
    fpr, tpr, roc_thr = roc_curve(y_true, prob)
    roc_auc = float(roc_auc_score(y_true, prob))
    ap = float(average_precision_score(y_true, prob))
    n_rows = int(len(y_true))

    rows: list[dict[str, Any]] = []
    for t in np.linspace(0.05, 0.95, 19):
        pred = (prob >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        row: dict[str, Any] = {
            "threshold": float(t),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "predicted_positive_rate": float(pred.mean()),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }
        if cost_mode:
            assert fp_cost is not None and fn_cost is not None
            total_cost = (
                float(fp_cost) * fp
                + float(fn_cost) * fn
                - float(tp_benefit) * tp
                - float(tn_benefit) * tn
            )
            row["expected_cost_total"] = float(total_cost)
            row["expected_cost_mean"] = float(total_cost / n_rows) if n_rows else float("nan")
        rows.append(row)

    best_f1 = max(rows, key=lambda item: item["f1"])
    high_recall = max(rows, key=lambda item: (item["recall"], item["precision"]))
    constrained = [r for r in rows if r["precision"] >= 0.5]
    high_recall_prec = (
        max(constrained, key=lambda item: item["recall"]) if constrained else high_recall
    )
    high_precision = max(rows, key=lambda item: (item["precision"], item["recall"]))
    min_cost_point: dict[str, Any] | None = None
    if cost_mode:
        min_cost_point = min(
            rows,
            key=lambda item: (
                item["expected_cost_total"],
                -item["f1"],
                item["threshold"],
            ),
        )

    recommended = min_cost_point if min_cost_point is not None else best_f1
    recommendation_basis = "min_expected_cost" if min_cost_point is not None else "best_f1"
    operating_points = {
        "best_f1": best_f1,
        "high_recall": high_recall,
        "high_precision": high_precision,
        "precision_constrained_high_recall": high_recall_prec,
    }
    if min_cost_point is not None:
        operating_points["min_expected_cost"] = min_cost_point

    cost_model = None
    if cost_mode:
        assert fp_cost is not None and fn_cost is not None
        cost_model = {
            "fp_cost": float(fp_cost),
            "fn_cost": float(fn_cost),
            "tp_benefit": float(tp_benefit),
            "tn_benefit": float(tn_benefit),
            "formula": (
                "fp_cost*FP + fn_cost*FN - tp_benefit*TP - tn_benefit*TN "
                "(totals over the scored partition)"
            ),
        }

    interpretation = [
        f"ROC-AUC={roc_auc:.3f}; PR-AUC (AP)={ap:.3f}; prevalence={float(y_true.mean()):.3f}; "
        f"partition={partition}; n={n_rows}.",
        (
            f"Recommended threshold={recommended['threshold']:.2f} "
            f"(basis={recommendation_basis}): "
            f"P={recommended['precision']:.3f}, R={recommended['recall']:.3f}, "
            f"F1={recommended['f1']:.3f}."
        ),
        (
            f"Best F1 @ threshold={best_f1['threshold']:.2f}: "
            f"P={best_f1['precision']:.3f}, R={best_f1['recall']:.3f}, "
            f"F1={best_f1['f1']:.3f}."
        ),
    ]
    if min_cost_point is not None:
        interpretation.append(
            f"Minimum expected cost @ threshold={min_cost_point['threshold']:.2f}: "
            f"total={min_cost_point['expected_cost_total']:.4f}, "
            f"mean={min_cost_point['expected_cost_mean']:.6f}."
        )
    else:
        interpretation.append(
            f"Precision≥0.5 operating point: threshold={high_recall_prec['threshold']:.2f} "
            f"(P={high_recall_prec['precision']:.3f}, R={high_recall_prec['recall']:.3f})."
        )

    recommendations = [
        (
            "Select the decision threshold on validation (or an explicit policy set), "
            "then confirm the fixed cutoff once on untouched test."
        ),
        (
            f"If false negatives dominate cost, consider threshold≈"
            f"{high_recall['threshold']:.2f} (recall={high_recall['recall']:.3f})."
        ),
        (
            f"If false positives dominate cost, consider threshold≈"
            f"{high_precision['threshold']:.2f} "
            f"(precision={high_precision['precision']:.3f})."
        ),
    ]
    if cost_mode:
        recommendations.insert(
            0,
            (
                "Expected-cost minimization used the supplied FP/FN costs "
                f"(and benefits) on partition={partition}; re-run when prevalence or costs change."
            ),
        )
    else:
        recommendations.insert(
            0,
            "Pass fp_cost and fn_cost for explicit expected-cost minimization instead of F1 alone.",
        )

    report = DiagnosticReport(
        kind="threshold_sweep",
        payload={
            "partition": partition,
            "positive_class": positive,
            "n_rows": n_rows,
            "prevalence": float(y_true.mean()),
            "roc_auc": roc_auc,
            "average_precision": ap,
            "rows": rows,
            "operating_points": operating_points,
            "recommended_threshold": recommended,
            "recommendation_basis": recommendation_basis,
            "expected_cost_at_recommended": (
                recommended.get("expected_cost_total") if cost_mode else None
            ),
            "cost_model": cost_model,
            "best_f1_threshold": best_f1,
            "high_recall_threshold": high_recall,
            "high_precision_threshold": high_precision,
            "precision_constrained_high_recall": high_recall_prec,
            "min_expected_cost_threshold": min_cost_point,
            "pr_curve": {
                "precision": _downsample(precision),
                "recall": _downsample(recall),
                "n_thresholds": int(len(thr)),
            },
            "roc_curve": {
                "fpr": _downsample(fpr),
                "tpr": _downsample(tpr),
                "n_thresholds": int(len(roc_thr)),
            },
        },
        recommendations=recommendations,
        interpretation=interpretation,
    )
    return _maybe_export_diagnostic_board(
        report,
        dataset=dataset,
        split_plan=split_plan,
        fit_result=fit_result,
        partition=partition,
        export_figures=export_figures,
        export_html=export_html,
        board_panels=("threshold_tradeoff", "roc_curve", "pr_curve"),
    )


def learning_curve_report(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    estimator: Any,
    *,
    task: Literal["classification", "regression", "auto"] = "auto",
    cv: int = 5,
    export_figures: str | Path | None = None,
    export_html: str | Path | None = None,
) -> DiagnosticReport:
    """Find out whether more data would help, or whether the model is the limit.

    Answers a question worth asking before spending months collecting more data.
    The estimator is trained on progressively larger subsets and scored at each
    size, and the shape of the two resulting curves says which constraint you
    are actually up against.

    Both curves converging at a disappointing level is **underfitting**: the
    model is too simple to capture the pattern, and more rows of the same data
    will change nothing. Try a more expressive model or better features.

    A large persistent gap: training score high, validation score well below :
    is **overfitting**: the model is memorising. Here more data genuinely helps,
    and so does regularisation or a simpler model.

    A validation curve still climbing at the largest size means you have not yet
    saturated; more data is likely to pay. One that flattened long ago means it
    will not.

    Parameters
    ----------
    dataset:
        The data.
    split_plan:
        The split. Curves are computed within the train partition.
    estimator:
        The estimator to profile. Refitted at every size, so this is expensive.
    task:
        ``'classification'``, ``'regression'``, or ``'auto'``, choosing the
        scoring metric.
    cv:
        Folds at each size. Reduced automatically when the data is small.
    export_figures:
        Directory for the curve plot.
    export_html:
        Path for an HTML report.

    Returns
    -------
    DiagnosticReport
        Training sizes with mean and standard deviation of train and validation
        scores at each, plus the gap that indicates over- or underfitting.

    Notes
    -----
    **This costs roughly five cross-validations.** Five training sizes each
    cross-validated is a lot of fitting; use a smaller ``cv`` or a faster
    estimator when iterating.

    **The curves describe this estimator at these settings.** A different model
    or different hyperparameters can have a completely different shape, so do
    not conclude "more data will not help" in general from one curve.

    **Watch the bands, not just the lines.** Wide standard deviations at small
    sizes are expected; wide ones at the largest size mean the estimate itself
    is unstable.

    **The largest size is the whole train partition**, so the rightmost point is
    what you already have. Extrapolating past it is a judgement about the trend,
    not a measurement.

    See Also
    --------
    buildml.model.selection.cv_score : Scoring at the full training size.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None
    x, y, _, _, _ = _feature_target_frames(dataset, split_plan, "train")
    scoring = "f1_weighted" if task != "regression" else "r2"
    if task == "auto":
        scoring = (
            "r2"
            if pd.api.types.is_numeric_dtype(y) and y.nunique() > 15
            else "f1_weighted"
        )

    n_splits = min(cv, max(2, len(x) // 5))
    train_sizes, train_scores, valid_scores = learning_curve(
        estimator,
        x,
        y,
        cv=n_splits,
        scoring=scoring,
        train_sizes=np.linspace(0.2, 1.0, 5),
        n_jobs=None,
        shuffle=True,
        random_state=0,
    )
    tr_mean = [float(v) for v in train_scores.mean(axis=1)]
    va_mean = [float(v) for v in valid_scores.mean(axis=1)]
    tr_std = [float(v) for v in train_scores.std(axis=1)]
    va_std = [float(v) for v in valid_scores.std(axis=1)]
    sizes = [int(v) for v in train_sizes]
    gap = tr_mean[-1] - va_mean[-1]
    slope = va_mean[-1] - va_mean[0]
    payload = {
        "scoring": scoring,
        "cv_folds": int(n_splits),
        "train_sizes": sizes,
        "train_scores_mean": tr_mean,
        "valid_scores_mean": va_mean,
        "train_scores_std": tr_std,
        "valid_scores_std": va_std,
        "final_gap": float(gap),
        "valid_score_gain": float(slope),
    }
    interpretation = [
        f"Final {scoring}: train={tr_mean[-1]:.3f}, valid={va_mean[-1]:.3f}, "
        f"gap={gap:.3f}; valid gain from smallest→largest size={slope:.3f}.",
    ]
    tips = []
    if gap > 0.1:
        tips.append(
            "Large train/validation gap: likely overfitting; "
            "add regularization or more data."
        )
    elif slope > 0.03:
        tips.append(
            "Validation score still rising with more data: "
            "collecting additional labeled rows looks valuable."
        )
    else:
        tips.append(
            "Learning-curve gap is moderate and gains are flattening: "
            "prefer feature depth / model family changes over more of the same data."
        )

    report = DiagnosticReport(
        kind="learning_curve",
        payload=payload,
        recommendations=tips,
        interpretation=interpretation,
    )
    if export_figures is not None:
        # Learning-curve-only board needs a FitResult-like path; export via plot helper.
        try:
            from buildml.eda.visualize import save_figures
            from buildml.model.plot_boards import _plot_learning_curve, _require_viz
            from buildml.model.supervised import FitResult as FR

            plt, _sns = _require_viz()
            # Minimal shim when caller passed a raw estimator (common API).
            fit_shim = FR(
                estimator=estimator if hasattr(estimator, "predict") else estimator,
                task="classification" if scoring != "r2" else "regression",
                feature_columns=tuple(x.columns.astype(str)),
                target_column=str(y.name) if y.name is not None else "target",
                n_train_rows=int(len(x)),
            )
            # Ensure estimator can be cloned/fit: learning curve uses the provided estimator.
            fig, _tips = _plot_learning_curve(
                dataset, split_plan, fit_shim, plt=plt, cv=n_splits
            )
            root = save_figures({"learning_curve": fig}, export_figures)
            report.figure_dir = str(root)
            png = root / "learning_curve.png"
            if png.exists():
                report.figure_paths = {"learning_curve": str(png)}
        except Exception:  # noqa: BLE001 - structured report still valid
            pass
    if export_html is not None:
        report.export_html(export_html)
    return report


def permutation_importance_report(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    fit_result: FitResult,
    *,
    partition: Literal["train", "validation", "test"] = "test",
    n_repeats: int = 8,
    export_figures: str | Path | None = None,
    export_html: str | Path | None = None,
) -> DiagnosticReport:
    """Measure what each feature is worth by taking it away.

    Shuffle one column's values so it keeps its distribution and loses its
    relationship to the target, then re-score. However much performance drops is
    what that feature was contributing. Repeat for every column.

    The appeal over a model's built-in ``feature_importances_`` is that this
    measures contribution to *predictive performance* on held-out data, not to
    the model's internal splitting decisions. Tree impurity importance is
    notoriously biased toward high-cardinality features, which look important
    because they offer many places to split rather than because they predict
    anything. Permutation importance is model-agnostic and does not have that
    flaw.

    Parameters
    ----------
    dataset:
        The data.
    split_plan:
        Partition membership.
    fit_result:
        The fitted model to probe.
    partition:
        Which partition to measure on. Use held-out data: importance measured
        on training rows describes what the model memorised.
    n_repeats:
        Shuffles per feature. More gives a tighter estimate at proportional
        cost; the standard deviation across repeats is reported so you can tell
        whether a ranking is stable.
    export_figures:
        Directory for the importance plot.
    export_html:
        Path for an HTML report.

    Returns
    -------
    DiagnosticReport
        Features ranked by mean importance, each with its standard deviation and
        coefficient of variation, plus the near-zero features.

    Raises
    ------
    ValidationError
        If the split is missing, or if a feature column the model needs is
        absent from the partition.

    Notes
    -----
    **Correlated features share the blame and both look unimportant.** Shuffling
    one leaves the other carrying the same information, so the drop is small for
    each: even when the pair is jointly essential. Two features you expected to
    matter both scoring near zero is a signal to check their correlation, not to
    drop them.

    **Importance is a property of this model, not of the data.** A feature
    unimportant to a linear model can be central to a tree. This says what *this*
    model uses.

    **The cost is roughly ``n_features × n_repeats`` scoring passes**, which on a
    wide dataset is substantial.

    **Near-zero importance is a candidate for removal, not a verdict.** Check
    correlation first, then confirm by refitting without the feature.

    **Negative importance means shuffling helped**, which is noise: the feature
    contributes nothing and the estimate moved the wrong way by chance.

    See Also
    --------
    segment_error_report : Where the model fails rather than what it uses.
    """
    if split_plan is None:
        raise ValidationError("Split required")
    x, y, _, _, _ = _feature_target_frames(dataset, split_plan, partition)
    x = x[list(fit_result.feature_columns)]
    scoring = "f1_weighted" if fit_result.task == "classification" else "r2"
    result = permutation_importance(
        fit_result.estimator,
        x,
        y,
        n_repeats=n_repeats,
        random_state=0,
        scoring=scoring,
    )
    rows = sorted(
        (
            {
                "feature": feat,
                "importance_mean": float(mu),
                "importance_std": float(sd),
                "importance_cv": float(sd / abs(mu)) if abs(mu) > 1e-12 else None,
            }
            for feat, mu, sd in zip(
                fit_result.feature_columns,
                result.importances_mean,
                result.importances_std,
                strict=True,
            )
        ),
        key=lambda item: item["importance_mean"],
        reverse=True,
    )
    interpretation = []
    tips = []
    if rows:
        top = rows[0]
        interpretation.append(
            f"Top feature '{top['feature']}': "
            f"Δ{scoring}≈{top['importance_mean']:.4f} ± {top['importance_std']:.4f} "
            f"on partition={partition}."
        )
        tips.append(
            f"Top feature by permutation importance: {top['feature']}"
        )
        near_zero = [r["feature"] for r in rows if abs(r["importance_mean"]) < 1e-4]
        if near_zero:
            tips.append(
                f"{len(near_zero)} feature(s) near-zero importance: "
                "candidates for pruning after correlation review: "
                f"{near_zero[:8]}"
            )
        if len(rows) > 1 and abs(rows[0]["importance_mean"]) > 3 * max(
            abs(rows[1]["importance_mean"]), 1e-12
        ):
            tips.append(
                "Importance is highly concentrated: audit target leakage / proxy IDs."
            )
    else:
        tips.append("No features scored.")
    tips.append(
        "Treat correlated features carefully: importance can split across proxies."
    )

    report = DiagnosticReport(
        kind="permutation_importance",
        payload={
            "partition": partition,
            "scoring": scoring,
            "n_repeats": int(n_repeats),
            "n_rows": int(len(y)),
            "rows": rows,
        },
        recommendations=tips,
        interpretation=interpretation,
    )
    return _maybe_export_diagnostic_board(
        report,
        dataset=dataset,
        split_plan=split_plan,
        fit_result=fit_result,
        partition=partition,
        export_figures=export_figures,
        export_html=export_html,
        board_panels=("permutation_importance",),
        include_learning_curve=False,
        include_importance=True,
        n_importance_repeats=n_repeats,
    )


def _downsample(values: np.ndarray, max_points: int = 40) -> list[float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) <= max_points:
        return [float(v) for v in arr]
    idx = np.linspace(0, len(arr) - 1, max_points).astype(int)
    return [float(arr[i]) for i in idx]


def segment_error_report(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    fit_result: FitResult,
    *,
    by: str | Sequence[str],
    partition: Literal["train", "validation", "test"] = "test",
    max_segments: int = 20,
    min_segment_n: int = 5,
    export_html: str | Path | None = None,
) -> DiagnosticReport:
    """Find out who the model fails, rather than how often it fails overall.

    An aggregate metric averages over everyone, and averages hide exactly the
    failures that matter most. An overall accuracy of 0.90 is entirely
    compatible with 0.95 for the bulk of your users and 0.55 for a subgroup :
    and the aggregate will never show it, because the subgroup is small enough
    to be swamped.

    This slices predictions by columns you name and scores each slice
    separately. It is the basic tool for fairness review, for finding the
    segment where a deployment will embarrass you, and for noticing that the
    model's good average rests on being excellent at the easy cases.

    Segments smaller than ``min_segment_n`` are kept separately rather than
    ranked, because a 40% error rate over five rows is not a finding: it is two
    mistakes.

    Parameters
    ----------
    dataset:
        The data.
    split_plan:
        Partition membership.
    fit_result:
        The fitted model to assess.
    by:
        Column or columns to slice on. Several columns produce the cross
        product, which grows quickly and thins each segment.
    partition:
        Which partition to assess. Defaults to test.
    max_segments:
        Cap on segments reported, keeping the largest.
    min_segment_n:
        Below this many rows, a segment is held aside as too small to rank.
    export_html:
        Path for an HTML report, if wanted.

    Returns
    -------
    DiagnosticReport
        Per-segment metrics with row counts, the small segments kept separately,
        and the spread between the best and worst segment. Classification gets
        accuracy, error rate, per-class precision, recall and F1 for binary
        problems, and the predicted-positive rate; regression gets MAE, RMSE,
        median absolute error, and a residual summary.

    Raises
    ------
    ValidationError
        If the split is missing, or if a column named in ``by`` is not in the
        dataset.

    Notes
    -----
    **Slice by columns the model did not see, too.** Segmenting on a protected
    attribute that was deliberately excluded from the features is the whole
    point of a fairness check: a model can reproduce a disparity perfectly well
    through proxies.

    **Small segments produce large apparent differences.** Check the row count
    beside every rate before believing it; that is why the small ones are
    separated rather than mixed in.

    **The predicted-positive rate is worth as much as the accuracy.** Two
    segments can have identical accuracy while one is flagged far more often,
    which is a real difference in treatment that accuracy alone conceals.

    **Slicing many ways will eventually find a bad segment by chance.** Treat a
    surprising segment as a hypothesis to confirm on other data, not a
    conclusion.

    Examples
    --------
    Check for disparity across a protected attribute::

        report = segment_error_report(
            dataset, split_plan, fit,
            by="age_band",
            partition="test",
            min_segment_n=30,
        )
        for row in report.payload["segments"]:
            print(row)

    See Also
    --------
    permutation_importance_report : What the model uses, rather than who it fails.
    """
    if split_plan is None:
        raise ValidationError("Split required for segment error analysis")
    by_columns = _normalize_segment_columns(by, dataset.columns)
    if max_segments < 1:
        raise ValidationError("max_segments must be >= 1")
    if min_segment_n < 1:
        raise ValidationError("min_segment_n must be >= 1")

    frame = dataset._ensure_pandas()
    if partition == "train":
        indices = list(split_plan.train_indices)
    elif partition == "validation":
        indices = list(split_plan.validation_indices or ())
    else:
        indices = list(split_plan.test_indices)
    if not indices:
        raise ValidationError(f"Partition '{partition}' is empty")
    part = frame.iloc[indices]
    x = part[list(fit_result.feature_columns)]
    y = part[fit_result.target_column]
    preds = fit_result.estimator.predict(x)
    segment_labels = _segment_label_series(part, by_columns)

    counts = segment_labels.value_counts(dropna=False)
    if counts.empty:
        raise ValidationError(
            f"No segment values available for columns {by_columns!r} on partition '{partition}'"
        )
    top = counts.head(max_segments).index.tolist()
    binary_positive: str | None = None
    if fit_result.task == "classification" and hasattr(fit_result.estimator, "classes_"):
        classes = [str(c) for c in fit_result.estimator.classes_]
        if len(classes) == 2:
            binary_positive = classes[1]

    primary: list[dict[str, Any]] = []
    small: list[dict[str, Any]] = []
    if fit_result.task == "classification":
        y_true = y.astype(str).to_numpy()
        y_pred = pd.Series(preds).astype(str).to_numpy()
        for value in top:
            mask = segment_labels.to_numpy() == value
            n = int(mask.sum())
            if n == 0:
                continue
            row = _classification_segment_metrics(
                value=str(value),
                y_true=y_true[mask],
                y_pred=y_pred[mask],
                positive=binary_positive,
            )
            (primary if n >= min_segment_n else small).append(row)
        primary.sort(key=lambda item: (item["error_rate"], item["n"]), reverse=True)
        small.sort(key=lambda item: (item["error_rate"], item["n"]), reverse=True)
        metric_note = (
            "Primary segments ranked by classification error rate "
            f"(n >= {min_segment_n})."
        )
    else:
        y_true = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
        y_pred = np.asarray(preds, dtype=float)
        for value in top:
            mask = segment_labels.to_numpy() == value
            n = int(mask.sum())
            if n == 0:
                continue
            row = _regression_segment_metrics(
                value=str(value),
                y_true=y_true[mask],
                y_pred=y_pred[mask],
            )
            (primary if n >= min_segment_n else small).append(row)
        primary.sort(key=lambda item: item["mae"], reverse=True)
        small.sort(key=lambda item: item["mae"], reverse=True)
        metric_note = (
            f"Primary segments ranked by mean absolute error (n >= {min_segment_n})."
        )

    n_unique = int(counts.shape[0])
    n_omitted = max(0, n_unique - len(top))
    payload = {
        "partition": partition,
        "by": by_columns[0] if len(by_columns) == 1 else list(by_columns),
        "by_columns": list(by_columns),
        "task": fit_result.task,
        "n_rows": int(len(part)),
        "n_segments_reported": len(primary),
        "n_small_segments": len(small),
        "n_unique_segments": n_unique,
        "n_omitted_segments": n_omitted,
        "max_segments": max_segments,
        "min_segment_n": min_segment_n,
        "segments": primary,
        "small_segments": small,
        "binary_positive_class": binary_positive,
    }
    interpretation = [
        (
            f"Sliced {fit_result.task} errors on '{partition}' by "
            f"{', '.join(repr(c) for c in by_columns)}."
        ),
        metric_note,
        (
            f"Reported {len(primary)} primary segment(s) and {len(small)} small-n "
            f"segment(s); {n_omitted} less-frequent segment(s) omitted by max_segments."
        ),
    ]
    if not primary and small:
        interpretation.append(
            "No segment met min_segment_n; inspect small_segments only as unstable hints."
        )
    elif not primary and not small:
        interpretation.append("No segment rows were produced for the requested columns.")

    tips = [
        "Segment gaps are observational; they do not prove unfairness or causality.",
        (
            f"Treat segments with n < {min_segment_n} as unstable; they are listed under "
            "small_segments and excluded from primary ranking."
        ),
        "Prefer validation while exploring slices; reserve test for a fixed final estimate.",
    ]
    if primary:
        worst = primary[0]
        tips.append(
            f"Highest-error primary segment: {worst.get('segment')!r} "
            f"(n={worst.get('n')})."
        )
    elif small:
        tips.append(
            f"No primary segments; highest-error small-n preview: "
            f"{small[0].get('segment')!r} (n={small[0].get('n')})."
        )

    report = DiagnosticReport(
        kind="segment_errors",
        payload=payload,
        recommendations=tips,
        interpretation=interpretation,
    )
    if export_html is not None:
        report.export_html(export_html)
    return report


def _normalize_segment_columns(by: str | Sequence[str], columns: Sequence[str]) -> list[str]:
    if isinstance(by, str):
        names = [by]
    else:
        names = [str(item) for item in by]
    if not names:
        raise ValidationError("by must name at least one segment column")
    missing = [name for name in names if name not in columns]
    if missing:
        raise ValidationError(
            f"Segment column(s) not in the dataset: {', '.join(repr(m) for m in missing)}"
        )
    # Preserve order, drop duplicates.
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def _segment_label_series(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    parts: list[pd.Series] = []
    for column in columns:
        series = frame[column].astype("string").fillna("<NA>")
        parts.append(series.map(lambda value, col=column: f"{col}={value}"))
    if len(parts) == 1:
        return parts[0]
    joined = parts[0]
    for part in parts[1:]:
        joined = joined + " | " + part
    return joined


def _classification_segment_metrics(
    *,
    value: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive: str | None,
) -> dict[str, Any]:
    n = int(len(y_true))
    correct = int((y_true == y_pred).sum())
    row: dict[str, Any] = {
        "segment": value,
        "n": n,
        "accuracy": correct / n if n else float("nan"),
        "error_rate": 1.0 - (correct / n) if n else float("nan"),
        "n_correct": correct,
        "n_incorrect": n - correct,
    }
    if positive is not None:
        y_bin = (y_true == positive).astype(int)
        p_bin = (y_pred == positive).astype(int)
        row.update(
            {
                "precision": float(precision_score(y_bin, p_bin, zero_division=0)),
                "recall": float(recall_score(y_bin, p_bin, zero_division=0)),
                "f1": float(f1_score(y_bin, p_bin, zero_division=0)),
                "support_positive": int(y_bin.sum()),
                "predicted_positive_rate": float(p_bin.mean()) if n else float("nan"),
            }
        )
    return row


def _regression_segment_metrics(
    *,
    value: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    n = int(len(y_true))
    resid = y_true - y_pred
    abs_resid = np.abs(resid)
    return {
        "segment": value,
        "n": n,
        "mae": float(np.mean(abs_resid)) if n else float("nan"),
        "rmse": float(np.sqrt(np.mean(resid**2))) if n else float("nan"),
        "median_ae": float(np.median(abs_resid)) if n else float("nan"),
        "mean_residual": float(np.mean(resid)) if n else float("nan"),
        "max_abs_error": float(np.max(abs_resid)) if n else float("nan"),
    }


def _maybe_export_diagnostic_board(
    report: DiagnosticReport,
    *,
    dataset: Dataset,
    split_plan: SplitPlan,
    fit_result: FitResult,
    partition: Literal["train", "validation", "test"],
    export_figures: str | Path | None,
    export_html: str | Path | None,
    board_panels: tuple[str, ...] = (),
    include_learning_curve: bool = False,
    include_importance: bool = False,
    n_importance_repeats: int = 6,
) -> DiagnosticReport:
    """Optionally render/export plot-board panels for a diagnostic report."""
    if export_figures is None and export_html is None:
        return report
    try:
        from buildml.model.plot_boards import build_eval_plot_board
    except Exception:  # noqa: BLE001
        if export_html is not None:
            report.export_html(export_html)
        return report

    board = build_eval_plot_board(
        dataset,
        split_plan,
        fit_result,
        partition=partition,
        include_learning_curve=include_learning_curve,
        include_importance=include_importance,
        n_importance_repeats=n_importance_repeats,
        export_figures=export_figures,
        export_html=None,
        show=False,
    )
    # Keep only requested panels when a subset was asked for.
    if board_panels and board.figure_paths:
        keep = {k: v for k, v in board.figure_paths.items() if k in board_panels}
        report.figure_paths = keep
    else:
        report.figure_paths = dict(board.figure_paths)
    report.figure_dir = board.figure_dir
    if export_html is not None:
        payload = report.to_dict()
        payload["figure_paths"] = report.figure_paths
        payload["figure_dir"] = report.figure_dir
        from buildml.model.html_diagnostics import export_diagnostics_html

        selected_figures = (
            {key: value for key, value in board.figures.items() if key in board_panels}
            if board_panels
            else board.figures
        )
        report.html_path = str(
            export_diagnostics_html(payload, export_html, figures=selected_figures)
        )
    return report
