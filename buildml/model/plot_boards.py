"""Evaluation plot boards for classification and regression.

Adaptive: panels that need ``predict_proba``, binary targets, or numeric
features degrade gracefully with structured skip reasons rather than failing
the whole board.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import learning_curve

from buildml.core.errors import MissingExtraError, ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.eda.visualize import save_figures
from buildml.explain.schemas import Finding, Recommendation
from buildml.model.evidence import (
    compatibility_recommendations,
    evidence_for_plot_board,
)
from buildml.model.supervised import FitResult, _feature_target_frames


def _require_viz() -> tuple[Any, Any]:
    try:
        import matplotlib

        # Non-interactive export path — avoid Tk/GUI backends in library use.
        if str(matplotlib.get_backend()).lower() != "agg":
            matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:
        raise MissingExtraError("viz", "Evaluation plot boards") from exc
    sns.set_theme(style="whitegrid", context="notebook")
    return plt, sns


@dataclass(slots=True)
class PlotBoardReport:
    """Collection of evaluation figures plus skip/interpretation metadata."""

    task: Literal["classification", "regression"]
    partition: str
    figures: dict[str, Any] = field(default_factory=dict)
    skipped: list[dict[str, str]] = field(default_factory=list)
    interpretation: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    figure_dir: str | None = None
    html_path: str | None = None
    figure_paths: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    recommendation_details: list[Recommendation] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "partition": self.partition,
            "figures": {
                key: ("figure" if value is not None and not isinstance(value, dict) else value)
                for key, value in self.figures.items()
            },
            "skipped": list(self.skipped),
            "interpretation": list(self.interpretation),
            "recommendations": list(self.recommendations),
            "figure_dir": self.figure_dir,
            "html_path": self.html_path,
            "figure_paths": dict(self.figure_paths),
            "metrics": dict(self.metrics),
            "findings": [item.to_dict() for item in self.findings],
            "recommendation_details": [
                item.to_dict() for item in self.recommendation_details
            ],
            "limitations": list(self.limitations),
            "methods": list(self.methods),
        }

    def show(self) -> None:
        print(
            f"Eval plot board · {self.task} · partition={self.partition} · "
            f"figures={len(self.figure_paths) or len(self.figures)} · "
            f"skipped={len(self.skipped)}"
        )
        for tip in self.interpretation[:8]:
            print(f"* {tip}")
        for tip in self.recommendations[:6]:
            print(f"- {tip}")
        if self.figure_dir:
            print(f"Figures: {self.figure_dir}")
        if self.html_path:
            print(f"HTML: {self.html_path}")


def build_eval_plot_board(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    fit_result: FitResult,
    *,
    partition: Literal["train", "validation", "test"] = "test",
    include_learning_curve: bool = True,
    include_importance: bool = True,
    n_importance_repeats: int = 6,
    learning_curve_cv: int = 3,
    export_figures: str | Path | None = None,
    export_html: str | Path | None = None,
    show: bool = False,
) -> PlotBoardReport:
    """Build a rich visual diagnostic board after fit/evaluate.

    Panels (adaptive)
    -----------------
    Classification:
      confusion matrix, ROC, PR, calibration reliability, threshold tradeoffs,
      learning curve, permutation importance bars.
    Regression:
      residual scatter, residual histogram, predicted-vs-actual, learning curve,
      permutation importance bars.

    Notes
    -----
    **Leakage:** Learning curves refit clones on the train partition only.
    Importance / calibration / threshold panels score the requested partition.
    Requires ``pip install 'buildml[viz]'`` for figure rendering.
    """
    if split_plan is None:
        raise ValidationError("Split required for evaluation plot boards")

    plt, sns = _require_viz()
    x, y_true, _, _ = _feature_target_frames(dataset, split_plan, partition)
    x = x[list(fit_result.feature_columns)]
    y_pred = fit_result.estimator.predict(x)

    figures: dict[str, Any] = {}
    skipped: list[dict[str, str]] = []
    interpretation: list[str] = []
    recommendations: list[str] = []
    metrics: dict[str, Any] = {}

    if fit_result.task == "classification":
        _classification_board(
            figures=figures,
            skipped=skipped,
            interpretation=interpretation,
            recommendations=recommendations,
            plt=plt,
            sns=sns,
            estimator=fit_result.estimator,
            x=x,
            y_true=y_true,
            y_pred=y_pred,
            metrics=metrics,
        )
    else:
        _regression_board(
            figures=figures,
            skipped=skipped,
            interpretation=interpretation,
            recommendations=recommendations,
            plt=plt,
            sns=sns,
            y_true=y_true,
            y_pred=y_pred,
            metrics=metrics,
        )

    if include_learning_curve:
        try:
            fig, tips = _plot_learning_curve(
                dataset,
                split_plan,
                fit_result,
                plt=plt,
                cv=learning_curve_cv,
            )
            figures["learning_curve"] = fig
            interpretation.extend(tips)
        except Exception as exc:  # noqa: BLE001 - adaptive board
            skipped.append({"panel": "learning_curve", "reason": str(exc)})
    else:
        skipped.append({"panel": "learning_curve", "reason": "disabled by caller"})

    if include_importance:
        try:
            fig, tips = _plot_permutation_importance(
                fit_result=fit_result,
                x=x,
                y_true=y_true,
                plt=plt,
                sns=sns,
                n_repeats=n_importance_repeats,
            )
            figures["permutation_importance"] = fig
            interpretation.extend(tips)
        except Exception as exc:  # noqa: BLE001
            skipped.append({"panel": "permutation_importance", "reason": str(exc)})
    else:
        skipped.append({"panel": "permutation_importance", "reason": "disabled by caller"})

    findings, recommendation_details, limitations = evidence_for_plot_board(
        fit_result.task, partition, metrics, skipped
    )
    recommendations = compatibility_recommendations(recommendation_details)

    report = PlotBoardReport(
        task=fit_result.task,
        partition=partition,
        figures=figures,
        skipped=skipped,
        interpretation=interpretation,
        recommendations=recommendations,
        metrics=metrics,
        findings=findings,
        recommendation_details=recommendation_details,
        limitations=limitations,
        methods=[
            "Panels use predictions on the named partition; learning curves "
            "refit clones on train-only cross-validation.",
            "Permutation importance uses repeated score degradation on the named partition.",
        ],
    )

    if export_figures is not None:
        root = save_figures(figures, export_figures)
        report.figure_dir = str(root)
        report.figure_paths = {
            key: str(root / f"{key}.png")
            for key, fig in figures.items()
            if fig is not None and not isinstance(fig, dict) and (root / f"{key}.png").exists()
        }

    if export_html is not None:
        from buildml.model.html_diagnostics import export_diagnostics_html

        report.html_path = str(
            export_diagnostics_html(
                {
                    "kind": "eval_plot_board",
                    "task": report.task,
                    "partition": report.partition,
                    "interpretation": report.interpretation,
                    "recommendations": report.recommendations,
                    "skipped": report.skipped,
                    "figure_paths": report.figure_paths,
                    "figure_dir": report.figure_dir,
                    "metrics": report.metrics,
                    "findings": [item.to_dict() for item in report.findings],
                    "recommendation_details": [
                        item.to_dict() for item in report.recommendation_details
                    ],
                    "limitations": report.limitations,
                    "methods": report.methods,
                },
                export_html,
                title="BuildML Evaluation Plot Board",
                figures=figures,
            )
        )

    # Close figures after optional save to avoid matplotlib open-figure warnings.
    for fig in figures.values():
        if fig is not None and not isinstance(fig, dict):
            try:
                plt.close(fig)
            except Exception:  # noqa: BLE001
                pass

    if show:
        report.show()
    return report


def _classification_board(
    *,
    figures: dict[str, Any],
    skipped: list[dict[str, str]],
    interpretation: list[str],
    recommendations: list[str],
    plt: Any,
    sns: Any,
    estimator: Any,
    x: pd.DataFrame,
    y_true: pd.Series,
    y_pred: Any,
    metrics: dict[str, Any],
) -> None:
    labels = sorted(pd.unique(pd.concat([y_true.astype(str), pd.Series(y_pred).astype(str)])))
    cm = confusion_matrix(y_true.astype(str), pd.Series(y_pred).astype(str), labels=labels)
    figures["confusion_matrix"] = _plot_confusion(cm, labels, plt, sns)
    diag = float(np.trace(cm))
    total = float(cm.sum()) or 1.0
    interpretation.append(
        f"Confusion matrix accuracy proxy (trace/total) = {diag / total:.3f} "
        f"across {len(labels)} labels."
    )
    metrics.update(
        {
            "accuracy_proxy": diag / total,
            "n_labels": len(labels),
        }
    )
    # Off-diagonal concentration tip
    if cm.shape[0] == 2 and total > 0:
        fn = float(cm[1, 0]) if cm.shape == (2, 2) else 0.0
        fp = float(cm[0, 1]) if cm.shape == (2, 2) else 0.0
        metrics.update({"false_negatives": fn, "false_positives": fp})
        if fn > fp * 1.5:
            recommendations.append(
                "False negatives dominate — lower the decision threshold or "
                "favor recall-oriented policies."
            )
        elif fp > fn * 1.5:
            recommendations.append(
                "False positives dominate — raise the threshold or tighten precision."
            )

    has_proba = hasattr(estimator, "predict_proba")
    if not has_proba:
        for panel in ("roc_curve", "pr_curve", "calibration", "threshold_tradeoff"):
            skipped.append(
                {"panel": panel, "reason": "estimator lacks predict_proba"}
            )
        recommendations.append(
            "Estimator has no predict_proba — ROC/PR/calibration/threshold "
            "panels skipped. Prefer probabilistic classifiers or CalibratedClassifierCV."
        )
        return

    try:
        proba = estimator.predict_proba(x)
    except Exception as exc:  # noqa: BLE001
        for panel in ("roc_curve", "pr_curve", "calibration", "threshold_tradeoff"):
            skipped.append({"panel": panel, "reason": f"predict_proba failed: {exc}"})
        return

    classes = list(getattr(estimator, "classes_", range(proba.shape[1])))
    if len(classes) != 2:
        skipped.append(
            {
                "panel": "roc_curve",
                "reason": "binary ROC board omitted for multiclass; see metrics",
            }
        )
        skipped.append(
            {
                "panel": "pr_curve",
                "reason": "binary PR board omitted for multiclass; see metrics",
            }
        )
        skipped.append(
            {
                "panel": "calibration",
                "reason": "binary reliability diagram omitted for multiclass",
            }
        )
        skipped.append(
            {
                "panel": "threshold_tradeoff",
                "reason": "threshold sweep currently binary-only",
            }
        )
        recommendations.append(
            "Multiclass probabilities available — use one-vs-rest calibration "
            "and per-class threshold policies outside the binary board."
        )
        return

    y_bin = pd.Series(y_true).astype(str)
    positive = str(classes[1])
    y_pos = (y_bin == positive).astype(int).to_numpy()
    prob_pos = proba[:, 1]

    fpr, tpr, _ = roc_curve(y_pos, prob_pos)
    prec, rec, _ = precision_recall_curve(y_pos, prob_pos)
    roc_auc = float(roc_auc_score(y_pos, prob_pos))
    ap = float(average_precision_score(y_pos, prob_pos))
    figures["roc_curve"] = _plot_roc(fpr, tpr, roc_auc, plt)
    figures["pr_curve"] = _plot_pr(rec, prec, ap, plt)
    interpretation.append(
        f"ROC-AUC={roc_auc:.3f}; Average Precision (PR-AUC)={ap:.3f} "
        f"(positive class='{positive}')."
    )
    metrics.update(
        {
            "roc_auc": roc_auc,
            "average_precision": ap,
            "positive_class": positive,
            "prevalence": float(y_pos.mean()),
        }
    )
    if roc_auc < 0.65:
        recommendations.append(
            "ROC-AUC is weak — revisit features/model family before thresholding."
        )
    if ap + 1e-9 < (y_pos.mean() + 0.05):
        recommendations.append(
            "PR-AUC near prevalence — ranking quality may be limited "
            "for the minority class."
        )

    try:
        frac_pos, mean_pred = calibration_curve(
            y_pos, prob_pos, n_bins=min(10, max(3, len(y_pos) // 5)), strategy="quantile"
        )
        ece = float(np.mean(np.abs(np.asarray(frac_pos) - np.asarray(mean_pred))))
        figures["calibration"] = _plot_calibration(mean_pred, frac_pos, ece, plt)
        interpretation.append(
            f"Calibration ECE (bin-mean |frac-mean_pred|) ≈ {ece:.3f}; "
            "0 is perfect reliability."
        )
        metrics["ece"] = ece
        if ece > 0.1:
            recommendations.append(
                "Reliability gap is material — apply CalibratedClassifierCV "
                "(isotonic/sigmoid) on train folds."
            )
    except Exception as exc:  # noqa: BLE001
        skipped.append({"panel": "calibration", "reason": str(exc)})

    rows = []
    for t in np.linspace(0.05, 0.95, 19):
        pred = (prob_pos >= t).astype(int)
        rows.append(
            (
                float(t),
                float(precision_score(y_pos, pred, zero_division=0)),
                float(recall_score(y_pos, pred, zero_division=0)),
                float(f1_score(y_pos, pred, zero_division=0)),
            )
        )
    figures["threshold_tradeoff"] = _plot_threshold_tradeoff(rows, plt)
    best = max(rows, key=lambda item: item[3])
    interpretation.append(
        f"Best F1 threshold≈{best[0]:.2f} "
        f"(precision={best[1]:.3f}, recall={best[2]:.3f}, F1={best[3]:.3f})."
    )
    metrics["best_f1_threshold"] = {
        "threshold": best[0],
        "precision": best[1],
        "recall": best[2],
        "f1": best[3],
    }
    recommendations.append(
        "Pick thresholds from cost/benefit, not F1 alone — the tradeoff curve "
        "shows the precision/recall frontier."
    )


def _regression_board(
    *,
    figures: dict[str, Any],
    skipped: list[dict[str, str]],
    interpretation: list[str],
    recommendations: list[str],
    plt: Any,
    sns: Any,
    y_true: pd.Series,
    y_pred: Any,
    metrics: dict[str, Any],
) -> None:
    y_t = y_true.to_numpy(dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    residuals = y_t - y_p
    figures["residuals_scatter"] = _plot_residual_scatter(y_p, residuals, plt, sns)
    figures["residuals_hist"] = _plot_residual_hist(residuals, plt, sns)
    figures["predicted_vs_actual"] = _plot_pred_vs_actual(y_t, y_p, plt, sns)
    rmse = float(np.sqrt(np.mean(residuals**2)))
    bias = float(np.mean(residuals))
    interpretation.append(
        f"Residual bias (mean)={bias:.4g}; RMSE={rmse:.4g}; "
        f"q05/q95=({np.quantile(residuals, 0.05):.4g}, "
        f"{np.quantile(residuals, 0.95):.4g})."
    )
    metrics.update(
        {
            "rmse": rmse,
            "residual_bias": bias,
            "residual_q05": float(np.quantile(residuals, 0.05)),
            "residual_q95": float(np.quantile(residuals, 0.95)),
        }
    )
    if abs(bias) > 0.25 * (rmse + 1e-12):
        recommendations.append(
            "Non-trivial residual bias — check target transforms or missing intercept structure."
        )
    # Heteroscedasticity heuristic: correlate |residual| with prediction
    if len(y_p) >= 8:
        corr = float(np.corrcoef(np.abs(residuals), y_p)[0, 1]) if np.std(y_p) > 0 else 0.0
        metrics["heteroscedasticity_correlation"] = corr
        if abs(corr) > 0.35:
            recommendations.append(
                f"|residual| correlates with prediction (r≈{corr:.2f}) — "
                "consider variance-stabilizing transforms or heteroscedastic models."
            )
    for panel in ("roc_curve", "pr_curve", "calibration", "threshold_tradeoff", "confusion_matrix"):
        skipped.append({"panel": panel, "reason": "not applicable to regression"})


def _plot_confusion(cm: np.ndarray, labels: list[Any], plt: Any, sns: Any) -> Any:
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[str(x) for x in labels],
        yticklabels=[str(x) for x in labels],
        ax=ax,
        cbar=False,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    return fig


def _plot_roc(fpr: np.ndarray, tpr: np.ndarray, auc: float, plt: Any) -> Any:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#1d3557", lw=2, label=f"ROC AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], ls="--", color="#adb5bd", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def _plot_pr(recall: np.ndarray, precision: np.ndarray, ap: float, plt: Any) -> Any:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#e76f51", lw=2, label=f"AP={ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def _plot_calibration(
    mean_pred: np.ndarray,
    frac_pos: np.ndarray,
    ece: float,
    plt: Any,
) -> Any:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], ls="--", color="#adb5bd", label="Perfect")
    ax.plot(mean_pred, frac_pos, marker="o", color="#2a9d8f", label=f"Model (ECE≈{ece:.3f})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration reliability diagram")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def _plot_threshold_tradeoff(rows: list[tuple[float, float, float, float]], plt: Any) -> Any:
    thr = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thr, [r[1] for r in rows], label="Precision", color="#264653")
    ax.plot(thr, [r[2] for r in rows], label="Recall", color="#e9c46a")
    ax.plot(thr, [r[3] for r in rows], label="F1", color="#e76f51")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold tradeoff")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    return fig


def _plot_residual_scatter(y_pred: np.ndarray, residuals: np.ndarray, plt: Any, sns: Any) -> Any:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(x=y_pred, y=residuals, ax=ax, alpha=0.65, color="#1d3557", s=36)
    ax.axhline(0.0, color="#adb5bd", ls="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (actual − predicted)")
    ax.set_title("Residuals vs predicted")
    fig.tight_layout()
    return fig


def _plot_residual_hist(residuals: np.ndarray, plt: Any, sns: Any) -> Any:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.histplot(residuals, kde=True, ax=ax, color="#457b9d")
    ax.axvline(0.0, color="#adb5bd", ls="--")
    ax.set_title("Residual distribution")
    ax.set_xlabel("Residual")
    fig.tight_layout()
    return fig


def _plot_pred_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, plt: Any, sns: Any) -> Any:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.scatterplot(x=y_true, y=y_pred, ax=ax, alpha=0.65, color="#2a9d8f", s=36)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], ls="--", color="#adb5bd")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs actual")
    fig.tight_layout()
    return fig


def _plot_learning_curve(
    dataset: Dataset,
    split_plan: SplitPlan,
    fit_result: FitResult,
    *,
    plt: Any,
    cv: int,
) -> tuple[Any, list[str]]:
    from sklearn.base import clone

    x_train, y_train, _, _ = _feature_target_frames(dataset, split_plan, "train")
    x_train = x_train[list(fit_result.feature_columns)]
    scoring = "f1_weighted" if fit_result.task == "classification" else "r2"
    n_splits = min(cv, max(2, len(x_train) // 5))
    train_sizes, train_scores, valid_scores = learning_curve(
        clone(fit_result.estimator),
        x_train,
        y_train,
        cv=n_splits,
        scoring=scoring,
        train_sizes=np.linspace(0.2, 1.0, 5),
        shuffle=True,
        random_state=0,
    )
    tr_mean = train_scores.mean(axis=1)
    va_mean = valid_scores.mean(axis=1)
    tr_std = train_scores.std(axis=1)
    va_std = valid_scores.std(axis=1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(train_sizes, tr_mean, "o-", color="#1d3557", label="Train")
    ax.fill_between(train_sizes, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15, color="#1d3557")
    ax.plot(train_sizes, va_mean, "o-", color="#e76f51", label="CV validation")
    ax.fill_between(train_sizes, va_mean - va_std, va_mean + va_std, alpha=0.15, color="#e76f51")
    ax.set_xlabel("Training examples")
    ax.set_ylabel(scoring)
    ax.set_title("Learning curve (train partition CV)")
    ax.legend(loc="best")
    fig.tight_layout()
    gap = float(tr_mean[-1] - va_mean[-1])
    tips = [
        f"Learning-curve final gap (train−valid {scoring})≈{gap:.3f} "
        f"at n={int(train_sizes[-1])}."
    ]
    if gap > 0.1:
        tips.append("Large gap suggests overfitting — regularize or gather more train data.")
    return fig, tips


def _plot_permutation_importance(
    *,
    fit_result: FitResult,
    x: pd.DataFrame,
    y_true: pd.Series,
    plt: Any,
    sns: Any,
    n_repeats: int,
) -> tuple[Any, list[str]]:
    scoring = "f1_weighted" if fit_result.task == "classification" else "r2"
    result = permutation_importance(
        fit_result.estimator,
        x,
        y_true,
        n_repeats=n_repeats,
        random_state=0,
        scoring=scoring,
    )
    order = np.argsort(result.importances_mean)[::-1]
    names = [fit_result.feature_columns[i] for i in order]
    means = result.importances_mean[order]
    stds = result.importances_std[order]
    top_n = min(20, len(names))
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.35 * top_n + 1.5)))
    y_labels = [str(n) for n in names[:top_n]]
    sns.barplot(
        x=means[:top_n],
        y=y_labels,
        hue=y_labels,
        ax=ax,
        palette="crest",
        orient="h",
        legend=False,
    )
    ax.errorbar(
        means[:top_n],
        range(top_n),
        xerr=stds[:top_n],
        fmt="none",
        ecolor="#6c757d",
        capsize=2,
    )
    ax.set_xlabel(f"Permutation importance ({scoring})")
    ax.set_title("Permutation feature importance")
    fig.tight_layout()
    tips = []
    if len(names):
        tips.append(
            f"Top permutation feature: '{names[0]}' "
            f"(Δ{scoring}≈{float(means[0]):.4f} ± {float(stds[0]):.4f})."
        )
        if len(names) > 1 and abs(float(means[0])) > 3 * max(abs(float(means[1])), 1e-12):
            tips.append(
                "Importance is highly concentrated in one feature — "
                "audit leakage/proxy targets."
            )
    tips.append("Correlated features can split importance — do not drop proxies blindly.")
    return fig, tips
