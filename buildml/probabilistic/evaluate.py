"""Evaluate probabilistic plans with proper scoring and interval coverage."""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import PartitionName, SplitPlan, frame_for_partition
from buildml.probabilistic.conformal import regression_intervals
from buildml.probabilistic.features import (
    decode_predictions,
    matrix_from_frame,
    norm_ppf,
)
from buildml.probabilistic.predict import predict_interval
from buildml.probabilistic.results import ProbabilisticEvalResult, ProbabilisticPlan

PartitionOrAll = PartitionName | Literal["all"]


def evaluate_probabilistic(
    dataset: Dataset,
    plan: ProbabilisticPlan,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionOrAll = "validation",
    alpha: float | None = None,
) -> ProbabilisticEvalResult:
    """Score a holdout partition with point + uncertainty metrics.

    Regression metrics include MAE/RMSE/R², Gaussian NLL (when ``return_std``
    is available), and interval coverage / mean width for the plan's interval
    method. Classification metrics include accuracy/F1, log-loss (NLL), and
    Brier (binary), plus prediction-set coverage when conformal was enabled.

    Holdout rows are never used for fitting or conformal calibration.
    """
    if plan is None:
        raise ValidationError("No ProbabilisticPlan. Call fit_probabilistic first.")

    resolved_alpha = float(plan.alpha if alpha is None else alpha)
    if partition == "all":
        frame = dataset._ensure_pandas()
        part_name = "all"
    else:
        if split_plan is None:
            raise ValidationError(
                f"partition='{partition}' requires a SplitPlan. Call session.split(...)."
            )
        frame = frame_for_partition(dataset, split_plan, partition)
        part_name = str(partition)

    missing = [c for c in plan.columns if c not in frame.columns]
    if missing:
        raise ValidationError(f"Missing feature columns for evaluation: {missing}")
    if plan.target_column not in frame.columns:
        raise ValidationError(
            f"Target column {plan.target_column!r} missing from evaluation frame."
        )

    disclosures = [
        "Probabilistic evaluation scores a holdout partition; rows were never "
        "used for fit or conformal calibration.",
        f"estimator={plan.estimator_name}, alpha={resolved_alpha}, "
        f"interval_method={plan.interval_method}.",
        "Classical Session.calibration() is unchanged and still targets "
        "classical fit(...) classifiers; this path reports NLL/Brier directly.",
    ]
    warnings: list[str] = []
    metrics: dict[str, float] = {}
    n_rows = int(len(frame))
    coverage: float | None = None
    mean_width: float | None = None

    if n_rows < 1:
        warnings.append("Evaluation partition is empty; metrics are empty.")
        return ProbabilisticEvalResult(
            partition=part_name,
            estimator_name=plan.estimator_name,
            task=plan.task,
            n_rows=0,
            alpha=resolved_alpha,
            metrics=metrics,
            disclosures=tuple(disclosures),
            warnings=tuple(warnings),
        )

    y_true = frame[plan.target_column]
    if y_true.isna().any():
        warnings.append(
            "Evaluation partition contains null targets; those rows are dropped."
        )
        mask = ~y_true.isna()
        frame = frame.loc[mask]
        y_true = y_true.loc[mask]
        n_rows = int(len(frame))

    if n_rows < 1:
        warnings.append("No labeled evaluation rows after dropping nulls.")
        return ProbabilisticEvalResult(
            partition=part_name,
            estimator_name=plan.estimator_name,
            task=plan.task,
            n_rows=0,
            alpha=resolved_alpha,
            metrics=metrics,
            disclosures=tuple(disclosures),
            warnings=tuple(warnings),
        )

    x = matrix_from_frame(frame, list(plan.columns))

    if plan.task == "regression":
        y_num = y_true.to_numpy(dtype=float)
        if plan.supports_return_std:
            mean, std = plan.estimator_.predict(x, return_std=True)
            y_hat = np.asarray(mean, dtype=float)
            std_arr = np.asarray(std, dtype=float)
            metrics["nll"] = float(_gaussian_nll(y_num, y_hat, std_arr))
        else:
            y_hat = np.asarray(plan.estimator_.predict(x), dtype=float)
        metrics["mae"] = float(mean_absolute_error(y_num, y_hat))
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_num, y_hat)))
        metrics["r2"] = float(r2_score(y_num, y_hat))

        # Interval coverage for the primary interval method.
        try:
            interval = predict_interval(
                dataset,
                plan,
                split_plan,
                partition=partition if partition != "all" else "all",
                alpha=resolved_alpha,
            )
            if interval.lower is not None and interval.upper is not None:
                lo = np.asarray(interval.lower, dtype=float)
                hi = np.asarray(interval.upper, dtype=float)
                # Align lengths if partition='all' vs filtered frame — rebuild
                # from the same x used above when lengths mismatch.
                if lo.shape[0] != y_num.shape[0]:
                    lo, hi = _regression_bounds_for_x(
                        plan, x, y_hat=y_hat, alpha=resolved_alpha
                    )
                inside = (y_num >= lo) & (y_num <= hi)
                coverage = float(np.mean(inside))
                mean_width = float(np.mean(hi - lo))
                metrics["interval_coverage"] = coverage
                metrics["mean_interval_width"] = mean_width
                metrics["interval_score"] = float(
                    _winkler_interval_score(y_num, lo, hi, resolved_alpha)
                )
        except ValidationError as exc:
            warnings.append(f"Interval metrics skipped: {exc}")
    else:
        raw = plan.estimator_.predict(x)
        preds = decode_predictions(raw, plan.label_encoder_)
        y_true_s = y_true.astype(str).to_numpy()
        y_pred_s = np.asarray([str(v) for v in preds])
        metrics["accuracy"] = float(accuracy_score(y_true_s, y_pred_s))
        metrics["f1_macro"] = float(
            f1_score(y_true_s, y_pred_s, average="macro", zero_division=0)
        )
        metrics["f1_weighted"] = float(
            f1_score(y_true_s, y_pred_s, average="weighted", zero_division=0)
        )
        if plan.supports_predict_proba:
            proba = np.asarray(plan.estimator_.predict_proba(x), dtype=float)
            classes = [str(c) for c in (plan.classes_ or ())]
            # Align log_loss labels with probability columns.
            metrics["nll"] = float(
                log_loss(y_true_s, proba, labels=classes)
            )
            if proba.shape[1] == 2:
                # Brier on the positive class column (last label in sorted vocab).
                pos = proba[:, 1]
                y_bin = (y_true_s == classes[1]).astype(int)
                metrics["brier"] = float(brier_score_loss(y_bin, pos))
                metrics["ece"] = float(_expected_calibration_error(y_bin, pos))

        if plan.conformal_quantile_ is not None:
            try:
                interval = predict_interval(
                    dataset,
                    plan,
                    split_plan,
                    partition=partition if partition != "all" else "all",
                    alpha=resolved_alpha,
                )
                if interval.prediction_sets is not None:
                    sets = interval.prediction_sets
                    if len(sets) != n_rows:
                        # Rebuild on filtered frame features.
                        from buildml.probabilistic.conformal import (
                            classification_prediction_sets,
                        )

                        proba = np.asarray(
                            plan.estimator_.predict_proba(x), dtype=float
                        )
                        sets = tuple(
                            classification_prediction_sets(
                                proba,
                                plan.conformal_quantile_,
                                plan.classes_ or (),
                            )
                        )
                    hits = [
                        y_true_s[i] in {str(v) for v in sets[i]}
                        for i in range(n_rows)
                    ]
                    coverage = float(np.mean(hits))
                    mean_width = float(np.mean([len(s) for s in sets]))
                    metrics["set_coverage"] = coverage
                    metrics["mean_set_size"] = mean_width
            except ValidationError as exc:
                warnings.append(f"Prediction-set metrics skipped: {exc}")

    return ProbabilisticEvalResult(
        partition=part_name,
        estimator_name=plan.estimator_name,
        task=plan.task,
        n_rows=n_rows,
        alpha=resolved_alpha,
        metrics=metrics,
        interval_coverage=coverage,
        mean_interval_width=mean_width,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )


def _gaussian_nll(y: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    """Average Gaussian negative log-likelihood."""
    s = np.maximum(np.asarray(std, dtype=float), 1e-12)
    m = np.asarray(mean, dtype=float)
    yy = np.asarray(y, dtype=float)
    return float(np.mean(0.5 * np.log(2 * np.pi * s**2) + 0.5 * ((yy - m) / s) ** 2))


def _winkler_interval_score(
    y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float
) -> float:
    """Mean Winkler / interval score (lower is better)."""
    yy = np.asarray(y, dtype=float)
    lower = np.asarray(lo, dtype=float)
    upper = np.asarray(hi, dtype=float)
    width = upper - lower
    below = yy < lower
    above = yy > upper
    score = width.copy()
    score[below] += (2.0 / alpha) * (lower[below] - yy[below])
    score[above] += (2.0 / alpha) * (yy[above] - upper[above])
    return float(np.mean(score))


def _expected_calibration_error(
    y_bin: np.ndarray, proba: np.ndarray, n_bins: int = 10
) -> float:
    """Simple equal-width ECE for binary probabilities."""
    p = np.asarray(proba, dtype=float)
    y = np.asarray(y_bin, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(p)
    if n == 0:
        return 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(y[mask]))
        conf = float(np.mean(p[mask]))
        ece += (np.sum(mask) / n) * abs(acc - conf)
    return float(ece)


def _regression_bounds_for_x(
    plan: ProbabilisticPlan,
    x: np.ndarray,
    *,
    y_hat: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild interval bounds on an already-filtered design matrix."""
    if plan.conformal and plan.conformal_quantile_ is not None:
        return regression_intervals(y_hat, plan.conformal_quantile_)
    if plan.supports_return_std:
        _, std = plan.estimator_.predict(x, return_std=True)
        z = norm_ppf(1.0 - alpha / 2.0)
        std_arr = np.asarray(std, dtype=float)
        return y_hat - z * std_arr, y_hat + z * std_arr
    raise ValidationError("No interval method available for coverage metrics.")
