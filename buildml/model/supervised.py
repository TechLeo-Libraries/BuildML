"""Supervised estimator helpers with leakage-safe fit scope and deep evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin, clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_recall_fscore_support,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.validation import has_fit_parameter

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.data.dataset import Dataset
from buildml.data.engines.prep import MaterializePrepResult, prepare_design_frame
from buildml.data.splits import SplitPlan, assert_fit_partition, frame_for_partition

TaskType = Literal["classification", "regression", "auto"]


@dataclass(slots=True)
class FitResult:
    """Outcome of fitting an estimator on the train partition."""

    estimator: Any
    task: Literal["classification", "regression"]
    feature_columns: tuple[str, ...]
    target_column: str
    n_train_rows: int
    weight_column: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimator": type(self.estimator).__name__,
            "task": self.task,
            "feature_columns": list(self.feature_columns),
            "target_column": self.target_column,
            "n_train_rows": self.n_train_rows,
            "weight_column": self.weight_column,
        }


@dataclass(slots=True)
class EvaluateResult:
    """Deep evaluation card for a fitted estimator on a partition."""

    partition: str
    task: Literal["classification", "regression"]
    metrics: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    n_rows: int = 0
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "task": self.task,
            "metrics": dict(self.metrics),
            "diagnostics": self.diagnostics,
            "n_rows": self.n_rows,
            "recommendations": list(self.recommendations),
        }

    def show(self) -> None:
        """Print a high-signal evaluation digest."""
        print(f"Evaluate · {self.task} · partition={self.partition} · n={self.n_rows}")
        for key, value in self.metrics.items():
            print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")
        for tip in self.recommendations[:10]:
            print(f"  - {tip}")


def weight_column(dataset: Dataset) -> str | None:
    """Return the single weight-role column, or None when unset."""
    cols = dataset.role_columns(ColumnRole.WEIGHT)
    if not cols:
        return None
    if len(cols) > 1:
        raise ValidationError(
            f"Expected at most one weight column, found {len(cols)}: {cols}"
        )
    return cols[0]


def resolve_feature_columns(dataset: Dataset) -> list[str]:
    """Resolve modeling feature columns, always excluding weight/id/group/time roles."""
    target = dataset.require_target()
    feature_cols = dataset.role_columns(ColumnRole.FEATURE)
    if not feature_cols:
        skip = {
            *dataset.role_columns(ColumnRole.TARGET),
            *dataset.role_columns(ColumnRole.ID),
            *dataset.role_columns(ColumnRole.IGNORE),
            *dataset.role_columns(ColumnRole.GROUP),
            *dataset.role_columns(ColumnRole.TIME),
            *dataset.role_columns(ColumnRole.WEIGHT),
        }
        feature_cols = [c for c in dataset.columns if c not in skip and c != target]
    weight = weight_column(dataset)
    if weight is not None and weight in feature_cols:
        raise ValidationError(
            f"Weight column '{weight}' cannot also be treated as a feature. "
            "Remove it from FEATURE roles or clear the WEIGHT role."
        )
    if not feature_cols:
        raise ValidationError("No feature columns available for modeling")
    return feature_cols


def validate_sample_weights(weights: pd.Series, *, column: str) -> pd.Series:
    """Validate a weight series for sklearn sample_weight use."""
    numeric = pd.to_numeric(weights, errors="coerce")
    if numeric.isna().any():
        raise ValidationError(
            f"Weight column '{column}' contains non-numeric or missing values"
        )
    values = numeric.to_numpy(dtype=float)
    if np.any(values < 0):
        raise ValidationError(f"Weight column '{column}' must be non-negative")
    if not np.any(values > 0):
        raise ValidationError(f"Weight column '{column}' must contain at least one positive weight")
    return numeric.astype(float)


def fit_kwargs_for_sample_weight(estimator: Any, sample_weight: pd.Series | None) -> dict[str, Any]:
    """Build fit kwargs, failing clearly when weights are set but unsupported."""
    if sample_weight is None:
        return {}
    if not has_fit_parameter(estimator, "sample_weight"):
        raise ValidationError(
            f"{type(estimator).__name__} does not accept sample_weight, but a "
            "ColumnRole.WEIGHT column is set. Choose an estimator that supports "
            "sample_weight, or clear the weight role."
        )
    return {"sample_weight": sample_weight.to_numpy(dtype=float)}


def _feature_target_frames(
    dataset: Dataset,
    split_plan: SplitPlan,
    partition: Literal["train", "validation", "test"],
    *,
    sample_rows: int | None = None,
    random_state: int | None = 0,
    materialize_prep: bool = True,
) -> tuple[pd.DataFrame, pd.Series, list[str], str, pd.Series | None]:
    target = dataset.require_target()
    feature_cols = resolve_feature_columns(dataset)
    weight = weight_column(dataset)

    frame = frame_for_partition(dataset, split_plan, partition)
    if not materialize_prep:
        sample_weight = None
        if weight is not None:
            if weight not in frame.columns:
                raise ValidationError(f"Weight column '{weight}' is missing from the dataset")
            sample_weight = validate_sample_weights(frame[weight], column=weight)
        return frame[feature_cols], frame[target], feature_cols, target, sample_weight

    # Engine-aware projection/sampling on the partition slice before sklearn.
    project_cols = [*feature_cols, target]
    if weight is not None:
        if weight not in frame.columns:
            raise ValidationError(f"Weight column '{weight}' is missing from the dataset")
        project_cols.append(weight)
    partition_ds = Dataset.from_pandas(
        frame,
        schema=dataset.schema,
        mode=dataset.mode,
        engine=dataset.engine,
        source=dataset.source,
        roles=dict(dataset.roles),
    )
    prep = prepare_design_frame(
        partition_ds,
        project_cols,
        sample_rows=sample_rows,
        random_state=random_state,
        context=f"estimator {partition} design matrix",
    )
    prepared = prep.frame
    sample_weight = None
    if weight is not None:
        sample_weight = validate_sample_weights(prepared[weight], column=weight)
    return (
        prepared[feature_cols],
        prepared[target],
        feature_cols,
        target,
        sample_weight,
    )


def materialize_partition_design(
    dataset: Dataset,
    split_plan: SplitPlan,
    partition: Literal["train", "validation", "test"] = "train",
    *,
    sample_rows: int | None = None,
    random_state: int | None = 0,
) -> MaterializePrepResult:
    """Project/sample partition columns via the active engine, then materialize."""
    target = dataset.require_target()
    feature_cols = resolve_feature_columns(dataset)
    weight = weight_column(dataset)
    project_cols = [*feature_cols, target]
    if weight is not None:
        project_cols.append(weight)
    frame = frame_for_partition(dataset, split_plan, partition)
    partition_ds = Dataset.from_pandas(
        frame,
        schema=dataset.schema,
        mode=dataset.mode,
        engine=dataset.engine,
        source=dataset.source,
        roles=dict(dataset.roles),
    )
    return prepare_design_frame(
        partition_ds,
        project_cols,
        sample_rows=sample_rows,
        random_state=random_state,
        context=f"estimator {partition} design matrix",
    )


def _infer_task(
    y: pd.Series,
    task: TaskType,
    estimator: Any,
) -> Literal["classification", "regression"]:
    if task != "auto":
        return task
    if isinstance(estimator, ClassifierMixin):
        return "classification"
    if isinstance(estimator, RegressorMixin):
        return "regression"
    if pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) > max(10, int(0.2 * len(y))):
        return "regression"
    return "classification"


def fit_estimator(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    estimator: Any,
    *,
    task: TaskType = "auto",
    sample_rows: int | None = None,
    random_state: int | None = 0,
) -> FitResult:
    """Fit a sklearn-compatible estimator on the train partition only.

    When a ``ColumnRole.WEIGHT`` column is set, it is passed as
    ``sample_weight`` when the estimator accepts it. Estimators that reject
    ``sample_weight`` raise :class:`~buildml.core.errors.ValidationError`
    instead of silently ignoring weights.

    When the dataset engine is Polars or DuckDB, column projection (and optional
    row sampling) runs on the native engine before the Pandas design matrix is
    materialized for sklearn. Sampling is disclosed and does not enable
    out-of-core training.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None

    x_train, y_train, feature_cols, target, sample_weight = _feature_target_frames(
        dataset,
        split_plan,
        "train",
        sample_rows=sample_rows,
        random_state=random_state,
    )
    resolved_task = _infer_task(y_train, task, estimator)
    model = clone(estimator)
    model.fit(x_train, y_train, **fit_kwargs_for_sample_weight(model, sample_weight))
    return FitResult(
        estimator=model,
        task=resolved_task,
        feature_columns=tuple(feature_cols),
        target_column=target,
        n_train_rows=int(len(x_train)),
        weight_column=weight_column(dataset),
    )


def predict_estimator(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    fit_result: FitResult,
    *,
    partition: Literal["train", "validation", "test"] = "test",
    return_proba: bool = False,
) -> pd.Series | pd.DataFrame:
    """Predict labels (and optionally probabilities) on a partition."""
    if split_plan is None:
        raise ValidationError("A split is required for partitioned prediction")
    x, _, _, _, _ = _feature_target_frames(dataset, split_plan, partition)
    missing = [c for c in fit_result.feature_columns if c not in x.columns]
    if missing:
        raise ValidationError(f"Missing feature columns for prediction: {missing}")
    x = x[list(fit_result.feature_columns)]
    preds = fit_result.estimator.predict(x)
    if return_proba and hasattr(fit_result.estimator, "predict_proba"):
        proba = fit_result.estimator.predict_proba(x)
        classes = getattr(fit_result.estimator, "classes_", range(proba.shape[1]))
        columns = [f"proba_{c}" for c in classes]
        return pd.DataFrame(proba, columns=columns, index=x.index)
    return pd.Series(preds, index=x.index, name="prediction")


def evaluate_estimator(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    fit_result: FitResult,
    *,
    partition: Literal["train", "validation", "test"] = "test",
) -> EvaluateResult:
    """Evaluate a fitted estimator on a partition.

    When a weight role is set, metrics that accept ``sample_weight`` use it.
    """
    if split_plan is None:
        raise ValidationError("A split is required for partitioned evaluation")
    x, y_true, _, _, sample_weight = _feature_target_frames(dataset, split_plan, partition)
    x = x[list(fit_result.feature_columns)]
    y_pred = fit_result.estimator.predict(x)
    sw = None if sample_weight is None else sample_weight.to_numpy(dtype=float)
    metrics: dict[str, float] = {}
    diagnostics: dict[str, Any] = {}
    tips: list[str] = []
    if sw is not None:
        diagnostics["sample_weight_column"] = fit_result.weight_column or weight_column(dataset)
        tips.append("Evaluation metrics use ColumnRole.WEIGHT as sample_weight where supported.")

    if fit_result.task == "regression":
        residuals = y_true.to_numpy(dtype=float) - np.asarray(y_pred, dtype=float)
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred, sample_weight=sw))
        metrics["mse"] = float(mean_squared_error(y_true, y_pred, sample_weight=sw))
        metrics["rmse"] = float(np.sqrt(metrics["mse"]))
        metrics["median_ae"] = float(median_absolute_error(y_true, y_pred, sample_weight=sw))
        metrics["r2"] = float(r2_score(y_true, y_pred, sample_weight=sw))
        try:
            metrics["mape"] = float(
                mean_absolute_percentage_error(y_true, y_pred, sample_weight=sw)
            )
        except ValueError:
            tips.append("MAPE unavailable (zeros/near-zeros in target).")
        mean_resid = float(
            np.average(residuals, weights=sw) if sw is not None else np.mean(residuals)
        )
        diagnostics["residual_summary"] = {
            "mean": mean_resid,
            "std": float(np.std(residuals)),
            "q05": float(np.quantile(residuals, 0.05)),
            "q50": float(np.quantile(residuals, 0.50)),
            "q95": float(np.quantile(residuals, 0.95)),
        }
        if metrics["r2"] < 0:
            tips.append("Negative R² — model underperforms a mean baseline on this partition.")
    else:
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred, sample_weight=sw))
        metrics["balanced_accuracy"] = float(
            balanced_accuracy_score(y_true, y_pred, sample_weight=sw)
        )
        metrics["precision_weighted"] = float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0, sample_weight=sw)
        )
        metrics["recall_weighted"] = float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0, sample_weight=sw)
        )
        metrics["f1_weighted"] = float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0, sample_weight=sw)
        )
        metrics["f1_macro"] = float(
            f1_score(y_true, y_pred, average="macro", zero_division=0, sample_weight=sw)
        )
        labels = sorted(
            pd.unique(pd.concat([y_true.astype(str), pd.Series(y_pred).astype(str)]))
        )
        cm = confusion_matrix(
            y_true.astype(str),
            pd.Series(y_pred).astype(str),
            labels=labels,
            sample_weight=sw,
        )
        diagnostics["confusion_matrix"] = {
            "labels": labels,
            "matrix": cm.tolist(),
        }
        report = classification_report(
            y_true.astype(str),
            pd.Series(y_pred).astype(str),
            output_dict=True,
            zero_division=0,
            sample_weight=sw,
        )
        diagnostics["classification_report"] = report
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true.astype(str),
            pd.Series(y_pred).astype(str),
            labels=labels,
            average=None,
            zero_division=0,
            sample_weight=sw,
        )
        diagnostics["per_class"] = {
            str(label): {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(s),
            }
            for label, p, r, f, s in zip(labels, precision, recall, f1, support, strict=True)
        }
        if hasattr(fit_result.estimator, "predict_proba"):
            try:
                proba = fit_result.estimator.predict_proba(x)
                metrics["log_loss"] = float(
                    log_loss(
                        y_true,
                        proba,
                        labels=fit_result.estimator.classes_,
                        sample_weight=sw,
                    )
                )
                if len(fit_result.estimator.classes_) == 2:
                    metrics["roc_auc"] = float(
                        roc_auc_score(y_true, proba[:, 1], sample_weight=sw)
                    )
                    metrics["average_precision"] = float(
                        average_precision_score(y_true, proba[:, 1], sample_weight=sw)
                    )
                else:
                    metrics["roc_auc_ovr_weighted"] = float(
                        roc_auc_score(
                            y_true,
                            proba,
                            multi_class="ovr",
                            average="weighted",
                            sample_weight=sw,
                        )
                    )
            except ValueError as exc:
                tips.append(f"Probability metrics unavailable: {exc}")
        if metrics.get("balanced_accuracy", 1) + 1e-9 < metrics.get("accuracy", 0):
            tips.append("Accuracy ≫ balanced accuracy — inspect class imbalance / majority bias.")

    if not tips:
        tips.append(
            "No urgent evaluation warnings — compare against baselines and other estimators."
        )

    return EvaluateResult(
        partition=partition,
        task=fit_result.task,
        metrics=metrics,
        diagnostics=diagnostics,
        n_rows=int(len(y_true)),
        recommendations=tips,
    )
