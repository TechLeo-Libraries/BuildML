"""Train-fitted outlier handling with detect / cap / drop actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.core.validation import validate_column_names
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition, frame_for_partition
from buildml.explain.schemas import (
    Action,
    ActionPriority,
    Evidence,
    EvidenceKind,
    Finding,
    FindingSeverity,
    Recommendation,
)
from buildml.ingest.detect import schema_from_dataframe
from buildml.preprocess.result import PreprocessResult

OutlierMethod = Literal["iqr", "zscore"]
OutlierAction = Literal["detect", "cap", "drop"]


@dataclass(slots=True)
class OutlierPlan:
    """Train-fitted outlier fences and chosen action."""

    columns: tuple[str, ...]
    method: OutlierMethod
    action: OutlierAction
    lower_: dict[str, float]
    upper_: dict[str, float]
    n_flagged_train: int
    n_dropped: int
    iqr_multiplier: float
    zscore_threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "method": self.method,
            "action": self.action,
            "lower_": dict(self.lower_),
            "upper_": dict(self.upper_),
            "n_flagged_train": self.n_flagged_train,
            "n_dropped": self.n_dropped,
            "iqr_multiplier": self.iqr_multiplier,
            "zscore_threshold": self.zscore_threshold,
        }


def fit_outlier_plan(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    columns: list[str] | None = None,
    method: OutlierMethod = "iqr",
    action: OutlierAction = "cap",
    iqr_multiplier: float = 1.5,
    zscore_threshold: float = 3.0,
) -> OutlierPlan:
    """Learn outlier fences on the train partition only."""
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None
    if method not in {"iqr", "zscore"}:
        raise ValidationError(f"Unsupported outlier method '{method}'")
    if action not in {"detect", "cap", "drop"}:
        raise ValidationError(f"Unsupported outlier action '{action}'")
    if iqr_multiplier <= 0:
        raise ValidationError("iqr_multiplier must be positive")
    if zscore_threshold <= 0:
        raise ValidationError("zscore_threshold must be positive")

    train = frame_for_partition(dataset, split_plan, "train")
    cols = _resolve_numeric_columns(dataset, train, columns)
    lower: dict[str, float] = {}
    upper: dict[str, float] = {}
    for column in cols:
        series = pd.to_numeric(train[column], errors="coerce").dropna()
        if series.empty:
            raise ValidationError(
                f"Column '{column}' has no finite train values for outlier fences"
            )
        if method == "iqr":
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            lower[column] = q1 - iqr_multiplier * iqr
            upper[column] = q3 + iqr_multiplier * iqr
        else:
            mean = float(series.mean())
            std = float(series.std(ddof=0))
            if std == 0.0:
                lower[column] = mean
                upper[column] = mean
            else:
                lower[column] = mean - zscore_threshold * std
                upper[column] = mean + zscore_threshold * std

    mask = _flag_mask(train, cols, lower, upper)
    return OutlierPlan(
        columns=tuple(cols),
        method=method,
        action=action,
        lower_=lower,
        upper_=upper,
        n_flagged_train=int(mask.sum()),
        n_dropped=0,
        iqr_multiplier=float(iqr_multiplier),
        zscore_threshold=float(zscore_threshold),
    )


def apply_outlier_plan(
    dataset: Dataset,
    split_plan: SplitPlan,
    plan: OutlierPlan,
) -> tuple[Dataset, SplitPlan, OutlierPlan, PreprocessResult]:
    """Apply a fitted outlier plan; rebuild split membership when rows are dropped."""
    missing = [c for c in plan.columns if c not in dataset.columns]
    if missing:
        raise ValidationError(f"Outlier plan columns missing from dataset: {missing}")

    if plan.action == "detect":
        result = _build_result(plan, mutated=False)
        return dataset, split_plan, plan, result

    frame = dataset._ensure_pandas().copy()
    if plan.action == "cap":
        for column in plan.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            capped = values.clip(lower=plan.lower_[column], upper=plan.upper_[column])
            frame[column] = capped
        new_dataset = Dataset.from_transformed(
            dataset,
            frame,
            schema=schema_from_dataframe(frame),
        )
        updated = OutlierPlan(
            columns=plan.columns,
            method=plan.method,
            action=plan.action,
            lower_=dict(plan.lower_),
            upper_=dict(plan.upper_),
            n_flagged_train=plan.n_flagged_train,
            n_dropped=0,
            iqr_multiplier=plan.iqr_multiplier,
            zscore_threshold=plan.zscore_threshold,
        )
        return new_dataset, split_plan, updated, _build_result(updated, mutated=True)

    # drop: remove flagged rows using train-learned fences; rebuild partitions.
    keep_mask = ~_flag_mask(frame, list(plan.columns), plan.lower_, plan.upper_)
    kept_positions = np.flatnonzero(keep_mask.to_numpy())
    old_to_new = {int(old): int(new) for new, old in enumerate(kept_positions)}

    def _remap(indices: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(old_to_new[i] for i in indices if i in old_to_new)

    new_split = SplitPlan(
        kind=f"outlier_drop_{split_plan.kind}",
        test_size=split_plan.test_size,
        validation_size=split_plan.validation_size,
        random_state=split_plan.random_state,
        stratify_column=split_plan.stratify_column,
        train_indices=_remap(split_plan.train_indices),
        validation_indices=_remap(split_plan.validation_indices),
        test_indices=_remap(split_plan.test_indices),
    )
    if not new_split.train_indices or not new_split.test_indices:
        raise ValidationError(
            "Outlier drop removed an entire train or test partition. "
            "Widen fences, switch to action='cap', or review columns."
        )
    new_split.assert_disjoint()

    new_frame = frame.iloc[list(kept_positions)].reset_index(drop=True)
    new_dataset = Dataset.from_transformed(
        dataset,
        new_frame,
        schema=schema_from_dataframe(new_frame),
    )
    n_dropped = int(len(frame) - len(new_frame))
    updated = OutlierPlan(
        columns=plan.columns,
        method=plan.method,
        action=plan.action,
        lower_=dict(plan.lower_),
        upper_=dict(plan.upper_),
        n_flagged_train=plan.n_flagged_train,
        n_dropped=n_dropped,
        iqr_multiplier=plan.iqr_multiplier,
        zscore_threshold=plan.zscore_threshold,
    )
    return new_dataset, new_split, updated, _build_result(updated, mutated=True)


def _flag_mask(
    frame: pd.DataFrame,
    columns: list[str],
    lower: dict[str, float],
    upper: dict[str, float],
) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        mask = mask | (values < lower[column]) | (values > upper[column])
    return mask


def _resolve_numeric_columns(
    dataset: Dataset,
    train: pd.DataFrame,
    columns: list[str] | None,
) -> list[str]:
    if columns is not None:
        names = validate_column_names(columns, dataset.columns)
        for name in names:
            if not pd.api.types.is_numeric_dtype(train[name]):
                raise ValidationError(
                    f"Outlier handling requires numeric columns; '{name}' is not numeric"
                )
        return names
    target_cols = set(dataset.role_columns("target"))
    numeric = [
        str(c)
        for c in train.columns
        if c not in target_cols and pd.api.types.is_numeric_dtype(train[c])
    ]
    if not numeric:
        raise ValidationError("No numeric columns available for outlier handling")
    return numeric


def _build_result(plan: OutlierPlan, *, mutated: bool) -> PreprocessResult:
    evidence = [
        Evidence(
            key="outlier.train_flagged",
            kind=EvidenceKind.METRIC,
            summary="Train rows outside train-fitted fences.",
            value={"n_flagged_train": plan.n_flagged_train, "method": plan.method},
            source="train.outlier_fences",
            limitations=(
                "Fence rules are heuristic screens, not proof of error or contamination.",
            ),
        )
    ]
    if plan.action == "drop":
        evidence.append(
            Evidence(
                key="outlier.dropped",
                kind=EvidenceKind.METRIC,
                summary="Rows removed after applying train-fitted fences.",
                value={"n_dropped": plan.n_dropped},
                source="dataset.outlier_drop",
                limitations=("Dropped holdout rows change evaluation support.",),
            )
        )
    severity = FindingSeverity.MEDIUM if plan.n_flagged_train > 0 else FindingSeverity.INFO
    findings = [
        Finding(
            key="outlier.screen",
            title="Train-fitted outlier screen",
            detail=(
                f"Method '{plan.method}' flagged {plan.n_flagged_train} train row(s); "
                f"action='{plan.action}'."
            ),
            severity=severity,
            evidence=tuple(evidence),
            affected_columns=plan.columns,
        )
    ]
    recommendations: list[Recommendation] = []
    if plan.action == "detect" and plan.n_flagged_train > 0:
        recommendations.append(
            Recommendation(
                key="outlier.consider-cap",
                title="Consider capping instead of silent deletion",
                rationale=(
                    "Capping preserves row membership while limiting extreme magnitudes "
                    "using the same train-fitted fences."
                ),
                priority=ActionPriority.NEXT,
                action=Action(
                    key="outlier.consider-cap-action",
                    label="Session.handle_outliers(action='cap')",
                    operation="handle_outliers",
                    parameters={"action": "cap", "method": plan.method},
                ),
                based_on=("outlier.screen",),
                caveats=(
                    "Capping assumes extremes are measurement noise rather than rare valid events.",
                ),
            )
        )
    interpretation = [
        (f"Fences were learned on train with method '{plan.method}' and action '{plan.action}'."),
        (
            "Dataset values were left unchanged."
            if not mutated
            else (
                f"Applied '{plan.action}' using frozen train fences"
                + (f"; dropped {plan.n_dropped} row(s)." if plan.n_dropped else ".")
            )
        ),
    ]
    limitations = [
        "IQR and z-score fences assume roughly unimodal numeric features.",
        "Flagged points may be valid rare events; domain review remains required.",
        "Fences must not be re-fit on validation or test rows.",
    ]
    methods = [
        f"Train-only {plan.method} fences; action={plan.action}.",
        (
            f"IQR multiplier={plan.iqr_multiplier}."
            if plan.method == "iqr"
            else f"Z-score threshold={plan.zscore_threshold}."
        ),
    ]
    return PreprocessResult(
        operation="handle_outliers",
        plan=plan.to_dict(),
        evidence=evidence,
        findings=findings,
        interpretation=interpretation,
        limitations=limitations,
        recommendations=recommendations,
        methods=methods,
    )
