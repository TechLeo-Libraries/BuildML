"""Train-fitted dimensionality reduction (PCA) with explained-variance reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
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

ReduceMethod = Literal["pca"]


@dataclass(slots=True)
class ReducePlan:
    """Train-fitted dimensionality-reduction plan."""

    columns: tuple[str, ...]
    method: ReduceMethod
    n_components: int
    feature_names_: tuple[str, ...]
    explained_variance_ratio_: tuple[float, ...]
    cumulative_explained_variance_: tuple[float, ...]
    reducer_: Any = field(repr=False)
    drop_input_columns: bool = True
    prefix: str = "pc"

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "method": self.method,
            "n_components": self.n_components,
            "feature_names_": list(self.feature_names_),
            "explained_variance_ratio_": list(self.explained_variance_ratio_),
            "cumulative_explained_variance_": list(self.cumulative_explained_variance_),
            "drop_input_columns": self.drop_input_columns,
            "prefix": self.prefix,
            "total_explained_variance": (
                float(self.cumulative_explained_variance_[-1])
                if self.cumulative_explained_variance_
                else 0.0
            ),
        }


def fit_reducer(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    columns: list[str] | None = None,
    method: ReduceMethod = "pca",
    n_components: int | float | None = None,
    drop_input_columns: bool = True,
    prefix: str = "pc",
) -> ReducePlan:
    """Fit a dimensionality reducer on the train partition only.

    Parameters
    ----------
    n_components:
        Integer component count, a float in (0, 1] for variance target (PCA),
        or ``None`` to keep ``min(n_samples, n_features)`` components capped at
        the feature width.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None
    if method != "pca":
        raise ValidationError(f"Unsupported reduce method '{method}'")
    if not prefix or not str(prefix).replace("_", "").isalnum():
        raise ValidationError("prefix must be a non-empty alphanumeric token")

    train = frame_for_partition(dataset, split_plan, "train")
    cols = _resolve_numeric_columns(dataset, train, columns)
    x = train[list(cols)]
    if x.isna().any().any():
        raise ValidationError(
            "Dimensionality reduction requires non-null train features. "
            "Call session.impute(...) first."
        )
    n_samples, n_features = x.shape
    max_components = min(n_samples, n_features)
    if max_components < 1:
        raise ValidationError("Not enough train rows/columns for PCA")

    pca_n: int | float
    if n_components is None:
        pca_n = max_components
    elif isinstance(n_components, float):
        if not (0.0 < n_components <= 1.0):
            raise ValidationError("Float n_components must be in (0, 1] (variance target)")
        pca_n = n_components
    else:
        if int(n_components) < 1:
            raise ValidationError("Integer n_components must be >= 1")
        pca_n = min(int(n_components), max_components)

    reducer = PCA(n_components=pca_n, svd_solver="full")
    reducer.fit(x.to_numpy(dtype=float))
    ratios = tuple(float(v) for v in np.asarray(reducer.explained_variance_ratio_, dtype=float))
    cumulative = tuple(float(v) for v in np.cumsum(ratios))
    n_out = len(ratios)
    names = tuple(f"{prefix}_{i + 1}" for i in range(n_out))
    return ReducePlan(
        columns=tuple(cols),
        method="pca",
        n_components=n_out,
        feature_names_=names,
        explained_variance_ratio_=ratios,
        cumulative_explained_variance_=cumulative,
        reducer_=reducer,
        drop_input_columns=drop_input_columns,
        prefix=prefix,
    )


def transform_reducer(
    dataset: Dataset,
    plan: ReducePlan,
) -> tuple[Dataset, PreprocessResult]:
    """Apply a train-fitted reduction plan to the full dataset."""
    missing = [c for c in plan.columns if c not in dataset.columns]
    if missing:
        raise ValidationError(f"Reduce plan columns missing from dataset: {missing}")

    frame = dataset._ensure_pandas().copy()
    values = frame[list(plan.columns)].to_numpy(dtype=float)
    if np.isnan(values).any():
        raise ValidationError(
            "Dimensionality reduction transform found nulls. Impute before reduce_dimensions."
        )
    transformed = plan.reducer_.transform(values)
    component_frame = pd.DataFrame(
        transformed,
        columns=list(plan.feature_names_),
        index=frame.index,
    )
    roles = dict(dataset.roles)
    for column in plan.columns:
        roles.pop(column, None)
    for name in plan.feature_names_:
        roles[name] = ColumnRole.FEATURE

    if plan.drop_input_columns:
        frame = frame.drop(columns=list(plan.columns))
    out = pd.concat([frame, component_frame], axis=1)
    new_dataset = Dataset.from_transformed(
        dataset,
        out,
        schema=schema_from_dataframe(out),
        roles=roles,
    )
    return new_dataset, _build_result(plan)


def _resolve_numeric_columns(
    dataset: Dataset,
    train: pd.DataFrame,
    columns: list[str] | None,
) -> list[str]:
    protected = {
        ColumnRole.TARGET,
        ColumnRole.ID,
        ColumnRole.GROUP,
        ColumnRole.TIME,
        ColumnRole.WEIGHT,
    }
    if columns is not None:
        names = validate_column_names(columns, dataset.columns)
        names = [name for name in names if dataset.roles.get(name) not in protected]
    else:
        feature_roles = dataset.role_columns(ColumnRole.FEATURE)
        candidates = feature_roles or [
            str(c) for c in train.columns if dataset.roles.get(str(c)) not in protected
        ]
        names = [
            str(c)
            for c in candidates
            if c in train.columns and pd.api.types.is_numeric_dtype(train[c])
        ]
    if not names:
        raise ValidationError("No numeric columns available for dimensionality reduction")
    non_numeric = [c for c in names if not pd.api.types.is_numeric_dtype(train[c])]
    if non_numeric:
        raise ValidationError(
            "Dimensionality reduction requires numeric columns; "
            f"encode/scale first. Non-numeric: {non_numeric[:12]}"
        )
    return names


def _build_result(plan: ReducePlan) -> PreprocessResult:
    total = (
        float(plan.cumulative_explained_variance_[-1])
        if plan.cumulative_explained_variance_
        else 0.0
    )
    evidence = [
        Evidence(
            key="reduce_dimensions.explained_variance",
            kind=EvidenceKind.METRIC,
            summary="Train-fitted PCA explained-variance ratios.",
            value={
                "method": plan.method,
                "n_components": plan.n_components,
                "explained_variance_ratio": list(plan.explained_variance_ratio_),
                "cumulative_explained_variance": list(plan.cumulative_explained_variance_),
                "total_explained_variance": total,
                "source_columns": list(plan.columns),
            },
            source="train.pca",
            limitations=(
                "Explained variance is unsupervised; it is not predictive utility.",
            ),
        )
    ]
    findings = [
        Finding(
            key="reduce_dimensions.applied",
            title="PCA components fitted on train",
            detail=(
                f"Replaced {len(plan.columns)} numeric column(s) with "
                f"{plan.n_components} component(s) capturing {total:.1%} of "
                "train variance among those columns."
            ),
            severity=FindingSeverity.INFO,
            evidence=tuple(evidence),
            affected_columns=plan.columns,
        )
    ]
    recommendations = [
        Recommendation(
            key="reduce_dimensions.scale-first",
            title="Confirm scaling before interpreting PCA variance shares",
            rationale=(
                "Unscaled columns with large magnitudes dominate components. "
                "Compare holdout metrics with and without reduction before keeping it."
            ),
            priority=ActionPriority.NEXT,
            action=Action(
                key="reduce_dimensions.eval-action",
                label="Session.evaluate(partition='validation')",
                operation="evaluate",
                parameters={"partition": "validation"},
            ),
            based_on=("reduce_dimensions.applied",),
            caveats=("Variance explained is not a substitute for supervised selection.",),
        )
    ]
    return PreprocessResult(
        operation="reduce_dimensions",
        plan=plan.to_dict(),
        evidence=evidence,
        findings=findings,
        interpretation=[
            f"PCA kept {plan.n_components} component(s) from {len(plan.columns)} column(s).",
            f"Cumulative train variance explained: {total:.1%}.",
        ],
        limitations=[
            "PCA is fit on train only; holdout rows use the frozen rotation.",
            "Explained variance does not guarantee better predictive metrics.",
            "Components are linear mixes; interpret loadings before domain claims.",
        ],
        recommendations=recommendations,
        methods=[
            "PCA (sklearn, svd_solver='full') fitted on train numeric columns.",
            f"Output columns: {', '.join(plan.feature_names_[:8])}"
            + ("…" if len(plan.feature_names_) > 8 else ""),
        ],
        warnings=[],
    )
