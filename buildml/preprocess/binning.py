"""Numeric binning with train-fitted edges."""

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

BinStrategy = Literal["quantile", "uniform"]


@dataclass(slots=True)
class BinningPlan:
    """Train-fitted discretization edges per column."""

    columns: tuple[str, ...]
    strategy: BinStrategy
    n_bins: int
    edges_: dict[str, list[float]]
    labels_: dict[str, list[str]]
    encode_as: Literal["ordinal", "onehot"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "strategy": self.strategy,
            "n_bins": self.n_bins,
            "edges_": {key: list(values) for key, values in self.edges_.items()},
            "labels_": {key: list(values) for key, values in self.labels_.items()},
            "encode_as": self.encode_as,
        }


def fit_binning(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    columns: list[str] | None = None,
    strategy: BinStrategy = "quantile",
    n_bins: int = 5,
    encode_as: Literal["ordinal", "onehot"] = "ordinal",
) -> BinningPlan:
    """Learn bin edges on the train partition only."""
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None
    if n_bins < 2:
        raise ValidationError("n_bins must be at least 2")
    if strategy not in {"quantile", "uniform"}:
        raise ValidationError(f"Unsupported binning strategy '{strategy}'")
    if encode_as not in {"ordinal", "onehot"}:
        raise ValidationError(f"Unsupported encode_as '{encode_as}'")

    train = frame_for_partition(dataset, split_plan, "train")
    cols = _resolve_numeric_columns(dataset, train, columns)
    edges: dict[str, list[float]] = {}
    labels: dict[str, list[str]] = {}
    for column in cols:
        series = pd.to_numeric(train[column], errors="coerce").dropna()
        if series.empty:
            raise ValidationError(f"Column '{column}' has no finite train values for binning")
        unique = series.nunique(dropna=True)
        bins = min(n_bins, int(unique)) if unique >= 2 else 2
        if strategy == "quantile":
            quantiles = np.linspace(0.0, 1.0, bins + 1)
            raw_edges = np.unique(np.quantile(series.to_numpy(), quantiles))
        else:
            raw_edges = np.linspace(float(series.min()), float(series.max()), bins + 1)
            raw_edges = np.unique(raw_edges)
        if len(raw_edges) < 3:
            # Constant or near-constant column: create a degenerate two-edge span.
            center = float(series.iloc[0])
            raw_edges = np.array([center - 0.5, center + 0.5], dtype=float)
        # Ensure open-ended coverage for scoring extremes.
        raw_edges = raw_edges.astype(float)
        raw_edges[0] = float("-inf")
        raw_edges[-1] = float("inf")
        edge_list = [float(v) for v in raw_edges]
        edges[column] = edge_list
        labels[column] = [f"{column}_bin_{i}" for i in range(len(edge_list) - 1)]

    return BinningPlan(
        columns=tuple(cols),
        strategy=strategy,
        n_bins=n_bins,
        edges_=edges,
        labels_=labels,
        encode_as=encode_as,
    )


def transform_binning(dataset: Dataset, plan: BinningPlan) -> tuple[Dataset, PreprocessResult]:
    """Apply train-fitted bin edges to the full dataset."""
    missing = [c for c in plan.columns if c not in dataset.columns]
    if missing:
        raise ValidationError(f"Binning plan columns missing from dataset: {missing}")

    frame = dataset._ensure_pandas().copy()
    roles = dict(dataset.roles)
    from buildml.core.types import ColumnRole

    for column in plan.columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        edges = plan.edges_[column]
        codes = pd.cut(
            values,
            bins=edges,
            labels=False,
            include_lowest=True,
            right=True,
        )
        # NaN inputs stay NaN; out-of-edge should not occur with ±inf ends.
        if plan.encode_as == "ordinal":
            out_name = f"{column}_bin"
            frame[out_name] = codes.astype("float")
            roles[out_name] = roles.get(column, ColumnRole.FEATURE)
        else:
            n_levels = len(edges) - 1
            for level in range(n_levels):
                out_name = plan.labels_[column][level]
                frame[out_name] = (codes == level).astype("float")
                roles[out_name] = ColumnRole.FEATURE
        del frame[column]
        roles.pop(column, None)

    new_dataset = Dataset.from_transformed(
        dataset,
        frame,
        schema=schema_from_dataframe(frame),
        roles=roles,
    )
    return new_dataset, _build_result(plan)


def _resolve_numeric_columns(
    dataset: Dataset,
    train: pd.DataFrame,
    columns: list[str] | None,
) -> list[str]:
    if columns is not None:
        names = validate_column_names(columns, dataset.columns)
        for name in names:
            if not pd.api.types.is_numeric_dtype(train[name]):
                raise ValidationError(f"Binning requires numeric columns; '{name}' is not numeric")
        return names
    target_cols = set(dataset.role_columns("target"))
    numeric = [
        str(c)
        for c in train.columns
        if c not in target_cols and pd.api.types.is_numeric_dtype(train[c])
    ]
    if not numeric:
        raise ValidationError("No numeric columns available for binning")
    return numeric


def _build_result(plan: BinningPlan) -> PreprocessResult:
    evidence = [
        Evidence(
            key="binning.edges",
            kind=EvidenceKind.CONFIGURATION,
            summary="Train-fitted discretization edges per column.",
            value={"columns": list(plan.columns), "strategy": plan.strategy, "n_bins": plan.n_bins},
            source="train.bin_edges",
            limitations=(
                "Edges depend on train support; rare score-time extremes fall into end bins.",
            ),
        )
    ]
    findings = [
        Finding(
            key="binning.applied",
            title="Numeric features discretized",
            detail=(
                f"Strategy '{plan.strategy}' with requested n_bins={plan.n_bins} "
                f"produced train-fitted edges for {len(plan.columns)} column(s)."
            ),
            severity=FindingSeverity.INFO,
            evidence=tuple(evidence),
            affected_columns=plan.columns,
        )
    ]
    recommendations = [
        Recommendation(
            key="binning.review-cardinality",
            title="Review whether ordinal bins match the estimator family",
            rationale=(
                "Tree models often prefer raw numeric values; linear models may benefit from "
                "monotonic bins when relationships are stepwise."
            ),
            priority=ActionPriority.OPTIONAL,
            action=Action(
                key="binning.review-action",
                label="Session.explain('bin')",
                operation="explain",
                parameters={"operation": "bin"},
            ),
            based_on=("binning.applied",),
            caveats=("Discretization discards within-bin magnitude.",),
        )
    ]
    return PreprocessResult(
        operation="bin",
        plan=plan.to_dict(),
        evidence=evidence,
        findings=findings,
        interpretation=[
            f"Replaced {len(plan.columns)} numeric column(s) with {plan.encode_as} bin codes.",
            "Edges were learned on train only and frozen for all partitions.",
        ],
        limitations=[
            "Quantile edges can collapse when train support is sparse or discrete.",
            "Binning is irreversible information loss within each interval.",
            "Do not refit edges on validation or test rows.",
        ],
        recommendations=recommendations,
        methods=[
            f"Train-only {plan.strategy} edges; encode_as={plan.encode_as}.",
            "End bins use open ±inf edges so score-time extremes remain defined.",
        ],
    )
