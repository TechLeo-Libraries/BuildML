"""Adaptive visualization planning from data characteristics."""

from __future__ import annotations

from typing import Any

import pandas as pd

from buildml.core.types import ColumnRole
from buildml.data.dataset import Dataset


def build_adaptive_plan(
    dataset: Dataset,
    frame: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    max_plots: int = 24,
) -> list[dict[str, Any]]:
    """Choose high-impact plot specs from dtypes, cardinality, roles, and scale.

    The planner prefers revealing charts over decorative ones and caps volume
    so large schemas remain usable.
    """
    plan: list[dict[str, Any]] = []
    invalid_feature_roles = {
        ColumnRole.TARGET,
        ColumnRole.ID,
        ColumnRole.IGNORE,
        ColumnRole.GROUP,
        ColumnRole.TIME,
        ColumnRole.WEIGHT,
    }
    candidates = feature_columns if feature_columns is not None else list(frame.columns.astype(str))
    eligible = [
        str(column)
        for column in candidates
        if column in frame.columns
        and dataset.roles.get(str(column)) not in invalid_feature_roles
        and frame[column].nunique(dropna=False) > 1
    ]
    numeric = [
        column for column in eligible if pd.api.types.is_numeric_dtype(frame[column])
    ]
    categorical = [
        str(c)
        for c in eligible
        if c not in numeric
        and (
            pd.api.types.is_object_dtype(frame[c])
            or isinstance(frame[c].dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(frame[c])
            or pd.api.types.is_bool_dtype(frame[c])
        )
    ]
    datetime_cols = list(
        frame.select_dtypes(include=["datetime", "datetimetz"]).columns.astype(str)
    )
    target_cols = dataset.role_columns(ColumnRole.TARGET)
    target = target_cols[0] if target_cols else None

    plan.append({"kind": "missingness_matrix", "title": "Missingness map", "priority": 100})
    plan.append({"kind": "dtype_overview", "title": "Schema / dtype mosaic", "priority": 95})

    if numeric:
        plan.append(
            {
                "kind": "correlation_heatmap",
                "title": "Numeric association heatmap",
                "columns": numeric[:40],
                "priority": 90,
            }
        )
        for col in numeric[:8]:
            plan.append(
                {
                    "kind": "numeric_distribution",
                    "title": f"Distribution · {col}",
                    "column": col,
                    "priority": 80,
                }
            )
        if len(numeric) >= 2:
            plan.append(
                {
                    "kind": "pair_sample",
                    "title": "Top numeric relationships",
                    "columns": numeric[:5],
                    "priority": 75,
                }
            )

    for col in categorical[:8]:
        nunique = int(frame[col].nunique(dropna=True))
        kind = "categorical_bars" if nunique <= 30 else "categorical_topk"
        plan.append(
            {
                "kind": kind,
                "title": f"Category profile · {col}",
                "column": col,
                "priority": 70,
            }
        )

    if target is not None:
        plan.append(
            {
                "kind": "target_balance",
                "title": f"Target balance · {target}",
                "column": target,
                "priority": 92,
            }
        )
        for col in numeric[:6]:
            if col == target:
                continue
            plan.append(
                {
                    "kind": "target_vs_numeric",
                    "title": f"{col} by {target}",
                    "feature": col,
                    "target": target,
                    "priority": 85,
                }
            )
        for col in categorical[:5]:
            if col == target:
                continue
            plan.append(
                {
                    "kind": "target_vs_categorical",
                    "title": f"{target} vs {col}",
                    "feature": col,
                    "target": target,
                    "priority": 84,
                }
            )

    for col in datetime_cols[:3]:
        plan.append(
            {
                "kind": "temporal_density",
                "title": f"Temporal density · {col}",
                "column": col,
                "priority": 65,
            }
        )

    plan.append({"kind": "outlier_board", "title": "Outlier board (IQR)", "priority": 60})

    plan.sort(key=lambda item: int(item.get("priority", 0)), reverse=True)
    return plan[:max_plots]
