"""Numeric feature resolution for unsupervised Session ops (PCA-aware)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.core.validation import validate_column_names
from buildml.data.dataset import Dataset


def resolve_cluster_columns(
    dataset: Dataset,
    frame: pd.DataFrame,
    columns: list[str] | None,
    *,
    reduce_plan: Any | None = None,
    prefer_reduce_components: bool = True,
) -> tuple[list[str], bool, list[str]]:
    """Resolve numeric feature columns for clustering.

When ``prefer_reduce_components`` is True and a Session ``ReducePlan`` is
present with component columns still on the frame, those components are
preferred. This integrates with ``Session.reduce_dimensions`` rather than
forking a second PCA path.

Parameters
----------
dataset:
    BuildML dataset with features, target, and role metadata.
frame:
    Partition or full DataFrame slice used for this operation.
columns:
    Optional explicit feature column list; ``None`` auto-selects numerics.
reduce_plan:
    Optional preprocess reduce plan from Session.
prefer_reduce_components:
    Prefer reduced component columns when a reduce plan exists.

Returns
-------
tuple[list[str], bool, list[str]]
    Selected columns, whether reduce components were used, and disclosures.

Raises
------
ValidationError
    When preconditions for this operation are not met.
    """
    disclosures: list[str] = []
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
        if not names:
            raise ValidationError(
                "No usable columns after excluding protected roles "
                "(target/id/group/time/weight)."
            )
        _assert_numeric(frame, names)
        return names, False, disclosures

    if prefer_reduce_components and reduce_plan is not None:
        feature_names = getattr(reduce_plan, "feature_names_", None) or ()
        present = [str(c) for c in feature_names if str(c) in frame.columns]
        if present:
            _assert_numeric(frame, present)
            disclosures.append(
                "Used Session.reduce_dimensions component columns for clustering "
                f"({len(present)} component(s)). PCA rotation remains a separate "
                "train-fitted preprocess plan; clustering does not refit PCA."
            )
            return present, True, disclosures
        if feature_names:
            disclosures.append(
                "ReducePlan is attached but its component columns are missing from "
                "the frame; falling back to numeric feature roles."
            )

    feature_roles = dataset.role_columns(ColumnRole.FEATURE)
    candidates = feature_roles or [
        str(c) for c in frame.columns if dataset.roles.get(str(c)) not in protected
    ]
    names = [
        str(c)
        for c in candidates
        if c in frame.columns and pd.api.types.is_numeric_dtype(frame[c])
    ]
    if not names:
        raise ValidationError(
            "No numeric columns available for clustering. "
            "Encode/scale first, or call reduce_dimensions and cluster on components."
        )
    return names, False, disclosures


def matrix_from_frame(frame: pd.DataFrame, columns: list[str]) -> Any:
    """Build a float design matrix; refuse nulls with a precise message.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
frame:
    Partition or full DataFrame slice used for this operation.
columns:
    Optional explicit feature column list; ``None`` auto-selects numerics.

Returns
-------
Any
    Adapter-specific estimator or model object.

Raises
------
ValidationError
    When preconditions for this operation are not met.
    """
    import numpy as np

    block = frame[list(columns)]
    if block.isna().any().any():
        raise ValidationError(
            "Clustering requires non-null features. Call session.impute(...) first "
            "(and typically session.scale(...) before distance-based methods)."
        )
    return block.to_numpy(dtype=float)


def _assert_numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(frame[c])]
    if non_numeric:
        raise ValidationError(
            "Clustering requires numeric columns; encode/scale first. "
            f"Non-numeric: {non_numeric[:12]}"
        )
