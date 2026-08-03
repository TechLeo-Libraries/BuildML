"""Numeric feature resolution for anomaly Session ops (PCA-aware)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.core.validation import validate_column_names
from buildml.data.dataset import Dataset


def resolve_anomaly_columns(
    dataset: Dataset,
    frame: pd.DataFrame,
    columns: list[str] | None,
    *,
    reduce_plan: Any | None = None,
    prefer_reduce_components: bool = True,
    extra_exclude: set[str] | None = None,
) -> tuple[list[str], bool, list[str]]:
    """Resolve numeric feature columns for anomaly detection.

When ``prefer_reduce_components`` is True and a Session ``ReducePlan`` is
present with component columns still on the frame, those components are
preferred — same integration contract as clustering (no forked PCA).
Protected roles (target/id/group/time/weight) and any ``extra_exclude``
names (e.g. normal-label columns) are never used as features.

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
extra_exclude:
    extra exclude (set[str] | None).

Returns
-------
tuple[list[str], bool, list[str]]
    Tuple of results (tuple[list[str], bool, list[str]]) for downstream Session steps.

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
    exclude = set(extra_exclude or ())

    if columns is not None:
        names = validate_column_names(columns, dataset.columns)
        names = [
            name
            for name in names
            if dataset.roles.get(name) not in protected and name not in exclude
        ]
        if not names:
            raise ValidationError(
                "No usable columns after excluding protected roles "
                "(target/id/group/time/weight) and label columns."
            )
        _assert_numeric(frame, names)
        return names, False, disclosures

    if prefer_reduce_components and reduce_plan is not None:
        feature_names = getattr(reduce_plan, "feature_names_", None) or ()
        present = [
            str(c)
            for c in feature_names
            if str(c) in frame.columns and str(c) not in exclude
        ]
        if present:
            _assert_numeric(frame, present)
            disclosures.append(
                "Used Session.reduce_dimensions component columns for anomaly "
                f"detection ({len(present)} component(s)). PCA rotation remains a "
                "separate train-fitted preprocess plan; anomaly fit does not refit PCA."
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
        if c in frame.columns
        and c not in exclude
        and pd.api.types.is_numeric_dtype(frame[c])
    ]
    if not names:
        raise ValidationError(
            "No numeric columns available for anomaly detection. "
            "Encode/scale first, or call reduce_dimensions and score on components."
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
            "Anomaly detection requires non-null features. Call session.impute(...) "
            "first (and typically session.scale(...) before distance-based methods)."
        )
    return block.to_numpy(dtype=float)


def _assert_numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(frame[c])]
    if non_numeric:
        raise ValidationError(
            "Anomaly detection requires numeric columns; encode/scale first. "
            f"Non-numeric: {non_numeric[:12]}"
        )
