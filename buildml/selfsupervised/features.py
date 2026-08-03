"""Feature helpers for self-supervised Session ops."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.core.validation import validate_column_names
from buildml.data.dataset import Dataset


def resolve_ssl_columns(
    dataset: Dataset,
    frame: pd.DataFrame,
    columns: list[str] | None,
    *,
    reduce_plan: Any | None = None,
    prefer_reduce_components: bool = True,
) -> tuple[list[str], bool, list[str]]:
    """Resolve numeric feature columns for SSL pretext training.

    Excludes protected roles (target, id, group, time, weight) and prefers
    reduce-plan component columns when available.

    Parameters
    ----------
    dataset:
        Session dataset supplying column roles.
    frame:
        Partition frame used to validate column presence and dtype.
    columns:
        Explicit feature columns; when ``None``, roles and reduce plan are used.
    reduce_plan:
        Optional PCA/reduce plan whose component columns may be preferred.
    prefer_reduce_components:
        When True, use reduce-plan components before role-based resolution.

    Returns
    -------
    tuple[list[str], bool, list[str]]
        Resolved column names, whether reduce components were used, and disclosures.

    Raises
    ------
    ValidationError
        When no usable numeric columns remain after filtering.
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
                "No usable columns after excluding protected roles (target/id/group/time/weight)."
            )
        _assert_numeric(frame, names)
        return names, False, disclosures

    if prefer_reduce_components and reduce_plan is not None:
        feature_names = getattr(reduce_plan, "feature_names_", None) or ()
        present = [str(c) for c in feature_names if str(c) in frame.columns]
        if present:
            _assert_numeric(frame, present)
            disclosures.append(
                "Used Session.reduce_dimensions component columns for SSL pretext "
                f"({len(present)} component(s))."
            )
            return present, True, disclosures

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
            "No numeric columns available for self-supervised pretext. "
            "Encode/scale first, or call reduce_dimensions."
        )
    return names, False, disclosures


def matrix_from_frame(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Build a float design matrix from selected frame columns.

    Refuses null values because SSL pretext expects complete numeric features.

    Parameters
    ----------
    frame:
        Source dataframe containing the requested columns.
    columns:
        Column names to stack into a 2D float array.

    Returns
    -------
    numpy.ndarray
        Float feature matrix with shape ``(n_rows, len(columns))``.

    Raises
    ------
    ValidationError
        When any selected column contains null values.
    """
    block = frame[list(columns)]
    if block.isna().any().any():
        raise ValidationError(
            "Self-supervised learning requires non-null features. "
            "Call session.impute(...) first (and typically session.scale(...))."
        )
    return block.to_numpy(dtype=float)


def representation_column_names(prefix: str, latent_dim: int) -> tuple[str, ...]:
    """Generate stable embedding column names for attach/export.

    Names follow ``{prefix}_{i}`` so Session attach and bundle reload stay aligned.

    Parameters
    ----------
    prefix:
        Alphanumeric token prepended to each embedding index.
    latent_dim:
        Number of representation columns to name.

    Returns
    -------
    tuple[str, ...]
        Column names such as ``ssl_emb_0``, ``ssl_emb_1``, ...

    Raises
    ------
    ValidationError
        When ``prefix`` is empty or contains invalid characters.
    """
    token = str(prefix).strip() or "ssl_emb"
    if not token.replace("_", "").isalnum():
        raise ValidationError(
            "representation_prefix must be a non-empty alphanumeric token "
            f"(got {prefix!r})."
        )
    return tuple(f"{token}_{i}" for i in range(int(latent_dim)))


def _assert_numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(frame[c])]
    if non_numeric:
        raise ValidationError(
            "Self-supervised learning requires numeric columns; encode/scale first. "
            f"Non-numeric: {non_numeric[:12]}"
        )
