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
    """Resolve numeric feature columns for SSL pretext (exclude protected roles)."""
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
    """Build a float design matrix; refuse null features."""
    block = frame[list(columns)]
    if block.isna().any().any():
        raise ValidationError(
            "Self-supervised learning requires non-null features. "
            "Call session.impute(...) first (and typically session.scale(...))."
        )
    return block.to_numpy(dtype=float)


def representation_column_names(prefix: str, latent_dim: int) -> tuple[str, ...]:
    """Stable embedding column names for attach/export."""
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
