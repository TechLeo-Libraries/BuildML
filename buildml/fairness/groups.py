"""Sensitive-attribute group key composition (including intersectional)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError

GROUP_KEY_SEP = "|"


def normalize_sensitive_columns(
    sensitive_column: str | Sequence[str],
) -> tuple[str, ...]:
    """Normalize a single column name or sequence into a non-empty tuple.

    Parameters
    ----------
    sensitive_column:
        One column name, or an ordered list/tuple of column names whose
        Cartesian combination defines intersectional groups.

    Returns
    -------
    tuple[str, ...]
        Ordered unique column names (duplicates rejected).

    Raises
    ------
    ValidationError
        When empty, blank, or containing duplicates.
    """
    if isinstance(sensitive_column, str):
        cols: tuple[str, ...] = (sensitive_column,)
    else:
        cols = tuple(str(c) for c in sensitive_column)
    if not cols:
        raise ValidationError(
            "sensitive_column must be a non-empty str or sequence of column names."
        )
    cleaned: list[str] = []
    seen: set[str] = set()
    for col in cols:
        name = str(col).strip()
        if not name:
            raise ValidationError("sensitive_column names must be non-empty strings.")
        if name in seen:
            raise ValidationError(
                f"Duplicate sensitive_column name {name!r}; each column may appear once."
            )
        seen.add(name)
        cleaned.append(name)
    return tuple(cleaned)


def sensitive_column_label(columns: Sequence[str], *, sep: str = GROUP_KEY_SEP) -> str:
    """Human-readable composite column label for reports."""
    cols = tuple(str(c) for c in columns)
    if len(cols) == 1:
        return cols[0]
    return sep.join(cols)


def compose_group_keys(
    *parts: Any,
    sep: str = GROUP_KEY_SEP,
) -> np.ndarray:
    """Build string group keys from one or more aligned sensitive arrays.

    Missing values become the literal token ``<NA>`` so they form an explicit
    group rather than silently dropping rows.

    Parameters
    ----------
    *parts:
        Aligned 1-d arrays / Series of equal length.
    sep:
        Separator between attribute levels (default ``|``).

    Returns
    -------
    numpy.ndarray
        Object array of group key strings, length matching inputs.

    Raises
    ------
    ValidationError
        When no parts are given or lengths disagree.
    """
    if not parts:
        raise ValidationError("compose_group_keys requires at least one sensitive array.")
    arrays = [_to_1d(p) for p in parts]
    n = len(arrays[0])
    for arr in arrays[1:]:
        if len(arr) != n:
            raise ValidationError("All sensitive arrays must have equal length.")
    if len(arrays) == 1:
        return np.asarray([_cell_token(v) for v in arrays[0]], dtype=object)
    keys = []
    for i in range(n):
        keys.append(sep.join(_cell_token(arr[i]) for arr in arrays))
    return np.asarray(keys, dtype=object)


def extract_sensitive_keys(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    sep: str = GROUP_KEY_SEP,
) -> np.ndarray:
    """Compose group keys from DataFrame columns.

    Raises
    ------
    ValidationError
        When a named column is missing from ``frame``.
    """
    cols = normalize_sensitive_columns(columns)
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ValidationError(
            f"sensitive_column(s) missing from frame: {missing!r}."
        )
    parts = [frame[c].to_numpy() for c in cols]
    return compose_group_keys(*parts, sep=sep)


def _to_1d(values: Any) -> np.ndarray:
    if isinstance(values, pd.Series):
        return values.to_numpy()
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValidationError("Sensitive arrays must be 1-dimensional.")
    return arr


def _cell_token(value: Any) -> str:
    if value is None:
        return "<NA>"
    try:
        if pd.isna(value):
            return "<NA>"
    except (TypeError, ValueError):
        pass
    return str(value)
