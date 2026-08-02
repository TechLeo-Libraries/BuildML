"""Column-level preparation helpers and role-aware transform resolution."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.core.validation import validate_column_names
from buildml.data.dataset import Dataset
from buildml.ingest.detect import schema_from_dataframe

# Roles never mutated by default Session preprocess (scale/encode/impute/…).
# Explicit ``columns=…`` is the opt-in to force-include any of these.
DEFAULT_SKIP_ROLES: frozenset[ColumnRole] = frozenset(
    {
        ColumnRole.TARGET,
        ColumnRole.ID,
        ColumnRole.GROUP,
        ColumnRole.TIME,
        ColumnRole.WEIGHT,
        ColumnRole.IGNORE,
    }
)

ColumnKind = Literal["numeric", "categorical", "text", "any"]


def drop_columns(dataset: Dataset, columns: list[str] | tuple[str, ...]) -> Dataset:
    """Return a new dataset with the given columns removed.

    Parameters
    ----------
    dataset:
        Source dataset.
    columns:
        Column names to drop.

    Returns
    -------
    Dataset
        New dataset (original is not mutated).

    Notes
    -----
    Roles for removed columns are discarded. Split membership remains valid
    because row identity/order is unchanged.
    """
    cols = validate_column_names(columns, dataset.columns)
    remaining = [c for c in dataset.columns if c not in set(cols)]
    if not remaining:
        raise ValidationError("Cannot drop all columns from the dataset")

    frame = dataset._ensure_pandas().drop(columns=list(cols)).copy()
    roles = {k: v for k, v in dataset.roles.items() if k in remaining}
    return Dataset.from_transformed(
        dataset,
        frame,
        schema=schema_from_dataframe(frame),
        roles=roles,
    )


def select_columns(dataset: Dataset, columns: list[str] | tuple[str, ...]) -> Dataset:
    """Return a new dataset keeping only the requested columns."""
    cols = validate_column_names(columns, dataset.columns)
    frame: pd.DataFrame = dataset._ensure_pandas().loc[:, list(cols)].copy()
    roles = {k: v for k, v in dataset.roles.items() if k in cols}
    return Dataset.from_transformed(
        dataset,
        frame,
        schema=schema_from_dataframe(frame),
        roles=roles,
    )


def protected_role_columns(
    dataset: Dataset,
    *,
    skip_roles: frozenset[ColumnRole] | set[ColumnRole] | None = None,
) -> list[str]:
    """Return column names whose roles are skipped by default preprocess."""
    blocked = DEFAULT_SKIP_ROLES if skip_roles is None else frozenset(skip_roles)
    return [name for name, role in dataset.roles.items() if role in blocked]


def resolve_transform_columns(
    dataset: Dataset,
    train: pd.DataFrame,
    columns: list[str] | None,
    *,
    kind: ColumnKind = "numeric",
    require_dtype: bool = True,
    empty_message: str | None = None,
) -> list[str]:
    """Resolve columns for a Session preprocess transform.

    Default behaviour (``columns is None``)
    ---------------------------------------
    Prefer columns with role ``feature`` when any are set. Otherwise take all
    columns that are not in :data:`DEFAULT_SKIP_ROLES` (``target``, ``id``,
    ``group``, ``time``, ``weight``, ``ignore``). Filter by ``kind`` dtype.

    Explicit opt-in (``columns=[...]``)
    -----------------------------------
    Force-include the named columns even when their roles are ``ignore`` /
    ``id`` / etc. Dtype checks still apply when ``require_dtype=True``.

    Parameters
    ----------
    kind:
        ``numeric`` — pandas numeric dtypes;
        ``categorical`` — object / category / string;
        ``text`` — string / object (non-numeric);
        ``any`` — no dtype filter.
    require_dtype:
        When ``True`` and ``columns`` is explicit, raise if a named column
        fails the ``kind`` check (numeric/categorical/text).
    empty_message:
        Override the :class:`~buildml.core.errors.ValidationError` message
        when no columns resolve.
    """
    if columns is not None:
        names = validate_column_names(columns, dataset.columns)
        if require_dtype and kind != "any":
            bad = [n for n in names if not _matches_kind(train[n], kind)]
            if bad:
                raise ValidationError(_dtype_error(kind, bad))
        return names

    feature_roles = [str(c) for c in dataset.role_columns(ColumnRole.FEATURE) if c in train.columns]
    if feature_roles:
        candidates = feature_roles
    else:
        blocked = set(protected_role_columns(dataset))
        candidates = [str(c) for c in train.columns if c not in blocked]

    names = [c for c in candidates if c in train.columns and _matches_kind(train[c], kind)]
    if not names:
        raise ValidationError(
            empty_message
            or _default_empty_message(kind)
        )
    return names


def _matches_kind(series: pd.Series, kind: ColumnKind) -> bool:
    if kind == "any":
        return True
    if kind == "numeric":
        return bool(pd.api.types.is_numeric_dtype(series))
    if kind == "categorical":
        return bool(
            pd.api.types.is_object_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(series)
        )
    if kind == "text":
        return bool(
            (pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series))
            and not pd.api.types.is_numeric_dtype(series)
        )
    raise ValidationError(f"Unknown column kind '{kind}'")


def _dtype_error(kind: ColumnKind, bad: list[str]) -> str:
    shown = bad[:12]
    if kind == "numeric":
        return f"Requires numeric columns; non-numeric: {shown}"
    if kind == "categorical":
        return f"Requires categorical columns; non-categorical: {shown}"
    if kind == "text":
        return f"Requires text/object columns; invalid: {shown}"
    return f"Invalid columns for kind '{kind}': {shown}"


def _default_empty_message(kind: ColumnKind) -> str:
    if kind == "numeric":
        return (
            "No numeric feature columns available. "
            "Pass columns=... explicitly to include ignore/id roles."
        )
    if kind == "categorical":
        return (
            "No categorical feature columns available. "
            "Pass columns=... explicitly to include ignore/id roles."
        )
    if kind == "text":
        return (
            "No text/object feature columns available. "
            "Pass columns=... explicitly."
        )
    return (
        "No feature columns available. "
        "Pass columns=... explicitly to include ignore/id roles."
    )
