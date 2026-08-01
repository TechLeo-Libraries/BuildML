"""Shared validation helpers."""

from __future__ import annotations

from collections.abc import Iterable

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole

_VALID_ROLES = {role.value for role in ColumnRole}


def validate_role_name(role: str | ColumnRole) -> ColumnRole:
    """Normalize and validate a column role.

    Parameters
    ----------
    role:
        Role enum or string name.

    Returns
    -------
    ColumnRole
        Normalized role.

    Raises
    ------
    ValidationError
        If the role is not recognized.
    """
    if isinstance(role, ColumnRole):
        return role
    key = str(role).strip().lower()
    if key not in _VALID_ROLES:
        valid = ", ".join(sorted(_VALID_ROLES))
        raise ValidationError(f"Unknown column role '{role}'. Valid roles: {valid}")
    return ColumnRole(key)


def validate_column_names(columns: Iterable[str], known: Iterable[str]) -> list[str]:
    """Ensure all requested columns exist.

    Raises
    ------
    ValidationError
        If any column is missing.
    """
    known_set = set(known)
    cols = [str(c) for c in columns]
    missing = [c for c in cols if c not in known_set]
    if missing:
        raise ValidationError(f"Unknown column(s): {missing}")
    return cols
