"""Reject bad input at the boundary, with a message that says how to fix it.

Both helpers here exist to convert the failure that would happen later into one
that happens now. A misspelled role or column name will eventually surface as a
``KeyError`` from deep inside an operation, naming neither what was asked for
nor what was available. Caught at the entry point instead, the error can list
the valid roles, or name the exact columns that were missing.

See Also
--------
buildml.core.types.ColumnRole : The roles being validated.
"""

from __future__ import annotations

from collections.abc import Iterable

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole

_VALID_ROLES = {role.value for role in ColumnRole}


def validate_role_name(role: str | ColumnRole) -> ColumnRole:
    """Accept a role as an enum or a string, and reject anything else by name.

    Roles are assigned in user code, often as string literals, so this is where
    a typo is caught. The error lists every valid role, which turns a guess into
    a correction.

    Parameters
    ----------
    role:
        A :class:`~buildml.core.types.ColumnRole`, or its string name. Case and
        surrounding whitespace are ignored, so ``'Target'`` and ``' target '``
        both work.

    Returns
    -------
    ColumnRole
        The normalised role.

    Raises
    ------
    ValidationError
        If the name matches no role. The message lists the valid ones.

    Examples
    --------
    >>> validate_role_name("Target")
    <ColumnRole.TARGET: 'target'>
    >>> validate_role_name(ColumnRole.GROUP)
    <ColumnRole.GROUP: 'group'>
    >>> from buildml.core.errors import ValidationError
    >>> try:
    ...     validate_role_name("label")
    ... except ValidationError as exc:
    ...     print("target" in str(exc))
    True

    See Also
    --------
    buildml.core.types.ColumnRole : What each role means for the workflow.
    """
    if isinstance(role, ColumnRole):
        return role
    key = str(role).strip().lower()
    if key not in _VALID_ROLES:
        valid = ", ".join(sorted(_VALID_ROLES))
        raise ValidationError(f"Unknown column role '{role}'. Valid roles: {valid}")
    return ColumnRole(key)


def validate_column_names(columns: Iterable[str], known: Iterable[str]) -> list[str]:
    """Check every requested column exists, and name all of the ones that do not.

    Reports the full set of missing columns rather than stopping at the first,
    so a list of ten column names with three typos takes one round trip to fix
    instead of three.

    Parameters
    ----------
    columns:
        The requested column names. Coerced to strings, so a numeric column name
        can be passed as a number.
    known:
        The names that actually exist, normally a frame's columns.

    Returns
    -------
    list of str
        The requested names as strings, in the order given. Returning them makes
        this usable inline where the caller needs the normalised list anyway.

    Raises
    ------
    ValidationError
        If any requested column is absent. The message lists all of them.

    Notes
    -----
    **Order is preserved, and duplicates are not removed.** The result mirrors
    what was asked for, since callers often use it to select columns and the
    ordering is theirs to decide.

    Examples
    --------
    >>> validate_column_names(["age", "income"], ["age", "income", "city"])
    ['age', 'income']
    >>> from buildml.core.errors import ValidationError
    >>> try:
    ...     validate_column_names(["age", "salary", "zip"], ["age", "income"])
    ... except ValidationError as exc:
    ...     print(exc)
    Unknown column(s): ['salary', 'zip']
    """
    known_set = set(known)
    cols = [str(c) for c in columns]
    missing = [c for c in cols if c not in known_set]
    if missing:
        raise ValidationError(f"Unknown column(s): {missing}")
    return cols
