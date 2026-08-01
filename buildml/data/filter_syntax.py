"""Cross-engine helpers for :meth:`~buildml.data.dataset.Dataset.filter_expr`.

Polars and DuckDB both accept SQL-style boolean predicates, but identifier
quoting and string literals differ enough that ad-hoc strings break when the
engine changes. Prefer these helpers for simple comparisons that should work
on both engines when installed.
"""

from __future__ import annotations

from typing import Any

from buildml.core.errors import ValidationError

_ALLOWED_OPS = frozenset({">", ">=", "<", "<=", "==", "!=", "=", "<>"})


def quote_identifier(name: str) -> str:
    """Double-quote a SQL identifier (safe for Polars ``sql_expr`` and DuckDB).

    Parameters
    ----------
    name:
        Column name. Must be non-empty and must not contain null characters.
    """
    text = str(name)
    if not text or "\x00" in text:
        raise ValidationError("column name for filter_expr must be a non-empty string")
    return '"' + text.replace('"', '""') + '"'


def sql_literal(value: Any) -> str:
    """Render a Python scalar as a SQL literal for portable filter predicates.

    Supports ``None``, bool, int, float, and str. Other types raise
    :class:`~buildml.core.errors.ValidationError`.
    """
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if value != value:  # NaN
            raise ValidationError("float NaN is not a portable SQL literal")
        return repr(value)
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    raise ValidationError(
        f"unsupported literal type for portable filter_expr: {type(value).__name__}"
    )


def portable_filter_expr(column: str, op: str, value: Any) -> str:
    """Build a SQL-style predicate usable with Polars and DuckDB ``filter_expr``.

    Parameters
    ----------
    column:
        Column name (quoted automatically).
    op:
        Comparison operator: ``>``, ``>=``, ``<``, ``<=``, ``==`` / ``=``,
        ``!=`` / ``<>``.
    value:
        Scalar compared against ``column``. Use :func:`sql_literal` rules.

    Returns
    -------
    str
        Predicate such as ``"score" >= 0.5`` suitable for
        :meth:`~buildml.data.dataset.Dataset.filter_expr`.

    Notes
    -----
    This covers simple comparisons only. Joins, ``IN`` lists, function calls,
    and engine-specific SQL remain engine-specific — write those predicates
    directly for the active engine.

    Examples
    --------
    >>> from buildml.data.filter_syntax import portable_filter_expr
    >>> portable_filter_expr("a", ">", 2)
    '"a" > 2'
    >>> portable_filter_expr("status", "==", "ok")
    '"status" = \'ok\''
    """
    operator = str(op).strip()
    if operator not in _ALLOWED_OPS:
        raise ValidationError(
            "portable_filter_expr op must be one of "
            ">, >=, <, <=, ==, =, !=, <>"
        )
    if operator == "==":
        operator = "="
    if operator == "!=":
        operator = "<>"
    return f"{quote_identifier(column)} {operator} {sql_literal(value)}"


__all__ = ["portable_filter_expr", "quote_identifier", "sql_literal"]
