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
    """Quote a column name so the engine reads it as a name, not as syntax.

    Unquoted identifiers break on anything unusual — a space, a reserved word
    like ``order``, a leading digit, mixed case in a case-folding engine. Double
    quotes make the engine take the text literally.

    Parameters
    ----------
    name:
        The column name.

    Returns
    -------
    str
        The name in double quotes, with embedded quotes doubled.

    Raises
    ------
    ValidationError
        If the name is empty, or contains a null character.

    Notes
    -----
    **Doubling embedded quotes is what closes the injection hole.** A name
    containing ``"`` would otherwise end the quoted region early and let the
    rest be parsed as SQL.

    Examples
    --------
    >>> from buildml.data.filter_syntax import quote_identifier
    >>> quote_identifier("order")
    '"order"'

    See Also
    --------
    portable_filter_expr : Which applies this for you.
    """
    text = str(name)
    if not text or "\x00" in text:
        raise ValidationError("column name for filter_expr must be a non-empty string")
    return '"' + text.replace('"', '""') + '"'


def sql_literal(value: Any) -> str:
    """Render a Python value as a SQL literal both engines will accept.

    Handles the small conversions that differ between Python and SQL: ``None``
    becomes ``NULL``, ``True`` becomes ``TRUE``, and strings are single-quoted
    with embedded quotes doubled.

    Parameters
    ----------
    value:
        A scalar: ``None``, bool, int, float, or str.

    Returns
    -------
    str
        The SQL literal.

    Raises
    ------
    ValidationError
        If the value is NaN, or of an unsupported type. Dates, decimals, and
        collections are deliberately excluded — their SQL spellings are
        engine-specific, so a portable rendering would be a guess.

    Notes
    -----
    **NaN is rejected rather than rendered.** SQL has no portable NaN literal,
    and comparisons against it are false in every direction anyway, so a
    predicate containing one silently matches nothing. Test for missingness with
    ``IS NULL`` instead.

    **Doubling embedded quotes is what closes the injection hole** for string
    values, the same as in :func:`quote_identifier`.

    Examples
    --------
    >>> from buildml.data.filter_syntax import sql_literal
    >>> sql_literal("o'brien")
    "'o''brien'"

    See Also
    --------
    portable_filter_expr : Which applies this for you.
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
    r"""Build a comparison predicate that works on both Polars and DuckDB.

    Hand-written predicate strings tend to work on the engine they were written
    against and break on the other, usually over quoting. This builds the
    quoting correctly and accepts Python's ``==`` and ``!=`` alongside SQL's
    ``=`` and ``<>``, so the same call site survives an engine change.

    Parameters
    ----------
    column:
        The column name. Quoted for you.
    op:
        One of ``>``, ``>=``, ``<``, ``<=``, ``==`` or ``=``, ``!=`` or ``<>``.
    value:
        The scalar to compare against, rendered by :func:`sql_literal`.

    Returns
    -------
    str
        A predicate such as ``'"score" >= 0.5'``, ready for
        :meth:`~buildml.data.dataset.Dataset.filter_expr`.

    Raises
    ------
    ValidationError
        If the operator is not in the allowed set, the column name is invalid,
        or the value cannot be rendered portably. The allowlist is what keeps
        arbitrary text out of the generated SQL.

    Notes
    -----
    **Comparisons only.** ``IN`` lists, ``LIKE``, function calls, joins, and
    window functions are not portable and are not covered — write those directly
    for the engine you are on.

    **Comparing against ``None`` produces ``= NULL``, which is never true.** SQL
    requires ``IS NULL``, which this does not generate. Filter for missingness
    with an engine-specific predicate.

    Examples
    --------
    >>> from buildml.data.filter_syntax import portable_filter_expr
    >>> portable_filter_expr("a", ">", 2)
    '"a" > 2'
    >>> portable_filter_expr("status", "==", "ok")
    '"status" = \'ok\''

    See Also
    --------
    buildml.data.dataset.Dataset.filter_expr : Where the result goes.
    quote_identifier : The column half.
    sql_literal : The value half.
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
