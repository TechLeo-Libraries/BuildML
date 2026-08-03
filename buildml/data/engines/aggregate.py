"""Speak one aggregation vocabulary to three engines that disagree.

Every engine spells the same summary differently. A standard deviation is
``std`` in pandas and ``STDDEV_SAMP`` in SQL; a distinct count is ``nunique``,
``n_unique``, or ``COUNT(DISTINCT ...)``. Left alone, that means aggregation
code has to be written once per backend, and the versions drift.

This module defines the vocabulary BuildML accepts, validates a request against
it before any engine sees it, and provides the pandas implementation and the SQL
select-list builder. Polars translates the same normalised pairs into its own
expressions.

Accepted functions are ``sum``, ``mean``, ``min``, ``max``, ``count``,
``n_unique``, ``std``, ``median``, and integer percentiles ``q0`` through
``q100``. Output columns are named ``{column}_{func}``.

See Also
--------
buildml.data.dataset.Dataset.aggregate : The user-facing entry point.
buildml.data.engines.base.Engine.aggregate : The protocol method.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from buildml.core.errors import ValidationError

BASE_AGG_FUNCS = frozenset({"sum", "mean", "min", "max", "count", "n_unique", "std"})
# ``median`` and ``q{0-100}`` (integer percentiles) are also accepted.
SUPPORTED_AGG_FUNCS = BASE_AGG_FUNCS | frozenset({"median"}) | frozenset(
    f"q{p}" for p in range(0, 101)
)

_SQL_AGG = {
    "sum": "SUM",
    "mean": "AVG",
    "min": "MIN",
    "max": "MAX",
    "count": "COUNT",
    "n_unique": "COUNT(DISTINCT {expr})",
    "std": "STDDEV_SAMP",
}


def canonicalize_agg_func(func: str) -> str:
    """Reduce a function name to the one spelling the engines are given.

    Accepts the several ways people write the same thing — case and whitespace
    variation, ``quantile_0.25`` alongside ``q25`` — and returns a single
    canonical form, so downstream code has one name per operation to handle.

    Parameters
    ----------
    func:
        The function name as written.

    Returns
    -------
    str
        The canonical name: a base function, ``'median'``, or ``'qN'``.

    Raises
    ------
    ValidationError
        If the name is not recognised, or a quantile level is outside
        ``[0, 1]`` or not an integer percentage. The message lists what is
        supported.

    Notes
    -----
    **``q50`` canonicalises to ``median``**, so the same statistic has one name
    however it was requested.

    **Only integer percentiles are accepted.** ``quantile_0.333`` is rejected
    rather than rounded, because silently answering a slightly different
    question is worse than refusing.

    Examples
    --------
    >>> from buildml.data.engines.aggregate import canonicalize_agg_func
    >>> canonicalize_agg_func("quantile_0.5")
    'median'

    See Also
    --------
    normalize_aggregations : Which applies this across a request.
    """
    name = str(func).lower().strip()
    if name in BASE_AGG_FUNCS or name == "median":
        return name
    if name.startswith("q") and name[1:].isdigit():
        pct = int(name[1:])
        if 0 <= pct <= 100:
            return f"q{pct}"
    # Accept quantile_0.25 / quantile_0_25 → q25 when the percent is integral.
    if name.startswith("quantile_"):
        raw = name[len("quantile_") :].replace("_", ".", 1)
        try:
            q = float(raw)
        except ValueError as exc:
            raise ValidationError(
                f"Unsupported aggregate '{func}'. "
                f"Supported: {sorted(BASE_AGG_FUNCS)} plus 'median' and 'q0'..'q100'."
            ) from exc
        if not 0.0 <= q <= 1.0:
            raise ValidationError(
                f"Quantile level for '{func}' must be between 0 and 1 inclusive"
            )
        pct = round(q * 100)
        if abs(q * 100 - pct) > 1e-9:
            raise ValidationError(
                f"Unsupported aggregate '{func}'. "
                "Use integer percentiles via 'q25', 'q50', … or 'median'."
            )
        return "median" if pct == 50 else f"q{pct}"
    raise ValidationError(
        f"Unsupported aggregate '{func}'. "
        f"Supported: {sorted(BASE_AGG_FUNCS)} plus 'median' and 'q0'..'q100'."
    )


def quantile_level(func: str) -> float | None:
    """Turn a quantile function name into the fraction the engines want.

    Every engine's quantile call takes a number in ``[0, 1]``. This converts
    ``'median'`` and ``'qN'`` into that number, and returns ``None`` for
    anything else — which is how callers branch between quantile and non-
    quantile handling.

    Parameters
    ----------
    func:
        A canonical function name.

    Returns
    -------
    float or None
        The level, or ``None`` when the function is not a quantile.

    Notes
    -----
    **``None`` is a signal, not a failure.** Callers use it to decide which
    branch to take.

    Examples
    --------
    >>> from buildml.data.engines.aggregate import quantile_level
    >>> quantile_level("q75")
    0.75
    >>> quantile_level("mean") is None
    True

    See Also
    --------
    canonicalize_agg_func : Producing the names this accepts.
    """
    name = str(func).lower().strip()
    if name == "median":
        return 0.5
    if name.startswith("q") and name[1:].isdigit():
        pct = int(name[1:])
        if 0 <= pct <= 100:
            return pct / 100.0
    return None


def normalize_aggregations(
    aggregations: Mapping[str, str | Sequence[str]],
) -> list[tuple[str, str]]:
    """Flatten an aggregation request into ordered pairs.

    The convenient way to write a request — a mapping, with either one function
    or a list per column — is not the convenient way to consume it. This flattens
    it to ``(column, func)`` pairs in a fixed order, canonicalising each name on
    the way, so every engine builds its output columns identically.

    Parameters
    ----------
    aggregations:
        Column to function name, or to a list of names. ``'*'`` with
        ``'count'`` means a row count.

    Returns
    -------
    list of tuple of (str, str)
        Pairs in request order.

    Raises
    ------
    ValidationError
        If the request is empty, a column maps to an empty list, a function is
        not recognised, or ``'*'`` is paired with anything but ``'count'``.

    Notes
    -----
    **Order is preserved deliberately.** It determines output column order, so
    the same request gives the same layout on every engine.

    **Duplicates are not removed.** Asking for the same pair twice produces the
    column twice.

    Examples
    --------
    >>> from buildml.data.engines.aggregate import normalize_aggregations
    >>> normalize_aggregations({"x": ["mean", "std"], "*": "count"})
    [('x', 'mean'), ('x', 'std'), ('*', 'count')]

    See Also
    --------
    validate_aggregate_columns : Checking the columns exist.
    output_name : Naming the results.
    """
    if not aggregations:
        raise ValidationError("aggregations must include at least one column/function pair")
    pairs: list[tuple[str, str]] = []
    for column, raw in aggregations.items():
        col = str(column)
        funcs = [raw] if isinstance(raw, str) else list(raw)
        if not funcs:
            raise ValidationError(f"aggregations['{col}'] must not be empty")
        for func in funcs:
            name = canonicalize_agg_func(str(func))
            if col == "*" and name != "count":
                raise ValidationError("aggregations['*'] only supports 'count'")
            pairs.append((col, name))
    return pairs


def output_name(column: str, func: str) -> str:
    """Name the column an aggregation produces.

    One rule, applied by every engine, so results have the same column names
    whichever backend computed them.

    Parameters
    ----------
    column:
        The source column, or ``'*'``.
    func:
        The canonical function name.

    Returns
    -------
    str
        ``'{column}_{func}'``, or plain ``'count'`` for a row count.

    Notes
    -----
    **Names can collide with real columns.** A source column already called
    ``x_mean`` will clash with the mean of ``x``; nothing here detects that.

    Examples
    --------
    >>> from buildml.data.engines.aggregate import output_name
    >>> output_name("revenue", "mean")
    'revenue_mean'
    """
    if column == "*" and func == "count":
        return "count"
    return f"{column}_{func}"


def validate_aggregate_columns(
    columns: Sequence[str],
    by: Sequence[str] | None,
    pairs: Sequence[tuple[str, str]],
) -> None:
    """Check every referenced column exists before any engine sees the request.

    Catches typos here, where the message can name the missing column, rather
    than as a ``KeyError`` from pandas or a syntax error from a database several
    frames deeper.

    Parameters
    ----------
    columns:
        The columns actually available.
    by:
        Group-by columns to check, or ``None``.
    pairs:
        Normalised aggregation pairs.

    Returns
    -------
    None
        Returns nothing on success; the value is the absence of an exception.

    Raises
    ------
    ValidationError
        If any group-by or aggregated column is missing.

    Notes
    -----
    **``'*'`` is skipped**, since it refers to rows rather than a column.

    **Types are not checked.** Asking for the mean of a text column passes here
    and fails in the engine.

    See Also
    --------
    normalize_aggregations : Producing the pairs.
    """
    available = set(columns)
    if by is not None:
        missing_by = [c for c in by if c not in available]
        if missing_by:
            raise ValidationError(f"aggregate by-columns missing: {missing_by}")
    for column, _func in pairs:
        if column == "*":
            continue
        if column not in available:
            raise ValidationError(f"aggregate column missing: {column}")


def _series_agg(series: pd.Series, func: str) -> Any:
    if func == "sum":
        return series.sum(min_count=1)
    if func == "mean":
        return series.mean()
    if func == "min":
        return series.min()
    if func == "max":
        return series.max()
    if func == "count":
        return int(series.count())
    if func == "n_unique":
        return int(series.nunique(dropna=True))
    if func == "std":
        return series.std(ddof=1)
    q = quantile_level(func)
    if q is not None:
        # Linear (continuous) interpolation; matches Polars/DuckDB continuous paths.
        return series.quantile(q, interpolation="linear")
    raise ValidationError(f"Unsupported aggregate '{func}'")


def aggregate_pandas(
    frame: pd.DataFrame,
    pairs: Sequence[tuple[str, str]],
    *,
    by: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute the aggregations in pandas.

    Used directly by the pandas engine, and as the fallback whenever an
    optional engine is unavailable or a caller asks for exact pandas semantics.
    Its results are the reference the other engines are compared against.

    Parameters
    ----------
    frame:
        The data.
    pairs:
        Normalised aggregation pairs. Validate them first.
    by:
        Group-by columns. Omit for a single summary row.

    Returns
    -------
    pandas.DataFrame
        One row per group, or one row overall. Group keys come first, then the
        aggregates in request order.

    Raises
    ------
    ValidationError
        If a function name is not supported.

    Notes
    -----
    **Missing group keys form their own group** rather than being dropped, so
    rows with a null key still appear in the output.

    **Group order follows first appearance**, not sort order, which keeps
    results reproducible.

    **Sums of all-missing columns are missing, not zero.** ``sum`` uses
    ``min_count=1``, because a zero would look like a real measurement.

    **Standard deviation is the sample form** with ``ddof=1``.

    **Groups are iterated in Python**, so a very large number of groups is slow
    here in a way it is not in Polars or DuckDB.

    See Also
    --------
    sql_aggregate_select : The DuckDB equivalent.
    """
    if not by:
        data: dict[str, Any] = {}
        for column, func in pairs:
            name = output_name(column, func)
            if column == "*":
                data[name] = int(len(frame))
            else:
                data[name] = _series_agg(frame[column], func)
        return pd.DataFrame([data])

    by_cols = list(by)
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(by_cols, dropna=False, sort=False)
    for key, group in grouped:
        key_vals = key if isinstance(key, tuple) else (key,)
        row = {col: val for col, val in zip(by_cols, key_vals, strict=True)}
        for column, func in pairs:
            name = output_name(column, func)
            if column == "*":
                row[name] = int(len(group))
            else:
                row[name] = _series_agg(group[column], func)
        rows.append(row)
    if not rows:
        columns = by_cols + [output_name(c, f) for c, f in pairs]
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame(rows)
    ordered = by_cols + [output_name(c, f) for c, f in pairs]
    return out.loc[:, ordered].copy()


def sql_aggregate_select(
    pairs: Sequence[tuple[str, str]],
    *,
    by: Sequence[str] | None = None,
) -> str:
    """Render the aggregations as a SQL select list.

    Translates canonical function names into SQL — ``std`` to ``STDDEV_SAMP``,
    ``n_unique`` to ``COUNT(DISTINCT ...)``, quantiles to ``quantile_cont`` —
    and aliases each result to the shared output name, so DuckDB produces the
    same column names as the other engines.

    Parameters
    ----------
    pairs:
        Normalised aggregation pairs. Validate them first.
    by:
        Group-by columns, emitted ahead of the aggregates.

    Returns
    -------
    str
        A comma-separated select list, ready to interpolate after ``SELECT``.

    Notes
    -----
    **Identifiers are quoted, so unusual column names work.** Values are not
    involved — nothing user-supplied beyond column names reaches the SQL.

    **``quantile_cont`` can differ from pandas on ties.** Pass
    ``materialize=True`` at the Dataset level when the pandas value is the one
    that matters.

    See Also
    --------
    aggregate_pandas : The reference implementation.
    """
    parts: list[str] = []
    if by:
        parts.extend(f'"{c}"' for c in by)
    for column, func in pairs:
        alias = output_name(column, func)
        if column == "*":
            parts.append(f'COUNT(*) AS "{alias}"')
            continue
        expr = f'"{column}"'
        q = quantile_level(func)
        if q is not None:
            parts.append(f'quantile_cont({expr}, {q}) AS "{alias}"')
            continue
        template = _SQL_AGG[func]
        if "{expr}" in template:
            parts.append(f'{template.format(expr=expr)} AS "{alias}"')
        else:
            parts.append(f'{template}({expr}) AS "{alias}"')
    return ", ".join(parts)
