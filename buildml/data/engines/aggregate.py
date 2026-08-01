"""Shared helpers for engine-native group aggregations."""

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
    """Return a canonical aggregate name or raise ``ValidationError``."""
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
    """Return the quantile level in ``[0, 1]`` for median/qN funcs, else ``None``."""
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
    """Normalize ``{column: func | [funcs]}`` into ordered ``(column, func)`` pairs.

    Use ``"*"`` with ``"count"`` for a row-count aggregate. Quantiles accept
    ``median``, ``q25``/``q50``/…, or ``quantile_0.25`` (integral percentiles).
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
    if column == "*" and func == "count":
        return "count"
    return f"{column}_{func}"


def validate_aggregate_columns(
    columns: Sequence[str],
    by: Sequence[str] | None,
    pairs: Sequence[tuple[str, str]],
) -> None:
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
    """Compute aggregations with Pandas (core fallback)."""
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
    """Build a DuckDB/SQL SELECT list for aggregations.

    Quantiles use ``quantile_cont`` (continuous). Cross-engine percentile
    values can differ slightly on ties; use ``materialize=True`` for a
    Pandas-only result when exact Pandas linear semantics are required.
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
