"""Polars engine adapter (optional extra)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from buildml.core.errors import MissingExtraError
from buildml.core.types import EngineName


class PolarsEngine:
    """Adapter for Polars DataFrames and LazyFrames.

    A ``LazyFrame`` may be stored as ``Dataset.native``. Projection stays lazy;
    row counts, samples, filters with positional masks, and Pandas promotion
    collect. This is not an out-of-core sklearn path.
    """

    name = EngineName.POLARS

    def __init__(self) -> None:
        try:
            import polars as pl
        except ImportError as exc:  # pragma: no cover - guarded by get_engine
            raise MissingExtraError("polars", "Polars engine") from exc
        self._pl = pl

    def _is_lazy(self, table: Any) -> bool:
        return isinstance(table, self._pl.LazyFrame)

    def _collect(self, table: Any) -> Any:
        if self._is_lazy(table):
            return table.collect()
        return table

    def from_pandas(self, frame: pd.DataFrame) -> Any:
        return self._pl.from_pandas(frame)

    def from_parquet(self, path: str | Path, *, lazy: bool = False) -> Any:
        """Load a Parquet file or directory as an eager DataFrame or LazyFrame scan."""
        target = Path(path)
        if target.is_dir():
            pattern = str(target / "*.parquet")
            if lazy:
                return self._pl.scan_parquet(pattern)
            return self._pl.read_parquet(pattern)
        if lazy:
            return self._pl.scan_parquet(target)
        return self._pl.read_parquet(target)

    def write_parquet(
        self,
        table: Any,
        path: str | Path,
        *,
        compression: str = "zstd",
        row_group_size: int | None = None,
    ) -> Path:
        """Write a native table to Parquet (collects LazyFrames once)."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, Any] = {"compression": compression}
        if row_group_size is not None:
            kwargs["row_group_size"] = int(row_group_size)
        if self._is_lazy(table):
            sink = getattr(table, "sink_parquet", None)
            if callable(sink):
                try:
                    sink(destination, **kwargs)
                except TypeError:
                    sink(destination)
                return destination
            self._collect(table).write_parquet(destination, **kwargs)
            return destination
        try:
            table.write_parquet(destination, **kwargs)
        except TypeError:
            table.write_parquet(destination)
        return destination

    def to_pandas(self, table: Any) -> pd.DataFrame:
        return self._collect(table).to_pandas()

    def n_rows(self, table: Any) -> int:
        if self._is_lazy(table):
            # Count without materializing all columns into a wide eager frame.
            return int(table.select(self._pl.len()).collect().item())
        return int(table.height)

    def columns(self, table: Any) -> list[str]:
        if self._is_lazy(table):
            schema = getattr(table, "collect_schema", None)
            if callable(schema):
                return list(schema().names())
            return list(table.schema.keys())
        return list(table.columns)

    def head(self, table: Any, n: int = 5) -> pd.DataFrame:
        if self._is_lazy(table):
            return table.head(n).collect().to_pandas()
        return table.head(n).to_pandas()

    def select_columns(self, table: Any, columns: list[str]) -> Any:
        # Preserve laziness for projection chains.
        return table.select(list(columns))

    def sample_rows(
        self,
        table: Any,
        n: int,
        *,
        random_state: int | None = None,
    ) -> Any:
        eager = self._collect(table)
        take = min(int(n), int(eager.height))
        return eager.sample(n=take, seed=random_state, shuffle=True)

    def filter_rows(self, table: Any, mask: list[bool] | tuple[bool, ...]) -> Any:
        eager = self._collect(table)
        if len(mask) != int(eager.height):
            raise ValueError(
                f"filter mask length {len(mask)} does not match table rows {eager.height}"
            )
        return eager.filter(self._pl.Series("_buildml_mask", list(mask)))

    def filter_expr(self, table: Any, expression: str) -> Any:
        """Filter with a SQL-style predicate, preserving LazyFrames when possible.

        Prefers ``polars.sql_expr`` so lazy plans stay lazy. Falls back to an
        eager collect only when SQL expression helpers are unavailable.
        """
        expr = str(expression).strip()
        if not expr:
            raise ValueError("filter_expr requires a non-empty expression")
        sql_expr = getattr(self._pl, "sql_expr", None)
        if callable(sql_expr):
            return table.filter(sql_expr(expr))
        # Older Polars: collect then filter via SQL on a temporary frame.
        eager = self._collect(table)
        sql = getattr(eager, "sql", None)
        if callable(sql):
            return sql(f"SELECT * FROM self WHERE {expr}")
        raise ValueError(
            "Polars filter_expr requires polars.sql_expr or DataFrame.sql; "
            "upgrade polars or use filter_rows(mask=...)."
        )

    def is_lazy_handle(self, table: Any) -> bool:
        """Return True when ``table`` is a Polars LazyFrame."""
        return self._is_lazy(table)

    def aggregate(
        self,
        table: Any,
        aggregations: dict[str, str | list[str]],
        *,
        by: list[str] | None = None,
    ) -> Any:
        """Group aggregations via Polars expressions (LazyFrame-safe)."""
        from buildml.data.engines.aggregate import (
            normalize_aggregations,
            output_name,
            quantile_level,
            validate_aggregate_columns,
        )

        pairs = normalize_aggregations(aggregations)
        validate_aggregate_columns(self.columns(table), by, pairs)
        pl = self._pl
        exprs: list[Any] = []
        for column, func in pairs:
            alias = output_name(column, func)
            if column == "*":
                exprs.append(pl.len().alias(alias))
                continue
            col = pl.col(column)
            if func == "sum":
                exprs.append(col.sum().alias(alias))
            elif func == "mean":
                exprs.append(col.mean().alias(alias))
            elif func == "min":
                exprs.append(col.min().alias(alias))
            elif func == "max":
                exprs.append(col.max().alias(alias))
            elif func == "count":
                exprs.append(col.count().alias(alias))
            elif func == "n_unique":
                exprs.append(col.n_unique().alias(alias))
            elif func == "std":
                exprs.append(col.std().alias(alias))
            else:
                q = quantile_level(func)
                if q is None:
                    raise ValueError(f"Unsupported aggregate '{func}'")
                # Continuous/linear interpolation; see Dataset.aggregate notes.
                exprs.append(col.quantile(q, interpolation="linear").alias(alias))
        if by:
            return table.group_by(list(by), maintain_order=True).agg(exprs)
        # Global aggregate: one row without group keys.
        return table.select(exprs)
