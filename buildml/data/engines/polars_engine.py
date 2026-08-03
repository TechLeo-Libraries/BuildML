"""Fast columnar prep, with an option to defer the work entirely.

Polars offers two things pandas does not. Its eager DataFrame is simply faster —
multi-threaded, columnar, and free of the index that makes pandas alignment
surprising. Its ``LazyFrame`` is more interesting: operations build a plan
instead of computing, and when the plan finally runs the optimiser has seen all
of it at once, so it can push projections and filters down into the file scan
and skip reading what nothing asked for.

The adapter accepts either, and the distinction determines what is cheap.
Projection stays lazy. Row counts, samples, and positional masks force a
collect, because they need to know what the data actually is. So does pandas
promotion — which is where sklearn always ends up.

Requires the ``polars`` extra.

See Also
--------
buildml.data.engines.duckdb_engine : The SQL alternative.
buildml.data.dataset.Dataset : The caller.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from buildml.core.errors import MissingExtraError
from buildml.core.types import EngineName


class PolarsEngine:
    """Implement the engine protocol on Polars, eager or lazy.

    Every method accepts either a ``DataFrame`` or a ``LazyFrame`` and does the
    right thing, collecting only where an operation cannot be expressed lazily.
    Which operations those are is the useful thing to know, and it is recorded
    on each method.

    Attributes
    ----------
    name:
        :attr:`~buildml.core.types.EngineName.POLARS`.

    Notes
    -----
    **Lazy in, lazy out — mostly.** :meth:`select_columns`, :meth:`filter_expr`,
    and :meth:`aggregate` return LazyFrames when given one. :meth:`sample_rows`
    and :meth:`filter_rows` collect and return eager frames.

    **Not an out-of-core fitting path.** Laziness narrows what has to be loaded;
    sklearn still needs the design matrix in memory.

    See Also
    --------
    buildml.data.engines.base.Engine : The contract.
    """

    name = EngineName.POLARS

    def __init__(self) -> None:
        """Import Polars and prepare the adapter.

        Fails immediately and clearly when the optional extra is missing,
        rather than at the first operation.

        Raises
        ------
        MissingExtraError
            If ``polars`` is not installed. Install with
            ``pip install 'buildml[polars]'``.
        """
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
        """Convert a DataFrame into an eager Polars frame.

        A full conversion pass, worth paying when the operations that follow are
        heavy enough to benefit from Polars' speed.

        Parameters
        ----------
        frame:
            The data.

        Returns
        -------
        Any
            A Polars ``DataFrame``.

        Notes
        -----
        **Eager, not lazy.** There is no plan to optimise — the data is already
        in memory, so the deferral would buy nothing. Laziness pays off when
        reading from disk; see :meth:`from_parquet`.

        **The pandas index is dropped.** Polars has no index.

        **Object-dtype columns may not convert.** Mixed-type columns have no
        Polars equivalent and will raise; clean them first.
        """
        return self._pl.from_pandas(frame)

    def from_parquet(self, path: str | Path, *, lazy: bool = False) -> Any:
        """Read Parquet eagerly, or set up a scan that reads nothing yet.

        With ``lazy=True`` this is the most efficient entry point in the module.
        The scan knows the schema but has read no values, and any projection or
        filter applied afterwards is pushed into it — so columns you never ask
        for are never read off disk.

        Parameters
        ----------
        path:
            A ``.parquet`` file, or a directory whose ``*.parquet`` files are
            read as one table.
        lazy:
            Build a scan instead of reading. Strongly preferred for large files.

        Returns
        -------
        Any
            A ``LazyFrame`` when lazy, otherwise a ``DataFrame``.

        Notes
        -----
        **Directory reads assume a consistent schema.** Under ``lazy=True``,
        files that disagree fail at collection rather than here.

        See Also
        --------
        write_parquet : The other direction.
        """
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
        """Write a table to Parquet, streaming a lazy plan where possible.

        For a LazyFrame this prefers ``sink_parquet``, which executes the plan
        in batches and writes as it goes — so a result larger than memory can
        still be written.

        Parameters
        ----------
        table:
            A ``DataFrame`` or ``LazyFrame``.
        path:
            Destination. Parent directories are created.
        compression:
            Codec. ``zstd`` compresses well and decompresses fast, which suits
            data that will be re-read.
        row_group_size:
            Rows per group. Smaller groups allow finer-grained skipping on read;
            larger ones compress better.

        Returns
        -------
        pathlib.Path
            The path written.

        Notes
        -----
        **Only the sink path streams.** Where ``sink_parquet`` is unavailable
        the plan is collected in full first, and memory is then the limit.

        **``row_group_size`` is dropped on older Polars** rather than failing,
        since the write matters more than the tuning parameter.

        See Also
        --------
        from_parquet : Reading it back.
        """
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
        """Collect if lazy, then convert to a DataFrame.

        The boundary where deferral ends and sklearn begins. A LazyFrame
        executes its whole plan here, and the result lands in memory.

        Parameters
        ----------
        table:
            A ``DataFrame`` or ``LazyFrame``.

        Returns
        -------
        pandas.DataFrame
            The data.

        Notes
        -----
        **Peak memory is roughly double the result.** Both representations exist
        during the conversion.

        **Polars types do not all survive.** Nullable integers become floats or
        object columns in pandas, depending on version. Round-tripping is not
        guaranteed to preserve dtypes.
        """
        return self._collect(table).to_pandas()

    def n_rows(self, table: Any) -> int:
        """Report how many rows the table holds.

        On a LazyFrame the count is expressed as a plan that selects only the
        length, so the optimiser can skip reading column data entirely.

        Parameters
        ----------
        table:
            A ``DataFrame`` or ``LazyFrame``.

        Returns
        -------
        int
            The row count.

        Notes
        -----
        **Free on an eager frame; not always free on a lazy one.** A plain scan
        can answer from Parquet metadata, but a plan with filters must evaluate
        them.

        **The count is not cached.** Calling this in a loop over a lazy frame
        re-executes each time.
        """
        if self._is_lazy(table):
            # Count without materializing all columns into a wide eager frame.
            return int(table.select(self._pl.len()).collect().item())
        return int(table.height)

    def columns(self, table: Any) -> list[str]:
        """List the column names, in order.

        Read from the schema, which Polars knows for lazy plans as well as
        materialised frames.

        Parameters
        ----------
        table:
            A ``DataFrame`` or ``LazyFrame``.

        Returns
        -------
        list of str
            Column names.

        Notes
        -----
        **Cheap on both.** Nothing is collected, and no values are read.
        """
        if self._is_lazy(table):
            schema = getattr(table, "collect_schema", None)
            if callable(schema):
                return list(schema().names())
            return list(table.schema.keys())
        return list(table.columns)

    def head(self, table: Any, n: int = 5) -> pd.DataFrame:
        """Return the first few rows as pandas.

        The limit is applied before collection, so a lazy scan stops early
        rather than reading everything and slicing.

        Parameters
        ----------
        table:
            A ``DataFrame`` or ``LazyFrame``.
        n:
            How many rows.

        Returns
        -------
        pandas.DataFrame
            The first ``n`` rows.

        Notes
        -----
        **Row order is the file's order**, so on sorted or grouped data this
        shows one corner. Use :meth:`sample_rows` for something representative.
        """
        if self._is_lazy(table):
            return table.head(n).collect().to_pandas()
        return table.head(n).to_pandas()

    def select_columns(self, table: Any, columns: list[str]) -> Any:
        """Keep only the named columns.

        **The operation laziness exists for.** Applied to a lazy scan, the
        projection is pushed into the file read, so unselected columns are never
        touched — on a wide table that is most of the cost avoided.

        Parameters
        ----------
        table:
            A ``DataFrame`` or ``LazyFrame``.
        columns:
            Which to keep, in the desired order.

        Returns
        -------
        Any
            The same kind of object that was passed in, projected.

        Notes
        -----
        **Laziness is preserved**, so projections chain with filters and only
        the final collect does any work.

        **Missing columns fail at collection on a lazy frame**, which may be far
        from this call. :meth:`~buildml.data.dataset.Dataset.project` validates
        first.
        """
        # Preserve laziness for projection chains.
        return table.select(list(columns))

    def sample_rows(
        self,
        table: Any,
        n: int,
        *,
        random_state: int | None = None,
    ) -> Any:
        """Draw a random subset of rows.

        Sampling needs to know how many rows exist and reach them by position,
        so a lazy plan is collected first.

        Parameters
        ----------
        table:
            A ``DataFrame`` or ``LazyFrame``.
        n:
            How many rows. Clamped to the row count.
        random_state:
            Seed, for a reproducible draw.

        Returns
        -------
        Any
            An eager ``DataFrame``, even when given a LazyFrame.

        Notes
        -----
        **This breaks laziness.** A lazy chain that ends in a sample has
        materialised everything by the time the sample is taken. Project and
        filter first so that what gets collected is as small as possible.

        **The same seed gives different rows on a different engine.** Polars and
        DuckDB use unrelated generators.
        """
        eager = self._collect(table)
        take = min(int(n), int(eager.height))
        return eager.sample(n=take, seed=random_state, shuffle=True)

    def filter_rows(self, table: Any, mask: list[bool] | tuple[bool, ...]) -> Any:
        """Keep the rows a boolean mask selects.

        A positional mask has to line up with actual rows, so a lazy plan is
        collected before it can be applied.

        Parameters
        ----------
        table:
            A ``DataFrame`` or ``LazyFrame``.
        mask:
            One boolean per row, aligned to current order.

        Returns
        -------
        Any
            An eager ``DataFrame`` of the surviving rows.

        Raises
        ------
        ValueError
            If the mask length does not match the row count. A mismatch means
            the mask was built against different data.

        Notes
        -----
        **This breaks laziness**, and it is the main reason to prefer
        :meth:`filter_expr`, where the predicate is pushed into the scan and
        non-matching rows are never read.
        """
        eager = self._collect(table)
        if len(mask) != int(eager.height):
            raise ValueError(
                f"filter mask length {len(mask)} does not match table rows {eager.height}"
            )
        return eager.filter(self._pl.Series("_buildml_mask", list(mask)))

    def filter_expr(self, table: Any, expression: str) -> Any:
        """Filter with a predicate the optimiser can push into the scan.

        Unlike :meth:`filter_rows`, this keeps a LazyFrame lazy. The predicate
        becomes part of the plan, so Polars can use Parquet row-group statistics
        to skip blocks that cannot match — those rows are never read.

        Parameters
        ----------
        table:
            A ``DataFrame`` or ``LazyFrame``.
        expression:
            A SQL-style boolean predicate, such as ``'"score" >= 0.5'``.

        Returns
        -------
        Any
            The same kind of object that was passed in, filtered — unless the
            older fallback path runs, which collects.

        Raises
        ------
        ValueError
            If the expression is empty, or if the installed Polars provides
            neither ``sql_expr`` nor ``DataFrame.sql``.

        Notes
        -----
        **The dialect is Polars' SQL, not DuckDB's.** Simple comparisons written
        with :func:`~buildml.data.filter_syntax.portable_filter_expr` work on
        both; anything more does not.

        **Syntax errors surface at collection** on a lazy frame.
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
        """Report whether this handle is a lazy plan.

        Lets callers tell whether an operation will be free or will trigger
        execution — :class:`~buildml.data.dataset.Dataset` uses it to decide
        what to report about pending work.

        Parameters
        ----------
        table:
            The handle to inspect.

        Returns
        -------
        bool
            True for a ``LazyFrame``.

        Notes
        -----
        **Not part of the engine protocol.** Only this adapter offers it, and
        callers check with :func:`getattr` before using it.
        """
        return self._is_lazy(table)

    def aggregate(
        self,
        table: Any,
        aggregations: dict[str, str | list[str]],
        *,
        by: list[str] | None = None,
    ) -> Any:
        """Summarise columns, optionally per group, without collecting.

        Built from Polars expressions, so a LazyFrame stays lazy and the
        optimiser can read only the columns the aggregation touches.

        Parameters
        ----------
        table:
            A ``DataFrame`` or ``LazyFrame``.
        aggregations:
            Column to function name, or to a list of names. See
            :meth:`~buildml.data.engines.base.Engine.aggregate` for the
            supported set.
        by:
            Group-by columns. Omit for a single summary row.

        Returns
        -------
        Any
            The same kind of object that was passed in, aggregated. Output
            columns are named ``{column}_{func}``.

        Raises
        ------
        ValidationError
            If a named column does not exist.
        ValueError
            If a function name is not supported.

        Notes
        -----
        **Group order is preserved** as first encountered, so results are
        reproducible rather than dependent on hash ordering.

        **Quantiles interpolate linearly**, matching pandas. DuckDB's
        ``quantile_cont`` can differ slightly on ties.
        """
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
