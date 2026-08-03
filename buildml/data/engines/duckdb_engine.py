"""Run prep as SQL, against data that may never be fully read.

DuckDB is an analytical database that runs in this process. There is no server,
and for BuildML's purposes it is a query planner that can read Parquet directly,
push projections and predicates down into the scan, and return only what was
asked for. On a wide table where a dozen columns matter, that is the difference
between reading a gigabyte and reading forty megabytes.

Everything here works on *relations*: unexecuted query plans. Projecting,
filtering, sampling, and aggregating all compose more plan; nothing runs until
something calls for pandas or Arrow. Arrow is the bridge at that boundary,
because it shares memory layout with DuckDB and avoids a serialisation pass.

Requires the ``duckdb`` extra. Every relation carries a connection that must
eventually be closed: see :class:`DuckDBTable`.

See Also
--------
buildml.data.engines.polars_engine : The other lazy engine.
buildml.data.dataset.Dataset : The caller.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from buildml.core.errors import MissingExtraError
from buildml.core.types import EngineName


class DuckDBTable:
    """A DuckDB relation, kept together with the connection it needs.

    A relation is only a query plan; executing it requires the connection it was
    built against. If that connection is garbage collected, the relation becomes
    unusable. Newer DuckDB relations reject arbitrary attributes, so the
    connection cannot simply be stapled on: hence this pairing.

    Attributes
    ----------
    relation:
        The unexecuted plan.
    connection:
        The connection that backs it.

    Notes
    -----
    **One connection is shared by a whole family of handles.** The Dataset that
    first opened it owns it; every projection and filter derived from that
    Dataset shares it without owning it. Close the owner and the derived handles
    stop working too, so keep the owner alive as long as anything descends from
    it.

    **Adapters do not open connections; operations do.** Calling
    :func:`~buildml.data.engines.get_engine` repeatedly is free. The first
    ``from_pandas``, ``from_parquet``, or ``from_arrow`` opens one, and further
    operations reuse it.

    **Unclosed connections leak.** Use ``with dataset:`` or call
    ``Dataset.close_native()`` on the owner.

    See Also
    --------
    close_duckdb_connection : Closing one directly.
    buildml.data.dataset.Dataset.close_native : The usual way.
    """

    __slots__ = ("relation", "connection")

    def __init__(self, relation: Any, connection: Any) -> None:
        """Pair a relation with the connection that must outlive it.

        Neither is validated; the caller is responsible for supplying a relation
        that was actually built against this connection.

        Parameters
        ----------
        relation:
            The DuckDB relation: an unexecuted query plan.
        connection:
            The connection it was built against.

        Notes
        -----
        Set through ``object.__setattr__`` because of ``__slots__``.
        """
        object.__setattr__(self, "relation", relation)
        object.__setattr__(self, "connection", connection)

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attributes to the wrapped relation.

        Lets this object stand in for a relation for read-only use, so callers
        holding a handle can reach relation methods directly.

        Parameters
        ----------
        name:
            The attribute being looked up.

        Returns
        -------
        Any
            The relation's attribute.

        Raises
        ------
        AttributeError
            If the relation has no such attribute.

        Notes
        -----
        **Forwarded methods return bare relations, not wrapped handles**, so
        the connection is lost from anything obtained this way. Use the adapter
        methods when the result needs to stay usable.
        """
        return getattr(self.relation, name)


def close_duckdb_connection(connection: Any) -> None:
    """Close a connection, tolerating one that is already gone.

    Cleanup runs in destructors, exception handlers, and ``finally`` blocks,
    where the connection may already be closed or partly torn down. Raising
    there would replace the real error with a noisy secondary one, so this
    swallows failures instead.

    Parameters
    ----------
    connection:
        The connection. ``None`` is accepted and does nothing.

    Notes
    -----
    **Failures are silent by design**, which does mean a genuinely stuck
    connection closes quietly. That trade is deliberate: cleanup should not be
    able to mask the error that triggered it.

    Safe to call more than once.

    See Also
    --------
    buildml.data.dataset.Dataset.close_native : The usual entry point.
    """
    if connection is None:
        return
    closer = getattr(connection, "close", None)
    if not callable(closer):
        return
    try:
        closer()
    except Exception:  # noqa: BLE001
        pass


def _as_relation(table: Any) -> Any:
    if isinstance(table, DuckDBTable):
        return table.relation
    return table


def _connection_of(table: Any, fallback: Any) -> Any:
    if isinstance(table, DuckDBTable) and table.connection is not None:
        return table.connection
    return fallback


def _wrap(relation: Any, connection: Any) -> DuckDBTable:
    return DuckDBTable(relation=relation, connection=connection)


def _relation_to_arrow_table(rel: Any) -> Any:
    """Materialize a DuckDB relation to a PyArrow Table.

    DuckDB's ``.arrow()`` may return a ``RecordBatchReader``; read it fully.
    """
    arrow_fn = getattr(rel, "arrow", None)
    if not callable(arrow_fn):
        raise RuntimeError("DuckDB relation does not expose .arrow()")
    payload = arrow_fn()
    read_all = getattr(payload, "read_all", None)
    if callable(read_all):
        return read_all()
    return payload


class DuckDBEngine:
    """Implement the engine protocol by building DuckDB query plans.

    Every operation composes more plan rather than computing anything. Only
    :meth:`to_pandas`, :meth:`to_arrow`, and :meth:`write_parquet` execute, and
    by then the plan describes exactly the columns and rows that are wanted :
    so the scan reads only those.

    Attributes
    ----------
    name:
        :attr:`~buildml.core.types.EngineName.DUCKDB`.

    Notes
    -----
    **Constructing this opens nothing.** Connections appear when a table is
    first created and live on the resulting :class:`DuckDBTable`.

    **Several methods fall back when the fast path fails**, and the fallbacks
    are slower and materialise more. Which one ran is recorded internally, so
    an unexpectedly slow operation can be traced to the path it took rather
    than guessed at.

    See Also
    --------
    buildml.data.engines.base.Engine : The contract.
    DuckDBTable : The handle returned.
    """

    name = EngineName.DUCKDB

    def __init__(self) -> None:
        """Import DuckDB and prepare the adapter, without connecting.

        Fails fast and clearly when the optional extra is missing, rather than
        at the first operation.

        Raises
        ------
        MissingExtraError
            If ``duckdb`` is not installed. Install with
            ``pip install 'buildml[duckdb]'``.

        Notes
        -----
        No connection is opened here, so constructing adapters is cheap and
        :func:`~buildml.data.engines.get_engine` can cache them.
        """
        try:
            import duckdb
        except ImportError as exc:  # pragma: no cover - guarded by get_engine
            raise MissingExtraError("duckdb", "DuckDB engine") from exc
        self._duckdb = duckdb
        self._bridge: str = "arrow"
        self._last_pushdown: str | None = None

    def _new_connection(self) -> Any:
        return self._duckdb.connect(database=":memory:")

    def from_pandas(self, frame: pd.DataFrame, *, connection: Any | None = None) -> Any:
        """Register a DataFrame as a DuckDB relation.

        Goes through Arrow when PyArrow is available, since Arrow and DuckDB
        share a memory layout and the conversion is close to free. Without it,
        DuckDB's own DataFrame reader is used, which copies.

        Parameters
        ----------
        frame:
            The data.
        connection:
            An existing connection to reuse. Omit to open one, which the
            returned handle then owns.

        Returns
        -------
        DuckDBTable
            The relation and its connection.

        Notes
        -----
        **Reuse a connection when rebuilding.** Repeated conversions that each
        open their own connection accumulate until something closes them.

        **The pandas index is dropped.** Anything meaningful in it should be a
        column first.
        """
        con = connection if connection is not None else self._new_connection()
        try:
            import pyarrow as pa

            table = pa.Table.from_pandas(frame, preserve_index=False)
            rel = con.from_arrow(table)
            self._bridge = "arrow"
            return _wrap(rel, con)
        except Exception:  # noqa: BLE001
            self._bridge = "pandas"
            return _wrap(con.from_df(frame), con)

    def from_arrow(self, table: Any, *, connection: Any | None = None) -> Any:
        """Register an Arrow table as a DuckDB relation, without copying.

        The cheapest way in. DuckDB reads Arrow buffers directly, so this is
        effectively free compared with going through pandas: worth using
        whenever the data is already Arrow, such as after a Parquet read.

        Parameters
        ----------
        table:
            A PyArrow table.
        connection:
            An existing connection to reuse. Omit to open one.

        Returns
        -------
        DuckDBTable
            The relation and its connection.

        Notes
        -----
        **The Arrow table must stay alive.** DuckDB references its buffers
        rather than copying them; releasing it while the relation is in use is
        undefined behaviour.

        See Also
        --------
        to_arrow : The other direction.
        """
        con = connection if connection is not None else self._new_connection()
        rel = con.from_arrow(table)
        self._bridge = "arrow"
        return _wrap(rel, con)

    def from_parquet(
        self,
        path: str | Path,
        *,
        lazy: bool = False,
        connection: Any | None = None,
        compression: str | None = None,
    ) -> Any:
        """Point DuckDB at a Parquet file or directory, reading nothing yet.

        **The best entry point for large data.** The relation knows the schema
        but has read no values. Projections and filters applied afterwards are
        pushed into the scan, so columns you never ask for are never read off
        disk at all.

        Parameters
        ----------
        path:
            A ``.parquet`` file, or a directory whose ``*.parquet`` files are
            read as one table.
        lazy:
            Ignored. DuckDB relations are always lazy; the parameter exists so
            call sites can be written once and work on Polars too.
        connection:
            An existing connection to reuse. Omit to open one.
        compression:
            Ignored on read. Parquet records its own codec.

        Returns
        -------
        DuckDBTable
            A relation over the file or directory.

        Notes
        -----
        **Directory reads assume a consistent schema.** Files that disagree
        about column names or types fail at execution, not here: the error
        surfaces on the first materialisation.

        **The path is interpolated into SQL for directory reads.** Directory
        names containing quotes will not work.

        See Also
        --------
        write_parquet : The other direction.
        """
        del lazy, compression
        target = Path(path)
        con = connection if connection is not None else self._new_connection()
        if target.is_dir():
            pattern = str(target / "*.parquet").replace("\\", "/")
            rel = con.sql(f"SELECT * FROM read_parquet('{pattern}')")
        else:
            rel = con.read_parquet(str(target))
        self._bridge = "parquet"
        self._last_pushdown = "read_parquet"
        return _wrap(rel, con)

    def write_parquet(
        self,
        table: Any,
        path: str | Path,
        *,
        compression: str = "zstd",
        row_group_size: int | None = None,
    ) -> Path:
        """Write a relation to Parquet without going through pandas.

        Executes the plan and streams the result to disk. Because pandas is not
        involved, the whole table never has to exist in memory at once.

        Parameters
        ----------
        table:
            The relation to write.
        path:
            Destination. Parent directories are created.
        compression:
            Codec. ``zstd`` compresses well and decompresses fast, which is
            usually the right default for data that gets re-read.
        row_group_size:
            Rows per group. Smaller groups allow finer-grained skipping on
            read; larger ones compress better.

        Returns
        -------
        pathlib.Path
            The path written.

        Notes
        -----
        **Three paths are tried in order**: the relation's own writer, a SQL
        ``COPY``, then an Arrow write. Only the last materialises the whole
        table, and it is reached only when the first two are unavailable.

        **``row_group_size`` is honoured only on the SQL path.** The relation
        writer does not accept it.

        See Also
        --------
        from_parquet : Reading it back.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        rel = _as_relation(table)
        con = _connection_of(table, None)
        codec = str(compression or "zstd").upper()
        writer = getattr(rel, "write_parquet", None)
        if callable(writer):
            try:
                writer(str(destination), compression=compression)
            except TypeError:
                writer(str(destination))
            self._last_pushdown = "write_parquet"
            return destination
        if con is not None:
            try:
                arrow_table = _relation_to_arrow_table(rel)
                con.register("_bml_write_src", arrow_table)
                opts = ["FORMAT PARQUET", f"COMPRESSION {codec}"]
                if row_group_size is not None:
                    opts.append(f"ROW_GROUP_SIZE {int(row_group_size)}")
                dest_sql = str(destination).replace("'", "''")
                con.execute(f"COPY _bml_write_src TO '{dest_sql}' ({', '.join(opts)})")
                try:
                    con.unregister("_bml_write_src")
                except Exception:  # noqa: BLE001
                    pass
                self._bridge = "sql"
                self._last_pushdown = "copy_parquet"
                return destination
            except Exception:  # noqa: BLE001
                pass
        import pyarrow.parquet as pq

        kwargs: dict[str, Any] = {"compression": compression}
        if row_group_size is not None:
            kwargs["row_group_size"] = int(row_group_size)
        pq.write_table(_relation_to_arrow_table(rel), destination, **kwargs)
        self._bridge = "arrow"
        self._last_pushdown = "arrow_write_parquet"
        return destination

    def to_pandas(self, table: Any) -> pd.DataFrame:
        """Execute the plan and return the result as a DataFrame.

        Where the deferral ends: the scan runs, the filters apply, and the whole
        result lands in memory.

        Parameters
        ----------
        table:
            The relation to materialise.

        Returns
        -------
        pandas.DataFrame
            The result.

        Notes
        -----
        **Goes through Arrow when possible**, which converts faster and
        preserves types better than DuckDB's direct DataFrame path.

        **Everything before this was free; this is not.** Narrow the plan first.
        """
        rel = _as_relation(table)
        try:
            return _relation_to_arrow_table(rel).to_pandas()
        except Exception:  # noqa: BLE001
            return rel.df()

    def to_arrow(self, table: Any) -> Any:
        """Execute the plan and return the result as Arrow.

        Cheaper than :meth:`to_pandas` and better at preserving types :
        especially nulls in integer columns, which pandas has historically
        widened to float. Prefer this when handing data to another Arrow-aware
        library.

        Parameters
        ----------
        table:
            The relation to materialise.

        Returns
        -------
        Any
            A PyArrow table.

        Raises
        ------
        RuntimeError
            If the relation does not expose Arrow output.

        Notes
        -----
        **A streamed reader is drained fully**, so the whole result is in memory
        when this returns. It is not an iterator.

        See Also
        --------
        from_arrow : The other direction.
        """
        return _relation_to_arrow_table(_as_relation(table))

    def n_rows(self, table: Any) -> int:
        """Count the rows the plan would produce.

        Uses the relation's shape when it is known, and otherwise runs a
        ``COUNT(*)``.

        Parameters
        ----------
        table:
            The relation.

        Returns
        -------
        int
            The row count.

        Notes
        -----
        **Counting can be nearly free or not, depending on the plan.** Parquet
        stores row counts in metadata, so a plain scan is cheap; a filtered plan
        has to evaluate the predicate to know the answer.
        """
        rel = _as_relation(table)
        shape = getattr(rel, "shape", None)
        if shape is not None:
            return int(shape[0])
        return int(rel.count("*").fetchone()[0])

    def columns(self, table: Any) -> list[str]:
        """List the columns the plan would produce.

        Read from the relation's schema, so this does not execute anything.

        Parameters
        ----------
        table:
            The relation.

        Returns
        -------
        list of str
            Column names, in order.

        Notes
        -----
        There are fallbacks through Arrow and pandas for relations that do not
        expose a schema directly, and those do materialise. In practice DuckDB
        always exposes one.
        """
        rel = _as_relation(table)
        cols = getattr(rel, "columns", None)
        if cols is not None:
            return [str(c) for c in cols]
        try:
            return [str(c) for c in _relation_to_arrow_table(rel).column_names]
        except Exception:  # noqa: BLE001
            return list(rel.df().columns.astype(str))

    def head(self, table: Any, n: int = 5) -> pd.DataFrame:
        """Return the first few rows as pandas.

        Adds a ``LIMIT`` to the plan before executing, so the scan stops early
        rather than reading the whole table and slicing.

        Parameters
        ----------
        table:
            The relation.
        n:
            How many rows.

        Returns
        -------
        pandas.DataFrame
            The first ``n`` rows.

        Notes
        -----
        **Which rows these are is not defined.** Relational results have no
        inherent order, so without an ``ORDER BY`` the same limit can return
        different rows across runs or after a change in the plan.
        """
        rel = _as_relation(table)
        limited = rel.limit(n)
        try:
            return _relation_to_arrow_table(limited).to_pandas()
        except Exception:  # noqa: BLE001
            return limited.df()

    def select_columns(self, table: Any, columns: list[str]) -> Any:
        """Add a projection to the plan.

        The most valuable operation here. On a Parquet-backed relation the
        projection is pushed into the scan, so unselected columns are never read
        off disk: a wide table costs only the columns you named.

        Parameters
        ----------
        table:
            The relation.
        columns:
            Which to keep, in the desired order.

        Returns
        -------
        DuckDBTable
            A projected relation, sharing the connection.

        Raises
        ------
        RuntimeError
            If the handle carries no connection.

        Notes
        -----
        **Column names are quoted but not validated here.** A name that does not
        exist fails at execution, which may be some distance from this call.
        :meth:`~buildml.data.dataset.Dataset.project` checks first.
        """
        rel = _as_relation(table)
        con = _connection_of(table, None)
        if con is None:
            raise RuntimeError("DuckDB select_columns requires a DuckDBTable with connection")
        exprs = ", ".join(f'"{c}"' for c in columns)
        self._last_pushdown = "project"
        return _wrap(rel.project(exprs), con)

    def sample_rows(
        self,
        table: Any,
        n: int,
        *,
        random_state: int | None = None,
    ) -> Any:
        """Draw rows inside the engine, without materialising the table.

        Two strategies, chosen by whether reproducibility is required. Unseeded
        draws use DuckDB's ``USING SAMPLE``, which is genuinely random and fast.
        Seeded draws order rows by a hash of their leading columns combined with
        the seed and take the first ``n``: deterministic, and still evaluated
        in the engine.

        Parameters
        ----------
        table:
            The relation.
        n:
            How many rows. Clamped to the row count.
        random_state:
            Seed. Omit for a genuinely random draw.

        Returns
        -------
        DuckDBTable
            A sampled relation, sharing the connection.

        Raises
        ------
        RuntimeError
            If the handle carries no connection.

        Notes
        -----
        **A seeded draw is deterministic but not uniformly random.** Hashing the
        first three columns and taking the smallest hashes is stable across
        runs, which is what reproducibility needs, but rows with similar leading
        values are not independently selected. Do not treat a seeded sample as a
        statistically clean draw.

        **The same seed gives different rows on a different engine**, since the
        hash functions differ.

        **Sampling requires a row count**, which executes enough of the plan to
        produce one.
        """
        rel = _as_relation(table)
        con = _connection_of(table, None)
        if con is None:
            raise RuntimeError("DuckDB sample_rows requires a DuckDBTable with connection")
        take = min(int(n), int(self.n_rows(table)))
        if take <= 0:
            self._last_pushdown = "limit0"
            return _wrap(rel.limit(0), con)
        if random_state is None:
            try:
                sampled = rel.query("s", f"SELECT * FROM s USING SAMPLE {take} ROWS")
                self._last_pushdown = "using_sample"
                self._bridge = "sql"
                return _wrap(sampled, con)
            except Exception:  # noqa: BLE001
                self._last_pushdown = "order_random_limit"
                return _wrap(rel.order("random()").limit(take), con)
        seed = int(random_state)
        order_exprs = ", ".join(
            f"hash(CAST(\"{c}\" AS VARCHAR) || '{seed}')" for c in self.columns(table)[:3]
        )
        if not order_exprs:
            self._last_pushdown = "limit"
            return _wrap(rel.limit(take), con)
        self._last_pushdown = "hash_order_limit"
        self._bridge = "sql"
        return _wrap(rel.order(order_exprs).limit(take), con)

    def filter_expr(self, table: Any, expression: str) -> Any:
        """Add a SQL predicate to the plan.

        **The most efficient way to drop rows.** On a Parquet-backed relation
        DuckDB pushes the predicate into the scan and uses row-group statistics
        to skip whole blocks: non-matching rows are never read.

        Parameters
        ----------
        table:
            The relation.
        expression:
            A SQL boolean expression, such as ``'"score" >= 0.5'``.

        Returns
        -------
        DuckDBTable
            A filtered relation, sharing the connection.

        Raises
        ------
        ValueError
            If the expression is empty or whitespace.
        RuntimeError
            If the handle carries no connection.

        Notes
        -----
        **The expression is interpolated into SQL, not parameterised.** Never
        build one from untrusted input; use
        :func:`~buildml.data.filter_syntax.portable_filter_expr`, which quotes
        and escapes.

        **Syntax errors surface at execution**, not here. The plan is built
        without validating the predicate.

        See Also
        --------
        filter_rows : When the condition is easier in Python.
        """
        if not str(expression).strip():
            raise ValueError("filter_expr requires a non-empty SQL expression")
        rel = _as_relation(table)
        con = _connection_of(table, None)
        if con is None:
            raise RuntimeError("DuckDB filter_expr requires a DuckDBTable with connection")
        self._last_pushdown = "filter_expr"
        self._bridge = "sql"
        return _wrap(rel.filter(str(expression)), con)

    def filter_rows(self, table: Any, mask: list[bool] | tuple[bool, ...]) -> Any:
        """Keep the rows a Python boolean mask selects.

        SQL has no notion of row position, so the mask is turned into a small
        table of keep-indices, registered, and joined against ``row_number()``.
        Only that index sidecar is materialised: the source table stays a plan.

        Parameters
        ----------
        table:
            The relation.
        mask:
            One boolean per row, aligned to current order.

        Returns
        -------
        DuckDBTable
            A filtered relation, sharing the connection.

        Raises
        ------
        ValueError
            If the mask length does not match the row count.
        RuntimeError
            If the handle carries no connection.

        Notes
        -----
        **``row_number()`` over an unordered relation is a stable numbering, not
        a meaningful one.** The mask must have been built from the same relation
        in the same state; against a re-planned or reordered relation it will
        select different rows without complaint.

        **Prefer :meth:`filter_expr` where the condition can be written as
        SQL.** A predicate is pushed into the scan; a mask requires the rows to
        be enumerated first.

        **There are two fallbacks, and both materialise the source**: an Arrow
        filter, then a pandas one. They run only when the SQL path fails.
        """
        n = int(self.n_rows(table))
        if len(mask) != n:
            raise ValueError(f"filter mask length {len(mask)} does not match table rows {n}")
        rel = _as_relation(table)
        con = _connection_of(table, None)
        if con is None:
            raise RuntimeError("DuckDB filter_rows requires a DuckDBTable with connection")
        keep_rn = [i + 1 for i, flag in enumerate(mask) if flag]
        if not keep_rn:
            self._last_pushdown = "filter_mask_empty"
            return _wrap(rel.limit(0), con)
        try:
            import pyarrow as pa

            # Only the keep-index sidecar is materialized: not the source table.
            idx_table = pa.table({"_bml_rn": pa.array(keep_rn, type=pa.int64())})
            con.register("_bml_keep_rn", idx_table)
            filtered = rel.query(
                "s",
                """
                SELECT * EXCLUDE (_bml_rn) FROM (
                  SELECT *, row_number() OVER () AS _bml_rn FROM s
                ) t
                WHERE _bml_rn IN (SELECT _bml_rn FROM _bml_keep_rn)
                """,
            )
            self._bridge = "sql"
            self._last_pushdown = "sql_mask"
            return _wrap(filtered, con)
        except Exception:  # noqa: BLE001
            pass
        # Honest fallback: full Arrow materialization of the source relation.
        try:
            import pyarrow as pa

            filtered = _relation_to_arrow_table(rel).filter(pa.array(list(mask), type=pa.bool_()))
            self._bridge = "arrow"
            self._last_pushdown = "arrow_filter_fallback"
            return _wrap(con.from_arrow(filtered), con)
        except Exception:  # noqa: BLE001
            pass
        frame = rel.df()
        frame = frame.loc[list(mask)].reset_index(drop=True)
        self._bridge = "pandas"
        self._last_pushdown = "pandas_filter_fallback"
        try:
            import pyarrow as pa

            return _wrap(
                con.from_arrow(pa.Table.from_pandas(frame, preserve_index=False)),
                con,
            )
        except Exception:  # noqa: BLE001
            return _wrap(con.from_df(frame), con)

    def aggregate(
        self,
        table: Any,
        aggregations: dict[str, str | list[str]],
        *,
        by: list[str] | None = None,
    ) -> Any:
        """Add a ``GROUP BY`` to the plan.

        Compiled to SQL and evaluated by DuckDB, which streams and spills to
        disk as needed: so aggregating a table larger than memory works, and
        only the small result comes back.

        Parameters
        ----------
        table:
            The relation.
        aggregations:
            Column to function name, or to a list of names. See
            :meth:`~buildml.data.engines.base.Engine.aggregate` for the
            supported set.
        by:
            Group-by columns. Omit for a single summary row.

        Returns
        -------
        DuckDBTable
            A relation over the aggregated result, sharing the connection.

        Raises
        ------
        ValidationError
            If a named column does not exist, or a function is not supported.
        RuntimeError
            If the handle carries no connection.

        Notes
        -----
        **Columns and functions are validated before the SQL is built**, so
        mistakes surface here rather than as a database error later.

        **Quantiles use ``quantile_cont``**, which can differ from pandas on
        ties.
        """
        from buildml.data.engines.aggregate import (
            normalize_aggregations,
            sql_aggregate_select,
            validate_aggregate_columns,
        )

        pairs = normalize_aggregations(aggregations)
        validate_aggregate_columns(self.columns(table), by, pairs)
        rel = _as_relation(table)
        con = _connection_of(table, None)
        if con is None:
            raise RuntimeError("DuckDB aggregate requires a DuckDBTable with connection")
        select_list = sql_aggregate_select(pairs, by=by)
        if by:
            group_list = ", ".join(f'"{c}"' for c in by)
            sql = f"SELECT {select_list} FROM s GROUP BY {group_list}"
        else:
            sql = f"SELECT {select_list} FROM s"
        self._last_pushdown = "aggregate"
        self._bridge = "sql"
        return _wrap(rel.query("s", sql), con)
