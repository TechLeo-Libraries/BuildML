"""DuckDB engine adapter (optional extra)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from buildml.core.errors import MissingExtraError
from buildml.core.types import EngineName


class DuckDBTable:
    """DuckDB relation plus connection keepalive.

    Newer DuckDB relation objects reject ad-hoc attributes, so the connection
    that backs a native handle is stored here instead.

    Ownership
    ---------
    The connection is typically owned by a root :class:`~buildml.data.dataset.Dataset`
    (``_owns_native_connection=True``). Derived project/filter handles share the
    same connection without taking ownership. Call ``Dataset.close_native()`` on
    the owner when finished so tests and long sessions do not leak connections.
    Adapters from :func:`~buildml.data.engines.get_engine` do not open a
    connection until ``from_pandas`` / ``from_parquet`` / ``from_arrow`` need one.
    """

    __slots__ = ("relation", "connection")

    def __init__(self, relation: Any, connection: Any) -> None:
        object.__setattr__(self, "relation", relation)
        object.__setattr__(self, "connection", connection)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.relation, name)


def close_duckdb_connection(connection: Any) -> None:
    """Close a DuckDB connection if it is still open (best-effort)."""
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
    """Adapter that moves tables through DuckDB relations/SQL.

    Prefer relation/SQL ops (project, filter, sample, read_parquet) before
    materializing full tables to Arrow/Pandas. Arrow remains the interchange
    bridge when a full collect is unavoidable.

    Connection lifecycle
    --------------------
    Construction does **not** open a DuckDB connection. Each
    ``from_pandas`` / ``from_parquet`` / ``from_arrow`` call creates a connection
    owned by the returned :class:`DuckDBTable` unless an existing ``connection``
    is supplied for reuse. Relation ops reuse ``DuckDBTable.connection`` so
    repeated :func:`~buildml.data.engines.get_engine` calls do not spawn
    adapter-owned connections.
    """

    name = EngineName.DUCKDB

    def __init__(self) -> None:
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
        """Attach a PyArrow table as a DuckDB relation (no Pandas)."""
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
        """Attach a Parquet file or directory as a DuckDB relation.

        Parameters
        ----------
        lazy:
            Accepted for API parity with Polars. DuckDB always returns a
            relation over ``read_parquet``; the flag only records intent.
        connection:
            Optional existing connection to reuse (Dataset reattach / rebuild).
        compression:
            Ignored on read; accepted for call-site symmetry with writers.
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
        """Write a DuckDB relation to Parquet without a Pandas round-trip."""
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
        rel = _as_relation(table)
        try:
            return _relation_to_arrow_table(rel).to_pandas()
        except Exception:  # noqa: BLE001
            return rel.df()

    def to_arrow(self, table: Any) -> Any:
        """Materialize a DuckDB relation as a PyArrow table when supported."""
        return _relation_to_arrow_table(_as_relation(table))

    def n_rows(self, table: Any) -> int:
        rel = _as_relation(table)
        shape = getattr(rel, "shape", None)
        if shape is not None:
            return int(shape[0])
        return int(rel.count("*").fetchone()[0])

    def columns(self, table: Any) -> list[str]:
        rel = _as_relation(table)
        cols = getattr(rel, "columns", None)
        if cols is not None:
            return [str(c) for c in cols]
        try:
            return [str(c) for c in _relation_to_arrow_table(rel).column_names]
        except Exception:  # noqa: BLE001
            return list(rel.df().columns.astype(str))

    def head(self, table: Any, n: int = 5) -> pd.DataFrame:
        rel = _as_relation(table)
        limited = rel.limit(n)
        try:
            return _relation_to_arrow_table(limited).to_pandas()
        except Exception:  # noqa: BLE001
            return limited.df()

    def select_columns(self, table: Any, columns: list[str]) -> Any:
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
        """Sample rows via DuckDB SQL/relation ops before full Arrow collect.

        Unseeded samples prefer ``USING SAMPLE`` on the relation. Seeded samples
        use a deterministic hash order + ``LIMIT`` (still relation-native). Full
        Arrow materialization is a last-resort fallback only.
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
        """Push a SQL predicate into the relation (no full-table Arrow collect)."""
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
        """Keep rows where ``mask`` is True without materializing the source table first.

        Registers a small mask index table and filters via ``row_number()`` join
        SQL. Falls back to Arrow ``Table.filter`` only when the SQL path fails.
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

            # Only the keep-index sidecar is materialized — not the source table.
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
        """Push group aggregations into DuckDB SQL before Arrow collect."""
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
