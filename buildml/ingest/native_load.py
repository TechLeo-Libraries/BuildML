"""Native-first path loaders for Polars and DuckDB (optional extras)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from buildml.core.errors import IngestError, MissingExtraError
from buildml.core.types import EngineName, SchemaField, TableSchema
from buildml.data.engines import get_engine


def load_native_path(
    path: Path,
    *,
    engine: EngineName,
    format_name: str,
    nrows: int | None = None,
    lazy: bool = False,
) -> tuple[Any, TableSchema, dict[str, Any]]:
    """Load a tabular path into an engine-native table without a Pandas-first pass.

    Parameters
    ----------
    path:
        Filesystem path.
    engine:
        ``polars`` or ``duckdb``.
    format_name:
        Detected format (``csv``, ``tsv``, ``parquet``, ``arrow``).
    nrows:
        Optional row cap (inspection aid).
    lazy:
        When True and Polars supports a scan path, keep a ``LazyFrame`` as the
        native handle (collect on Pandas / sklearn promotion). DuckDB path scans
        remain relation-backed. This is not out-of-core sklearn training.

    Returns
    -------
    tuple
        ``(native_table, schema, details)``.
    """
    if engine == EngineName.POLARS:
        return _load_polars(path, format_name=format_name, nrows=nrows, lazy=lazy)
    if engine == EngineName.DUCKDB:
        return _load_duckdb(path, format_name=format_name, nrows=nrows)
    raise IngestError(f"Native path load is not defined for engine '{engine.value}'")


def _load_polars(
    path: Path,
    *,
    format_name: str,
    nrows: int | None,
    lazy: bool,
) -> tuple[Any, TableSchema, dict[str, Any]]:
    try:
        import polars as pl
    except ImportError as exc:  # pragma: no cover
        raise MissingExtraError("polars", "Polars native ingest") from exc

    get_engine(EngineName.POLARS)
    details: dict[str, Any] = {
        "native_loader": "polars",
        "lazy_scan": False,
        "lazy_handle": False,
        "pandas_first": False,
    }
    try:
        if format_name in {"csv", "tsv"}:
            sep = "\t" if format_name == "tsv" else ","
            if lazy and nrows is None:
                details["lazy_scan"] = True
                details["lazy_handle"] = True
                table = pl.scan_csv(path, separator=sep)
            else:
                table = pl.read_csv(path, separator=sep, n_rows=nrows)
        elif format_name == "parquet":
            if nrows is not None:
                table = pl.read_parquet(path).head(nrows)
            elif lazy:
                details["lazy_scan"] = True
                details["lazy_handle"] = True
                table = pl.scan_parquet(path)
            else:
                table = pl.read_parquet(path)
        elif format_name == "arrow":
            if lazy and nrows is None:
                details["lazy_scan"] = True
                details["lazy_handle"] = True
                table = pl.scan_ipc(path)
            else:
                table = pl.read_ipc(path)
                if nrows is not None:
                    table = table.head(nrows)
        else:
            raise IngestError(
                f"Unsupported format '{format_name}' for Polars native ingest. "
                "Supported: csv, tsv, parquet, arrow/feather."
            )
    except IngestError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise IngestError(f"Polars native load failed for '{path}': {exc}") from exc

    if isinstance(table, pl.LazyFrame):
        schema_names = (
            list(table.collect_schema().names())
            if hasattr(table, "collect_schema")
            else list(table.schema.keys())
        )
        dtypes = (
            [str(dtype) for dtype in table.collect_schema().dtypes()]
            if hasattr(table, "collect_schema")
            else [str(table.schema[name]) for name in schema_names]
        )
        # Row count via a narrow collect; does not promote Dataset.frame.
        n_rows = int(table.select(pl.len()).collect().item())
        schema = TableSchema(
            fields=tuple(
                SchemaField(name=str(name), dtype=str(dtype), nullable=True)
                for name, dtype in zip(schema_names, dtypes, strict=True)
            )
        )
        details["n_rows"] = n_rows
        details["columns"] = schema_names
        return table, schema, details

    schema = TableSchema(
        fields=tuple(
            SchemaField(name=str(name), dtype=str(dtype), nullable=True)
            for name, dtype in zip(table.columns, table.dtypes, strict=True)
        )
    )
    details["n_rows"] = int(table.height)
    details["columns"] = list(table.columns)
    return table, schema, details


def _load_arrow_table(path: Path, *, nrows: int | None) -> tuple[Any, str]:
    """Load Feather/IPC via pyarrow without a Pandas bridge."""
    import pyarrow as pa
    import pyarrow.feather as feather
    import pyarrow.ipc as ipc

    try:
        table = feather.read_table(path)
        bridge = "pyarrow_feather"
    except Exception:  # noqa: BLE001
        try:
            with pa.memory_map(str(path), "r") as source:
                try:
                    table = ipc.open_file(source).read_all()
                except Exception:  # noqa: BLE001
                    source.seek(0)
                    table = ipc.open_stream(source).read_all()
            bridge = "pyarrow_ipc"
        except Exception as exc:  # noqa: BLE001
            raise IngestError(
                f"Arrow/IPC load failed for '{path}' via pyarrow: {exc}"
            ) from exc
    if nrows is not None:
        table = table.slice(0, int(nrows))
    return table, bridge


def _load_duckdb(
    path: Path,
    *,
    format_name: str,
    nrows: int | None,
) -> tuple[Any, TableSchema, dict[str, Any]]:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover
        raise MissingExtraError("duckdb", "DuckDB native ingest") from exc

    from buildml.data.engines.duckdb_engine import DuckDBTable

    con = duckdb.connect(database=":memory:")
    details: dict[str, Any] = {
        "native_loader": "duckdb",
        "lazy_scan": True,
        "pandas_first": False,
    }
    path_sql = str(path).replace("'", "''")
    try:
        if format_name in {"csv", "tsv"}:
            sep = "\\t" if format_name == "tsv" else ","
            rel = con.sql(
                f"SELECT * FROM read_csv_auto('{path_sql}', delim='{sep}', header=true)"
            )
        elif format_name == "parquet":
            rel = con.sql(f"SELECT * FROM read_parquet('{path_sql}')")
        elif format_name == "arrow":
            arrow_table, bridge = _load_arrow_table(path, nrows=nrows)
            rel = con.from_arrow(arrow_table)
            details["lazy_scan"] = False
            details["arrow_bridge"] = bridge
            details["pandas_bridge"] = False
        else:
            raise IngestError(
                f"Unsupported format '{format_name}' for DuckDB native ingest. "
                "Supported: csv, tsv, parquet, arrow/feather."
            )
        if nrows is not None and format_name != "arrow":
            rel = rel.limit(int(nrows))
    except IngestError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise IngestError(f"DuckDB native load failed for '{path}': {exc}") from exc

    # Wrap so the connection stays alive (relations no longer accept ad-hoc attrs).
    handle = DuckDBTable(relation=rel, connection=con)
    cols = [str(c) for c in getattr(rel, "columns", [])]
    if not cols:
        arrow_fn = getattr(rel, "arrow", None)
        if callable(arrow_fn):
            cols = [str(c) for c in arrow_fn().column_names]
        else:
            cols = list(rel.df().columns.astype(str))
    try:
        n_rows = int(rel.shape[0]) if getattr(rel, "shape", None) is not None else int(
            rel.count("*").fetchone()[0]
        )
    except Exception:  # noqa: BLE001
        arrow_fn = getattr(rel, "arrow", None)
        if callable(arrow_fn):
            n_rows = int(arrow_fn().num_rows)
        else:
            n_rows = int(len(rel.df()))
    type_map: dict[str, str] = {}
    try:
        types = list(rel.types)
        type_map = {str(c): str(t) for c, t in zip(cols, types, strict=False)}
    except Exception:  # noqa: BLE001
        type_map = {}
    schema = TableSchema(
        fields=tuple(
            SchemaField(name=name, dtype=type_map.get(name, "object"), nullable=True)
            for name in cols
        )
    )
    details["n_rows"] = n_rows
    details["columns"] = cols
    return handle, schema, details
