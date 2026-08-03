"""Read files straight into Polars or DuckDB, never through pandas first.

The obvious implementation of "load a file with Polars" is to read it with
pandas and convert. It also defeats the purpose: the pandas frame is fully
materialized before the conversion, so the memory ceiling that made you reach
for Polars is hit anyway, on the way in.

These loaders go directly to the engine's own reader. Polars' ``scan_*``
functions and DuckDB's SQL readers build a query plan instead of a frame, which
means a 10 GB parquet file can be filtered down to what you need before anything
is allocated. That is the difference between processing data larger than memory
and not.

What lazy does and does not buy you: prep — filtering, selecting, aggregating —
stays out of core, and the eventual training matrix still has to fit. scikit-learn
takes a NumPy array. Lazy loading lets you narrow a huge source down to a
trainable subset; it does not make scikit-learn out-of-core.

Both engines are optional extras, so a missing one raises a
:class:`~buildml.core.errors.MissingExtraError` naming the install.

See Also
--------
buildml.ingest.loaders : The pandas path.
buildml.data.engines : The engine adapters.
"""

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
    """Load a file into the engine's own table type, skipping pandas entirely.

    Dispatches to the Polars or DuckDB reader for the detected format. The
    returned handle is engine-native — a Polars ``DataFrame`` or ``LazyFrame``,
    or a DuckDB relation — and stays that way until something needs pandas.

    The schema is extracted without materializing the data. For a lazy Polars
    frame that means reading the file's metadata plus one narrow query for the
    row count; for DuckDB it comes off the relation. So you know the columns,
    the types, and the size before committing to load anything.

    Parameters
    ----------
    path:
        The file to read.
    engine:
        ``POLARS`` or ``DUCKDB``. Anything else is an error — pandas has its own
        loaders.
    format_name:
        The format from :func:`~buildml.ingest.detect.detect_path_format`:
        ``'csv'``, ``'tsv'``, ``'parquet'``, or ``'arrow'``.
    nrows:
        Cap the rows, for inspecting a large file. **Setting this disables lazy
        scanning** — a row cap needs a concrete frame — so leave it out when the
        point is to avoid materializing.
    lazy:
        Keep a lazy handle where the engine supports one. Polars uses ``scan_*``
        and returns a ``LazyFrame``; DuckDB relations are lazy regardless.

    Returns
    -------
    tuple
        ``(native_table, schema, details)``. The details dict records what
        actually happened — ``lazy_scan``, ``lazy_handle``, ``pandas_first``,
        ``n_rows``, ``columns`` — because the requested and actual strategies
        can differ.

    Raises
    ------
    MissingExtraError
        If the engine's package is not installed. The message names the extra.
    IngestError
        If the engine is neither Polars nor DuckDB, if the format is not
        supported by that engine, or if the read fails.

    Notes
    -----
    **Lazy means prep is out-of-core, not training.** A ``LazyFrame`` is
    collected when something asks for pandas or a NumPy array, and that
    collection has to fit in memory. Filter and select before you get there.

    **Nullability in the returned schema is not observed.** Every field is
    marked nullable, because checking would require scanning the data and defeat
    the point. Contrast :func:`~buildml.ingest.detect.schema_from_dataframe`,
    which does observe it.

    **Check ``details['lazy_handle']``** when it matters. A combination the
    engine cannot scan falls back to an eager read silently, and this is where
    that shows.

    See Also
    --------
    buildml.ingest.loaders : The pandas path.
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
