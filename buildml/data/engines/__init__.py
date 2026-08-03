"""Choose which backend does the prep work, without changing the calling code.

Three engines are supported. Pandas is always present and always in memory.
Polars and DuckDB are optional, and both can defer work: building a plan and
executing it once at the end, which lets them read only the columns and rows
that are actually needed.

The point of this package is that the choice is a configuration detail. Code
calls :func:`get_engine` and works against the
:class:`~buildml.data.engines.base.Engine` protocol; switching a Session from
pandas to DuckDB changes how prep executes and nothing about how it is written.

See Also
--------
buildml.data.dataset.Dataset : Where engines are used.
buildml.data.engines.base.Engine : The shared contract.
"""

from __future__ import annotations

from typing import Any

from buildml.core.errors import MissingExtraError, ValidationError
from buildml.core.types import EngineName
from buildml.data.engines.aggregate import SUPPORTED_AGG_FUNCS, normalize_aggregations
from buildml.data.engines.base import Engine
from buildml.data.engines.pandas_engine import PandasEngine
from buildml.data.engines.prep import MaterializePrepResult, prepare_design_frame
from buildml.ingest.detect import available_engines

# Cached adapters. DuckDBEngine construction no longer opens a connection;
# Dataset-owned DuckDBTable connections are reused across get_engine() calls.
_ADAPTER_CACHE: dict[EngineName, Engine] = {}


def get_engine(name: EngineName | str) -> Engine:
    """Look up the adapter for an engine, checking that it is installed.

    The single place engines are resolved. Adapters are cached per process, so
    calling this repeatedly is cheap and every caller shares one instance.

    Parameters
    ----------
    name:
        ``'pandas'``, ``'polars'``, or ``'duckdb'``, as a string or an
        :class:`~buildml.core.types.EngineName`.

    Returns
    -------
    Engine
        The adapter.

    Raises
    ------
    MissingExtraError
        If the optional package is not installed. The message names the extra
        to install.
    ValidationError
        If the name is not a known engine.

    Notes
    -----
    **Caching is safe because adapters hold no data.** They are stateless
    translators; the tables and, for DuckDB, the connections live on
    :class:`~buildml.data.engines.duckdb_engine.DuckDBTable` and
    :class:`~buildml.data.dataset.Dataset`.

    **Getting a DuckDB adapter opens no connection.** Only creating a table
    does.

    Examples
    --------
    Resolve and use an adapter::

        engine = get_engine("polars")
        table = engine.from_pandas(frame)

    See Also
    --------
    buildml.data.engines.base.Engine : What comes back.
    """
    engine_name = EngineName(name)
    cached = _ADAPTER_CACHE.get(engine_name)
    if cached is not None:
        return cached

    installed = available_engines()
    if engine_name not in installed:
        if engine_name == EngineName.POLARS:
            raise MissingExtraError("polars", "Polars engine")
        if engine_name == EngineName.DUCKDB:
            raise MissingExtraError("duckdb", "DuckDB engine")
        raise ValidationError(f"Engine '{engine_name.value}' is not available")

    if engine_name == EngineName.PANDAS:
        adapter: Engine = PandasEngine()
    elif engine_name == EngineName.POLARS:
        from buildml.data.engines.polars_engine import PolarsEngine

        adapter = PolarsEngine()
    elif engine_name == EngineName.DUCKDB:
        from buildml.data.engines.duckdb_engine import DuckDBEngine

        adapter = DuckDBEngine()
    else:
        raise ValidationError(f"Unsupported engine '{engine_name.value}'")
    _ADAPTER_CACHE[engine_name] = adapter
    return adapter


def engine_roundtrip_pandas(frame: Any, name: EngineName | str) -> Any:
    """Send a frame through an engine and back, to see what survives.

    A diagnostic. Converting into an engine and out again exposes the type
    changes that a switch would introduce: nullable integers, datetime
    resolutions, categoricals: before they show up as a confusing model error.

    Parameters
    ----------
    frame:
        A pandas DataFrame.
    name:
        Which engine to route through.

    Returns
    -------
    Any
        A DataFrame after the round trip. Compare its dtypes with the input.

    Raises
    ------
    ValidationError
        If ``frame`` is not a DataFrame.
    MissingExtraError
        If the engine is not installed.

    Notes
    -----
    **Not a no-op, and that is the point.** Values normally come back intact
    while dtypes may not, so a difference here is a real difference the pipeline
    would see.

    **Materialises the whole frame twice.** Use a sample for large data.

    See Also
    --------
    get_engine : Resolving the adapter.
    """
    import pandas as pd

    if not isinstance(frame, pd.DataFrame):
        raise ValidationError("engine_roundtrip_pandas expects a pandas.DataFrame")
    engine = get_engine(name)
    native = engine.from_pandas(frame)
    return engine.to_pandas(native)


__all__ = [
    "SUPPORTED_AGG_FUNCS",
    "Engine",
    "MaterializePrepResult",
    "PandasEngine",
    "engine_roundtrip_pandas",
    "get_engine",
    "normalize_aggregations",
    "prepare_design_frame",
]
