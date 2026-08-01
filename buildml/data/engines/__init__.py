"""Compute engine adapters."""

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
    """Return an engine adapter, raising MissingExtraError when unavailable.

    Parameters
    ----------
    name:
        Engine name (``pandas``, ``polars``, or ``duckdb``).

    Notes
    -----
    Adapters are cached per process. DuckDB connections are **not** owned by the
    adapter — they live on :class:`~buildml.data.engines.duckdb_engine.DuckDBTable`
    / :class:`~buildml.data.dataset.Dataset` (see Dataset connection ownership).
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
    """Convert a Pandas frame through an engine and back (smoke utility)."""
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
