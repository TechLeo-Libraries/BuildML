"""Core loaders for in-memory ingest paths."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from buildml.core.errors import IngestError


def load_dataframe(source: pd.DataFrame) -> pd.DataFrame:
    """Return a defensive copy of a Pandas DataFrame."""
    if not isinstance(source, pd.DataFrame):
        raise IngestError(f"Expected a pandas.DataFrame, got {type(source)!r}")
    return source.copy()


def load_csv(path: Path, *, nrows: int | None = None) -> pd.DataFrame:
    """Load a CSV/TSV file into a DataFrame."""
    try:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        return pd.read_csv(path, sep=sep, nrows=nrows)
    except Exception as exc:  # noqa: BLE001 - surface as ingest error
        raise IngestError(f"Failed to load CSV from '{path}': {exc}") from exc


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a Parquet file into a DataFrame."""
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        raise IngestError(f"Failed to load Parquet from '{path}': {exc}") from exc


def load_arrow(path: Path) -> pd.DataFrame:
    """Load an Arrow IPC/Feather file into a DataFrame."""
    try:
        return pd.read_feather(path)
    except Exception as exc:  # noqa: BLE001
        raise IngestError(f"Failed to load Arrow/Feather from '{path}': {exc}") from exc
