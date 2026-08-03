"""Read a file into pandas, and turn any failure into one recognisable error.

Thin wrappers over ``pandas.read_*``, and the thinness is the point — pandas
already knows how to read these formats. What these add is a uniform failure
mode: every underlying exception, whatever library raised it, becomes an
:class:`~buildml.core.errors.IngestError` naming the path.

Without that, a caller has to catch ``ParserError``, ``ArrowInvalid``,
``UnicodeDecodeError``, ``FileNotFoundError``, and whatever the parquet engine
happens to raise this version — a list that changes with dependencies. The
original exception is chained, so the specific cause is still there when you
need it.

See Also
--------
buildml.ingest.native_load : Reading into Polars or DuckDB instead.
buildml.ingest.pipeline : The dispatcher that calls these.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from buildml.core.errors import IngestError


def load_dataframe(source: pd.DataFrame) -> pd.DataFrame:
    """Copy a caller's frame, so later work cannot modify what they passed in.

    The copy is the whole function. Someone who hands a frame to BuildML and
    keeps using it in the next notebook cell should not find that BuildML has
    changed it underneath them — and preprocessing does modify frames.

    Parameters
    ----------
    source:
        The DataFrame to adopt.

    Returns
    -------
    pandas.DataFrame
        An independent copy.

    Raises
    ------
    IngestError
        If ``source`` is not a DataFrame. Caught here because the alternative is
        an ``AttributeError`` several layers down, with nothing pointing back to
        the wrong argument.

    Notes
    -----
    **This doubles memory for the duration.** Both frames exist at once during
    the copy, so a 2 GiB frame briefly needs 4. It is the right default anyway;
    aliasing bugs are worse than a transient allocation.

    **The copy is shallow for object columns.** The frame's structure is
    independent, but a cell holding a mutable object still points at the same
    object. Rare in tabular data, and worth knowing.
    """
    if not isinstance(source, pd.DataFrame):
        raise IngestError(f"Expected a pandas.DataFrame, got {type(source)!r}")
    return source.copy()


def load_csv(path: Path, *, nrows: int | None = None) -> pd.DataFrame:
    """Read a delimited text file, choosing the separator from the extension.

    A ``.tsv`` file is read with tabs and everything else with commas. That is
    the only inference done here; pandas handles the rest, including dtype
    guessing, quoting, and encoding.

    Parameters
    ----------
    path:
        The file to read. Only the suffix is inspected, for the separator.
    nrows:
        Stop after this many rows. Useful for inspecting the shape of a large
        file cheaply — a header and a hundred rows tell you the columns without
        reading a gigabyte.

    Returns
    -------
    pandas.DataFrame
        The parsed data, with dtypes inferred by pandas.

    Raises
    ------
    IngestError
        On any failure — missing file, malformed rows, an encoding pandas
        cannot decode. The original exception is chained.

    Notes
    -----
    **Dtypes are inferred per column and can surprise you.** An ID column of
    zero-padded numbers becomes an integer and loses the padding; a column that
    is numeric except for one ``"N/A"`` becomes object. Fix these after loading
    rather than fighting the parser.

    **With ``nrows``, inference sees only those rows.** A column that is
    integers for the first hundred rows and text thereafter will be typed from
    the sample and mistyped for the file.

    See Also
    --------
    load_parquet : A format that stores its own dtypes, avoiding all of this.
    """
    try:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        return pd.read_csv(path, sep=sep, nrows=nrows)
    except Exception as exc:  # noqa: BLE001 - surface as ingest error
        raise IngestError(f"Failed to load CSV from '{path}': {exc}") from exc


def load_parquet(path: Path) -> pd.DataFrame:
    """Read a Parquet file, dtypes and all, with no guessing needed.

    Parquet stores its schema, so nothing is inferred: an integer column comes
    back as an integer, a timestamp as a timestamp, a category as a category.
    It is columnar and compressed, which usually makes it both smaller on disk
    and faster to read than the equivalent CSV.

    Parameters
    ----------
    path:
        The file to read. May also be a directory of parts, which pandas reads
        as one dataset.

    Returns
    -------
    pandas.DataFrame
        The data, with the dtypes recorded in the file.

    Raises
    ------
    IngestError
        On any failure, including a missing parquet engine. Chained to the
        original.

    Notes
    -----
    **Requires pyarrow or fastparquet.** Neither is in pandas' base install,
    and the resulting error is the most common failure here.

    **There is no ``nrows``.** Parquet is columnar, so reading fewer rows does
    not save proportionally; the practical saving comes from reading fewer
    columns.

    **Compressed on disk expands in memory, sometimes greatly.** A 100 MB file
    can become a multi-gigabyte frame, which is why the scale check in
    :mod:`buildml.ingest.detect` does not scale file size by a constant.

    See Also
    --------
    load_csv : Text, with inference.
    buildml.ingest.detect.estimate_path_bytes : Why file size is a weak signal.
    """
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        raise IngestError(f"Failed to load Parquet from '{path}': {exc}") from exc


def load_arrow(path: Path) -> pd.DataFrame:
    """Read an Arrow IPC or Feather file, the fastest of the three formats.

    Arrow's on-disk layout is its in-memory layout, so reading is close to a
    memory map rather than a parse. That makes it the right choice for
    intermediate artifacts — a cached frame between pipeline stages — where the
    file is written and read by the same tooling and speed matters more than
    portability.

    Parameters
    ----------
    path:
        The file to read. ``.feather``, ``.arrow``, and ``.ipc`` are all this
        format.

    Returns
    -------
    pandas.DataFrame
        The data, with dtypes from the file's schema.

    Raises
    ------
    IngestError
        On any failure, including a missing pyarrow. Chained to the original.

    Notes
    -----
    **Requires pyarrow.**

    **Poor archival choice.** The format has changed across Arrow versions, and
    it does not compress as parquet does. Use parquet for anything meant to be
    read next year or by someone else.

    See Also
    --------
    load_parquet : The portable columnar format.
    """
    try:
        return pd.read_feather(path)
    except Exception as exc:  # noqa: BLE001
        raise IngestError(f"Failed to load Arrow/Feather from '{path}': {exc}") from exc
