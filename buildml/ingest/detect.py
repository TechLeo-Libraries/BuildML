"""Work out what a file is, how big it is, and whether it will fit in memory.

Before BuildML reads anything it decides three things: the format, the scale,
and consequently which engine and mode to use. Getting that wrong is the
difference between a load that takes two seconds and one that fills the machine's
memory and takes the notebook kernel with it.

The scale checks here exist because pandas will cheerfully attempt something
impossible. Ask for a frame larger than available memory and you do not get an
error, you get swapping, then an out-of-memory kill with no traceback and no
indication of which line caused it. The gates turn that into a warning you can
read, before the allocation happens.

Two thresholds, with different intent. The soft limit — 250 MiB by default —
warns and continues, because the estimate is approximate and the user may know
better. The hard limit refuses, and is off unless you configure it, since a
library that refuses to load your data by default is a library you stop using.
Set ``BUILDML_MATERIALIZATION_HARD_LIMIT_BYTES`` in a shared or automated
environment where an out-of-memory kill is worse than a failed job.

See Also
--------
buildml.ingest.pipeline : Where these decisions are applied.
buildml.core.types : The ``DataMode`` and ``EngineName`` vocabulary.
"""

from __future__ import annotations

import importlib.util
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from buildml.core.errors import ValidationError
from buildml.core.types import DataMode, EngineName, SchemaField, TableSchema

# Heuristic thresholds (bytes). Documented; always overridable by the user.
MEMORY_SOFT_LIMIT = 250 * 1024 * 1024  # 250 MiB
LAZY_SOFT_LIMIT = 2 * 1024 * 1024 * 1024  # 2 GiB
# Opt-in hard refuse default when BUILDML_MATERIALIZATION_HARD_LIMIT_BYTES is set,
# or when callers pass hard_limit_bytes explicitly. None means no hard refuse.
MEMORY_HARD_LIMIT: int | None = None
_ENV_HARD = os.environ.get("BUILDML_MATERIALIZATION_HARD_LIMIT_BYTES")
if _ENV_HARD is not None and str(_ENV_HARD).strip():
    try:
        MEMORY_HARD_LIMIT = int(_ENV_HARD)
    except ValueError:
        MEMORY_HARD_LIMIT = None


@dataclass(slots=True)
class MaterializationTelemetry:
    """What a memory check found, and what it decided to do about it.

    Returned by :func:`check_materialization` whether or not anything was
    exceeded, so a caller can record the measurement rather than only the alarm.
    Knowing a fit ran at 40 MiB is useful the day it starts running at 400.

    Attributes
    ----------
    context:
        Where the check happened — ``'estimator fit'``, ``'preprocess'``. Names
        the boundary in warnings, which is what makes them actionable.
    nbytes:
        Estimated footprint. Approximate; see the notes.
    soft_limit_bytes:
        The warn threshold in force for this check.
    hard_limit_bytes:
        The refuse threshold, or ``None`` when no hard limit is configured.
    soft_exceeded:
        Whether the estimate reached the soft limit.
    hard_exceeded:
        Whether it reached the hard limit. Always ``False`` when there is none.
    warnings:
        The messages generated. Empty when nothing was exceeded.
    guidance:
        Concrete suggestions — use a lazy engine, narrow the columns. Populated
        only when a limit was exceeded, so it is advice at the moment it is
        needed rather than boilerplate on every check.

    Notes
    -----
    **The estimate is of the frame, not of the operation.** It comes from
    ``memory_usage(deep=True)``, which measures the frame as it stands. An
    operation that copies, or that converts to a NumPy array, may need two or
    three times this. Treat the number as a lower bound.

    See Also
    --------
    check_materialization : What produces this.
    """

    context: str
    nbytes: int
    soft_limit_bytes: int
    hard_limit_bytes: int | None
    soft_exceeded: bool
    hard_exceeded: bool
    warnings: list[str] = field(default_factory=list)
    guidance: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Flatten for history and reports, adding a human-readable size.

        ``nbytes_mib`` is derived rather than stored, because a raw byte count
        in a report is something the reader has to divide by 1,048,576 in their
        head before it means anything.

        Returns
        -------
        dict
            The telemetry fields plus ``nbytes_mib``, the estimate in mebibytes
            to three decimal places. Lists are copied so the caller cannot
            mutate the telemetry through them.

        See Also
        --------
        check_materialization : What produces this.
        """
        return {
            "context": self.context,
            "nbytes": self.nbytes,
            "nbytes_mib": round(self.nbytes / (1024 * 1024), 3),
            "soft_limit_bytes": self.soft_limit_bytes,
            "hard_limit_bytes": self.hard_limit_bytes,
            "soft_exceeded": self.soft_exceeded,
            "hard_exceeded": self.hard_exceeded,
            "warnings": list(self.warnings),
            "guidance": list(self.guidance),
        }


def schema_from_dataframe(frame: pd.DataFrame) -> TableSchema:
    """Describe a frame's columns as a schema BuildML can carry forward.

    Turns "what pandas happens to hold right now" into an explicit record of
    column names, dtypes, and nullability — the thing a score-time contract is
    later checked against.

    Nullability is observed, not declared. A column is marked nullable because
    this frame has a missing value in it, not because the source says it could.
    That is the useful reading at fit time and a trap at score time: a training
    column that happened to be complete will be marked non-nullable, and the
    first production batch with a gap in it will look like a schema change.

    Parameters
    ----------
    frame:
        The DataFrame to describe. Column names are stringified, so an integer
        column label becomes ``'0'``.

    Returns
    -------
    TableSchema
        Fields in column order, each with name, dtype string, and observed
        nullability.

    Notes
    -----
    **Dtypes are recorded as strings**, ``'int64'`` and ``'object'`` rather than
    NumPy objects, so a schema survives being written to JSON and read back
    somewhere else.

    **This scans every column for missing values**, which is a full pass over
    the frame. Cheap for the sizes ingest handles, not free.

    See Also
    --------
    buildml.core.types.TableSchema : The structure returned.
    buildml.pipeline.contract : Enforcing a schema at score time.
    """
    fields = tuple(
        SchemaField(
            name=str(col),
            dtype=str(frame[col].dtype),
            nullable=bool(frame[col].isna().any()),
        )
        for col in frame.columns
    )
    return TableSchema(fields=fields)


def detect_path_format(path: Path) -> str:
    """Guess the format from the file extension, and admit when it cannot.

    Extension-based, not content-based. Reading magic bytes would be more
    reliable, and it would also mean opening every candidate file during a
    directory scan; the extension is right almost always, and ``'unknown'`` is
    an honest answer when it is not.

    Parameters
    ----------
    path:
        The file path. Only the suffix is examined; the file need not exist.

    Returns
    -------
    str
        ``'csv'``, ``'tsv'``, ``'parquet'``, ``'arrow'``, or ``'unknown'``.
        Several extensions map to one format — ``.pq`` is parquet, and
        ``.feather``, ``.arrow``, and ``.ipc`` are all arrow.

    Notes
    -----
    **A misnamed file is detected wrong.** A parquet file called ``data.csv``
    reports ``'csv'`` and fails later with a parser error rather than here.

    **Compressed extensions are not handled.** ``data.csv.gz`` has suffix
    ``.gz`` and reports ``'unknown'``, even though pandas would read it.

    Examples
    --------
    >>> from pathlib import Path
    >>> detect_path_format(Path("sales.csv"))
    'csv'
    >>> detect_path_format(Path("events.pq"))
    'parquet'
    >>> detect_path_format(Path("notes.txt"))
    'unknown'
    """
    suffix = path.suffix.lower()
    mapping = {
        ".csv": "csv",
        ".tsv": "tsv",
        ".parquet": "parquet",
        ".pq": "parquet",
        ".feather": "arrow",
        ".arrow": "arrow",
        ".ipc": "arrow",
    }
    return mapping.get(suffix, "unknown")


def estimate_path_bytes(path: Path) -> int | None:
    """Ask the filesystem how big a file is, returning ``None`` if it cannot say.

    The cheap first estimate of scale, used before deciding whether to load
    eagerly or lazily. Any filesystem error — missing file, no permission, a
    path that is not a file — becomes ``None`` rather than an exception, because
    the caller's next step is to fall back to a row-count estimate, not to give
    up.

    Parameters
    ----------
    path:
        The file to measure.

    Returns
    -------
    int or None
        Size in bytes on disk, or ``None`` when it could not be determined.

    Notes
    -----
    **On-disk size is not memory size, and the gap is large.** Parquet is
    columnar and compressed, so a 100 MB file can expand past a gigabyte once
    read. CSV usually shrinks — text digits become 8-byte floats, but repeated
    strings become categories. Neither direction is predictable enough to scale
    by a constant, which is why the mode heuristic is conservative.

    See Also
    --------
    estimate_dataframe_bytes : Measuring the frame once it is loaded.
    """
    try:
        return path.stat().st_size
    except OSError:
        return None


def estimate_dataframe_bytes(frame: pd.DataFrame) -> int:
    """Measure what a frame currently occupies, following object references.

    ``deep=True`` is the whole point. Without it, an object column reports 8
    bytes per row — the size of the pointers — and a frame of a million strings
    appears to weigh 8 MB when it actually weighs several hundred. Deep
    accounting is slower, since it walks every Python object, and it is the only
    version whose answer is worth acting on.

    Parameters
    ----------
    frame:
        The DataFrame to measure.

    Returns
    -------
    int
        Estimated bytes, including the index and the contents of object columns.

    Notes
    -----
    **This is what the frame occupies, not what an operation on it will need.**
    A copy doubles it; converting to a NumPy array for scikit-learn can more
    than double it, since a mixed-dtype frame becomes one uniformly-typed block.
    Budget several times this figure at a materialization boundary.

    **Shared strings are counted once per reference.** A column of repeated
    categories may report more than it truly costs, since Python interns and
    shares many of those objects.

    See Also
    --------
    check_materialization : Comparing this against a limit.
    """
    return int(frame.memory_usage(deep=True).sum())


def check_materialization(
    frame: pd.DataFrame,
    *,
    context: str,
    soft_limit_bytes: int = MEMORY_SOFT_LIMIT,
    hard_limit_bytes: int | None = MEMORY_HARD_LIMIT,
    on_soft: Literal["warn", "ignore"] = "warn",
    on_hard: Literal["error", "warn"] = "error",
) -> MaterializationTelemetry:
    """Check a frame against the memory limits before something copies it.

    Called at materialization boundaries — the moments where a lazy frame stops
    being a query plan and becomes real memory, most often when handing a design
    matrix to scikit-learn. That is the point where a job either proceeds or
    dies, and it is worth measuring.

    Two thresholds with different jobs. The soft limit warns: the estimate is
    approximate, the machine might have plenty of room, and refusing would be
    presumptuous. The hard limit refuses, and does not exist unless someone
    configures it — the right setting for a shared cluster or a nightly job,
    where an out-of-memory kill costs more than a clear failure.

    Parameters
    ----------
    frame:
        The frame about to be copied or converted.
    context:
        A short label naming the boundary, such as ``'estimator fit'``. It goes
        into the warning, and a warning that says where it came from is one
        someone can act on.
    soft_limit_bytes:
        Warn at or above this. Defaults to 250 MiB, chosen to be well under a
        typical laptop's headroom while allowing for the copies that follow.
    hard_limit_bytes:
        Refuse at or above this. Defaults to
        ``BUILDML_MATERIALIZATION_HARD_LIMIT_BYTES`` if set, otherwise ``None``
        for no hard limit.
    on_soft:
        ``'warn'`` to emit a warning, ``'ignore'`` to record it in the telemetry
        silently — for a caller that already knows the frame is large and does
        not want the noise on every batch.
    on_hard:
        ``'error'`` to raise, ``'warn'`` to record and continue.

    Returns
    -------
    MaterializationTelemetry
        The measurement and the verdict, returned whether or not any limit was
        exceeded.

    Raises
    ------
    ValidationError
        When the hard limit is exceeded and ``on_hard='error'``. The message
        names the size, the limit, and what to do about it.

    Notes
    -----
    **The estimate is a lower bound.** It measures the frame, and the operation
    about to happen may need several times that.

    **The right fix is usually upstream.** Narrow the columns, filter the rows,
    or keep prep lazy with Polars or DuckDB and materialize only the training
    matrix. Raising the limit treats the symptom.

    Examples
    --------
    ::

        telemetry = check_materialization(
            train_frame,
            context="estimator fit",
            hard_limit_bytes=2 * 1024**3,
        )
        history.append(telemetry.to_dict())

    See Also
    --------
    warn_if_large_materialization : The same check when only the messages matter.
    """
    nbytes = estimate_dataframe_bytes(frame)
    soft_exceeded = nbytes >= soft_limit_bytes
    hard_exceeded = hard_limit_bytes is not None and nbytes >= hard_limit_bytes
    notes: list[str] = []
    guidance = [
        "Prefer engine='polars' or 'duckdb' with mode='lazy' for prep.",
        "Materialize only the train design matrix needed for fit/predict, not the full prep frame.",
        "Set hard_limit_bytes or BUILDML_MATERIALIZATION_HARD_LIMIT_BYTES "
        "to refuse oversized copies.",
    ]
    mb = nbytes / (1024 * 1024)
    if soft_exceeded:
        soft_mb = soft_limit_bytes / (1024 * 1024)
        message = (
            f"Materializing ~{mb:.1f} MiB for {context} exceeds the "
            f"{soft_mb:.0f} MiB soft limit. Prefer lazy prep, then "
            "materialize only the train design matrix. Soft gates disclose risk; "
            "they do not refuse unless a hard limit is also configured."
        )
        notes.append(message)
        if on_soft == "warn":
            warnings.warn(message, UserWarning, stacklevel=3)
    if hard_exceeded:
        assert hard_limit_bytes is not None
        hard_mb = hard_limit_bytes / (1024 * 1024)
        hard_message = (
            f"Materializing ~{mb:.1f} MiB for {context} exceeds the "
            f"{hard_mb:.0f} MiB hard limit. Narrow columns/rows, use a lazy "
            "engine for prep, or raise hard_limit_bytes explicitly if the copy "
            "is intentional."
        )
        notes.append(hard_message)
        if on_hard == "error":
            raise ValidationError(hard_message)
        warnings.warn(hard_message, UserWarning, stacklevel=3)
    return MaterializationTelemetry(
        context=context,
        nbytes=nbytes,
        soft_limit_bytes=soft_limit_bytes,
        hard_limit_bytes=hard_limit_bytes,
        soft_exceeded=soft_exceeded,
        hard_exceeded=hard_exceeded,
        warnings=notes,
        guidance=guidance if (soft_exceeded or hard_exceeded) else [],
    )


def warn_if_large_materialization(
    frame: pd.DataFrame,
    *,
    context: str,
    soft_limit_bytes: int = MEMORY_SOFT_LIMIT,
    hard_limit_bytes: int | None = MEMORY_HARD_LIMIT,
) -> list[str]:
    """Run the memory check and return just the messages.

    A thin wrapper over :func:`check_materialization` for callers that want to
    append warnings to a report and do not need the numbers. Same behaviour:
    warns on soft exceedance, raises on hard exceedance when one is configured.

    Parameters
    ----------
    frame:
        The frame about to be materialized.
    context:
        A short label naming the boundary, used in the messages.
    soft_limit_bytes:
        Warn at or above this.
    hard_limit_bytes:
        Refuse at or above this. ``None`` means no hard limit.

    Returns
    -------
    list of str
        The warning messages. Empty when nothing was exceeded, which is the
        common case and the reason this is safe to call unconditionally.

    Raises
    ------
    ValidationError
        When the hard limit is exceeded. This wrapper does not expose
        ``on_hard``, so a configured hard limit always refuses here.

    Notes
    -----
    **Use :func:`check_materialization` when the size matters**, not just the
    alarm. Recording ``nbytes`` on every run is what lets you see a job growing
    before it fails.

    See Also
    --------
    check_materialization : The full telemetry.
    """
    telemetry = check_materialization(
        frame,
        context=context,
        soft_limit_bytes=soft_limit_bytes,
        hard_limit_bytes=hard_limit_bytes,
    )
    return list(telemetry.warnings)


def available_engines() -> tuple[EngineName, ...]:
    """Report which dataframe engines are actually installed.

    Polars and DuckDB are optional extras, so what BuildML can do depends on
    what the environment has. This checks by looking for the module spec rather
    than importing, which is fast and has no side effects — importing Polars to
    find out whether Polars is installed costs a noticeable fraction of a second
    and pulls a large library into memory that may never be used.

    Returns
    -------
    tuple of EngineName
        The available engines. Always contains ``PANDAS``, which is a hard
        dependency; ``POLARS`` and ``DUCKDB`` appear when importable.

    Notes
    -----
    **Importable is not the same as working.** A broken installation — a
    corrupt wheel, a missing native library — has a module spec and fails on
    import. Rare, and it surfaces at load time rather than here.

    **Install both with** ``pip install 'buildml[engines]'``. They are the
    difference between processing data that fits in memory and data that does
    not.

    See Also
    --------
    recommend_engine : Choosing among these for a given mode.
    """
    found: list[EngineName] = [EngineName.PANDAS]
    if importlib.util.find_spec("polars") is not None:
        found.append(EngineName.POLARS)
    if importlib.util.find_spec("duckdb") is not None:
        found.append(EngineName.DUCKDB)
    return tuple(found)


def recommend_mode(
    *,
    byte_estimate: int | None,
    row_estimate: int | None,
) -> DataMode:
    """Decide between loading everything and streaming, from a size estimate.

    Memory mode loads the frame and keeps it; lazy mode builds a query plan and
    materializes only what an operation needs. Memory is simpler and faster when
    the data fits, and lazy is the only option when it does not.

    Bytes are preferred over rows when both are available, because a million
    rows of two integers and a million rows of two hundred text columns are
    different problems. When only a row count is known, five million is the
    fallback trigger.

    Parameters
    ----------
    byte_estimate:
        Estimated size in bytes, or ``None`` if unknown.
    row_estimate:
        Estimated row count, or ``None`` if unknown.

    Returns
    -------
    DataMode
        ``LAZY`` when the data looks large, ``MEMORY`` otherwise. With no
        information at all, ``MEMORY`` — the assumption that an unmeasurable
        source is small, which is right far more often than not.

    Notes
    -----
    **This is a recommendation and can be overridden.** An explicit ``mode``
    passed to ingest wins; the heuristic only fills a gap.

    **The threshold accounts for what comes next.** 250 MiB of data is not a
    problem on a modern machine, but 250 MiB that gets copied twice during
    preprocessing and once more into a NumPy array is.

    Examples
    --------
    >>> from buildml.core.types import DataMode
    >>> recommend_mode(byte_estimate=1024, row_estimate=None) is DataMode.MEMORY
    True
    >>> recommend_mode(byte_estimate=10**9, row_estimate=None) is DataMode.LAZY
    True
    >>> recommend_mode(byte_estimate=None, row_estimate=10_000_000) is DataMode.LAZY
    True

    See Also
    --------
    recommend_engine : Choosing the engine once the mode is known.
    """
    if byte_estimate is None:
        if row_estimate is not None and row_estimate >= 5_000_000:
            return DataMode.LAZY
        return DataMode.MEMORY
    if byte_estimate >= MEMORY_SOFT_LIMIT:
        return DataMode.LAZY
    return DataMode.MEMORY


def recommend_engine(
    *,
    mode: DataMode,
    installed: tuple[EngineName, ...],
) -> tuple[EngineName, list[str]]:
    """Pick the engine for a mode, and say so when the best one is missing.

    Memory mode always gets pandas: the ecosystem is built around it, and
    nothing is gained by using something else on data that fits.

    Lazy mode wants Polars first, DuckDB second. Both do lazy evaluation well;
    Polars is preferred because its dataframe semantics are closer to pandas, so
    the behaviour a caller expects carries over. DuckDB is the SQL-shaped
    alternative and equally capable.

    When lazy mode is wanted and neither is installed, this falls back to pandas
    with a warning rather than failing. Continuing on a best effort beats
    refusing to load data at all — but the warning matters, because pandas in
    lazy mode is pandas, and the memory problem that prompted lazy mode is still
    there.

    Parameters
    ----------
    mode:
        The chosen data mode.
    installed:
        What is available, from :func:`available_engines`.

    Returns
    -------
    tuple
        ``(EngineName, list[str])`` — the engine, and any warnings to surface.
        The list is empty on the normal paths.

    Notes
    -----
    **Read the warnings.** A silent pandas fallback in lazy mode means the scale
    strategy is not in effect, and the failure it was meant to prevent is still
    ahead.

    Examples
    --------
    >>> from buildml.core.types import DataMode, EngineName
    >>> engine, notes = recommend_engine(
    ...     mode=DataMode.MEMORY, installed=(EngineName.PANDAS,)
    ... )
    >>> engine is EngineName.PANDAS, notes
    (True, [])
    >>> engine, notes = recommend_engine(
    ...     mode=DataMode.LAZY, installed=(EngineName.PANDAS,)
    ... )
    >>> engine is EngineName.PANDAS, len(notes)
    (True, 1)

    See Also
    --------
    available_engines : What to pass as ``installed``.
    """
    warnings: list[str] = []
    if mode == DataMode.MEMORY:
        return EngineName.PANDAS, warnings

    if EngineName.POLARS in installed:
        return EngineName.POLARS, warnings
    if EngineName.DUCKDB in installed:
        return EngineName.DUCKDB, warnings

    warnings.append(
        "Large/lazy mode recommended, but neither Polars nor DuckDB is installed. "
        "Install with: pip install 'buildml[engines]'. Continuing with Pandas when safe."
    )
    return EngineName.PANDAS, warnings


def build_scale_details(
    *,
    byte_estimate: int | None,
    row_estimate: int | None,
    mocked_bytes: int | None = None,
) -> dict[str, Any]:
    """Package the scale numbers and the thresholds they were judged against.

    An estimate on its own is not interpretable. "412 MiB" means nothing until
    you know the soft limit is 250; recording both is what makes an ingest
    report readable later, including by someone whose environment has different
    limits configured.

    Parameters
    ----------
    byte_estimate:
        Estimated size in bytes, or ``None``.
    row_estimate:
        Estimated row count, or ``None``.
    mocked_bytes:
        A size injected for testing, recorded so a test-run report is not
        mistaken for a measurement of real data.

    Returns
    -------
    dict
        The estimates, the three thresholds in force, and a guidance sentence
        about lazy prep and hard limits.

    Notes
    -----
    **The thresholds are read at call time**, so a report reflects the limits
    that were actually in force — including one set through the environment.

    See Also
    --------
    buildml.core.results.IngestReport : Where this lands.
    """
    return {
        "memory_soft_limit_bytes": MEMORY_SOFT_LIMIT,
        "lazy_soft_limit_bytes": LAZY_SOFT_LIMIT,
        "memory_hard_limit_bytes": MEMORY_HARD_LIMIT,
        "byte_estimate": byte_estimate,
        "row_estimate": row_estimate,
        "mocked_bytes": mocked_bytes,
        "materialization_guidance": (
            "Soft gates warn near 250 MiB; configure "
            "BUILDML_MATERIALIZATION_HARD_LIMIT_BYTES or hard_limit_bytes to refuse. "
            "Keep prep lazy when possible and materialize only the train design matrix."
        ),
    }
