"""One call that inspects a source, decides how to read it, and says why.

The front door for getting data into BuildML. Detects the format, estimates the
scale, picks a mode and engine, loads, and returns both the dataset and a report
of every decision it made.

The report is not decoration. Ingest makes several choices on your behalf — this
file looked large, so lazy; Polars was not installed, so pandas — and each one
changes what happens later. A loader that made those choices silently would be a
loader you could not debug.

The one hard rule: a source that looks large is not loaded blindly into pandas.
Ingest refuses, explains, and offers four ways forward — force it, inspect it,
sample it, or install an engine that can stream it. That refusal is deliberate
friction, chosen because the alternative is a killed kernel with no message.

See Also
--------
buildml.ingest.detect : The detection and scale heuristics.
buildml.core.results.IngestReport : What the report contains.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from buildml.core.errors import IngestError, MissingExtraError
from buildml.core.results import IngestReport
from buildml.core.types import DataMode, EngineName, TableSchema, coerce_data_mode
from buildml.data.dataset import Dataset
from buildml.ingest import detect, loaders
from buildml.ingest.native_load import load_native_path


def ingest(
    source: pd.DataFrame | str | Path,
    *,
    mode: DataMode | str | None = None,
    engine: EngineName | str | None = None,
    dry_run: bool = False,
    mock_byte_estimate: int | None = None,
    read_nrows: int | None = None,
) -> tuple[Dataset | None, IngestReport]:
    """Load a source, choosing how based on what it is and how big it looks.

    Handles a DataFrame you already have or a path you want read. In both cases
    it produces a :class:`~buildml.data.dataset.Dataset` and a report recording
    the format, the schema, the size estimates, the recommended mode and engine,
    what was actually chosen, and every warning raised on the way.

    The refusal is the part worth knowing about. A path that looks large gets no
    blind pandas load — instead an :class:`~buildml.core.errors.IngestError`
    listing four ways forward: ``mode='memory'`` to insist, ``dry_run=True`` to
    inspect the schema without reading, ``read_nrows=N`` to sample, or install
    ``buildml[engines]`` and read it natively. Annoying once, better than a
    kernel that dies without saying why.

    When a native engine is used, no pandas frame is created at all. The Dataset
    holds an engine handle and promotes to pandas only when something requires
    it — preprocessing, a scikit-learn fit, or an explicit ``to_pandas``.

    Parameters
    ----------
    source:
        A DataFrame, or a path to CSV, TSV, Parquet, or Arrow.
    mode:
        Force ``'memory'`` or ``'lazy'`` instead of the recommendation.
        ``'memory'`` also overrides the large-file refusal, which is how you say
        "I know, load it anyway".
    engine:
        Force ``'pandas'``, ``'polars'``, or ``'duckdb'``. Naming an
        uninstalled engine raises rather than falling back, since a silent
        fallback here would defeat the reason for asking.
    dry_run:
        Return the report and no Dataset. The schema and size are still
        determined, so this is how you inspect a file too large to open.
    mock_byte_estimate:
        Pretend the source is this many bytes. For exercising the scale
        heuristics in tests without generating large files; recorded in the
        report so a mocked run is not mistaken for a real measurement.
    read_nrows:
        Read at most this many rows. **Disables native lazy scanning**, since a
        row cap requires a concrete frame.

    Returns
    -------
    tuple
        ``(Dataset | None, IngestReport)``. The dataset is ``None`` under
        ``dry_run``, and the report is always present.

    Raises
    ------
    IngestError
        If the path does not exist, if the format is unrecognised, if a large
        source would need a blind pandas load, or if the read fails.
    MissingExtraError
        If an explicitly requested engine is not installed.

    Notes
    -----
    **Read ``report.warnings``.** They record the decisions that differ from
    what you might assume — a pandas fallback in lazy mode, a native handle that
    will collect later, a DataFrame source that keeps its pandas cache
    regardless of mode.

    **Lazy mode is about prep, not training.** scikit-learn needs an in-memory
    design matrix, so lazy loading lets you narrow a huge source down to
    something trainable; it does not train out-of-core.

    **A DataFrame source is always pandas-backed.** Asking for lazy mode on a
    frame that is already in memory attaches a native handle but cannot undo the
    fact that the data is there.

    Examples
    --------
    Inspect a large file without loading it::

        _, report = ingest("data/events.parquet", dry_run=True)
        print(report.schema.columns())
        print(report.recommended_mode, report.recommended_engine)

    Read it natively once you know what is in it::

        dataset, report = ingest(
            "data/events.parquet", engine="polars", mode="lazy"
        )
        for note in report.warnings:
            print(note)

    See Also
    --------
    buildml.ingest.detect.recommend_mode : The scale heuristic behind the choice.
    buildml.data.dataset.Dataset : What you get back.
    """
    installed = detect.available_engines()
    warnings: list[str] = []
    path: Path | None = None
    native: Any | None = None
    native_details: dict[str, Any] = {}
    frame: pd.DataFrame | None = None
    schema: TableSchema
    row_estimate: int | None
    byte_estimate: int | None

    if isinstance(source, pd.DataFrame):
        frame = loaders.load_dataframe(source)
        source_type = "dataframe"
        format_name = "pandas.DataFrame"
        schema = detect.schema_from_dataframe(frame)
        row_estimate = int(len(frame))
        byte_estimate = mock_byte_estimate or detect.estimate_dataframe_bytes(frame)
    else:
        path = Path(source)
        if not path.exists():
            raise IngestError(f"Source path does not exist: {path}")
        format_name = detect.detect_path_format(path)
        source_type = "path"
        byte_estimate = mock_byte_estimate or detect.estimate_path_bytes(path)
        recommended_probe_mode = detect.recommend_mode(
            byte_estimate=byte_estimate,
            row_estimate=None,
        )
        looks_large = recommended_probe_mode != DataMode.MEMORY
        force_memory = mode is not None and coerce_data_mode(mode) == DataMode.MEMORY
        explicit_engine = EngineName(engine) if engine is not None else None
        explicit_mode = coerce_data_mode(mode) if mode is not None else None
        # Large sources need an explicit engine and/or lazy mode —
        # do not auto-load just because optional engines happen to be installed.
        engine_requests_native = explicit_engine in {
            EngineName.POLARS,
            EngineName.DUCKDB,
        }
        mode_requests_native = explicit_mode == DataMode.LAZY
        native_engine_available = (
            explicit_engine in installed
            if engine_requests_native
            else (EngineName.POLARS in installed or EngineName.DUCKDB in installed)
        )
        can_native_large = (
            looks_large
            and not force_memory
            and read_nrows is None
            and (engine_requests_native or mode_requests_native)
            and native_engine_available
        )

        if looks_large and not force_memory and read_nrows is None and not can_native_large:
            warnings.append(
                f"Source looks large ({byte_estimate} bytes estimated). "
                "Refusing blind full Pandas load. Pass mode='memory' to force, "
                "use dry_run=True to inspect, read_nrows=... to sample, "
                "or install buildml[engines] and use engine='polars'/'duckdb' "
                "(or mode='lazy') for native-first paths."
            )
            schema = TableSchema(fields=())
            row_estimate = None
            recommended_engine, engine_warnings = detect.recommend_engine(
                mode=recommended_probe_mode,
                installed=installed,
            )
            warnings.extend(engine_warnings)
            report = _make_report(
                source_type=source_type,
                format_name=format_name,
                schema=schema,
                row_estimate=row_estimate,
                byte_estimate=byte_estimate,
                mode=recommended_probe_mode,
                engine=recommended_engine,
                installed=installed,
                warnings=warnings,
                mock_byte_estimate=mock_byte_estimate,
                path=path,
            )
            if dry_run:
                return None, report
            raise IngestError(
                "Refusing to auto-load a large file into Pandas memory. "
                "Re-run with mode='memory' to force, dry_run=True to inspect only, "
                "read_nrows=N to sample, or install/use large engines "
                "(pip install 'buildml[engines]') with engine='polars' or 'duckdb'."
            )

        # Resolve mode/engine early for path loads so native paths can run.
        recommended_mode = recommended_probe_mode
        recommended_engine, engine_warnings = detect.recommend_engine(
            mode=recommended_mode,
            installed=installed,
        )
        warnings.extend(engine_warnings)
        chosen_mode = coerce_data_mode(mode) if mode is not None else recommended_mode
        chosen_engine = (
            EngineName(engine) if engine is not None else recommended_engine
        )
        if can_native_large and engine is None and chosen_engine == EngineName.PANDAS:
            # Prefer an installed large engine when scale heuristics say so.
            if EngineName.POLARS in installed:
                chosen_engine = EngineName.POLARS
            elif EngineName.DUCKDB in installed:
                chosen_engine = EngineName.DUCKDB

        use_native = chosen_engine in {EngineName.POLARS, EngineName.DUCKDB}
        if use_native:
            if chosen_engine == EngineName.POLARS and EngineName.POLARS not in installed:
                raise MissingExtraError("polars", "Polars engine ingest")
            if chosen_engine == EngineName.DUCKDB and EngineName.DUCKDB not in installed:
                raise MissingExtraError("duckdb", "DuckDB engine ingest")
            lazy_scan = chosen_mode == DataMode.LAZY
            native, schema, native_details = load_native_path(
                path,
                engine=chosen_engine,
                format_name=format_name,
                nrows=read_nrows,
                lazy=lazy_scan,
            )
            row_estimate = int(native_details.get("n_rows") or 0)
            warnings.append(
                f"Native-first ingest via {chosen_engine.value}: no Pandas-first load. "
                "Pandas cache promotes on preprocess, sklearn fit, or to_pandas()."
            )
            if native_details.get("lazy_handle"):
                warnings.append(
                    "Polars LazyFrame stored as Dataset.native; collect runs on "
                    "to_pandas()/sklearn materialization. This is not out-of-core "
                    "sklearn training."
                )
            elif native_details.get("lazy_scan"):
                warnings.append(
                    f"{chosen_engine.value} used a scan path; Dataset.native remains "
                    "an engine handle (relation or collected table), not a full "
                    "out-of-core sklearn fit path."
                )
        else:
            frame = _load_path(path, format_name=format_name, nrows=read_nrows)
            schema = detect.schema_from_dataframe(frame)
            row_estimate = int(len(frame))
            if mock_byte_estimate is None:
                byte_estimate = detect.estimate_dataframe_bytes(frame)

        # Recompute recommendations with row estimate when available.
        recommended_mode = detect.recommend_mode(
            byte_estimate=byte_estimate,
            row_estimate=row_estimate,
        )
        recommended_engine, more_engine_warnings = detect.recommend_engine(
            mode=recommended_mode,
            installed=installed,
        )
        for tip in more_engine_warnings:
            if tip not in warnings:
                warnings.append(tip)
        if mode is None:
            if use_native:
                # Preserve lazy labels when native-first load ran.
                chosen_mode = (
                    recommended_probe_mode
                    if recommended_probe_mode != DataMode.MEMORY
                    else recommended_mode
                )
            else:
                # Pandas path loads are in-memory even when scale tips say lazy.
                chosen_mode = DataMode.MEMORY

        report = _finalize_path_ingest(
            source_type=source_type,
            format_name=format_name,
            schema=schema,
            row_estimate=row_estimate,
            byte_estimate=byte_estimate,
            recommended_mode=recommended_mode,
            recommended_engine=recommended_engine,
            chosen_mode=chosen_mode,
            chosen_engine=chosen_engine,
            installed=installed,
            warnings=warnings,
            mock_byte_estimate=mock_byte_estimate,
            path=path,
            dry_run=dry_run,
            frame=frame,
            native=native,
            native_details=native_details,
        )
        return report

    # DataFrame source path (always Pandas-backed; optional native attach).
    recommended_mode = detect.recommend_mode(
        byte_estimate=byte_estimate,
        row_estimate=row_estimate,
    )
    recommended_engine, engine_warnings = detect.recommend_engine(
        mode=recommended_mode,
        installed=installed,
    )
    warnings.extend(engine_warnings)

    chosen_mode = coerce_data_mode(mode) if mode is not None else recommended_mode
    chosen_engine = EngineName(engine) if engine is not None else recommended_engine

    if chosen_mode != DataMode.MEMORY and chosen_engine == EngineName.PANDAS:
        warnings.append(
            "Requested/recommended non-memory mode, but DataFrame ingest keeps a "
            "Pandas cache; attach Polars/DuckDB via engine= for native handles."
        )

    if chosen_engine == EngineName.POLARS and EngineName.POLARS not in installed:
        raise MissingExtraError("polars", "Polars engine ingest")
    if chosen_engine == EngineName.DUCKDB and EngineName.DUCKDB not in installed:
        raise MissingExtraError("duckdb", "DuckDB engine ingest")

    if chosen_engine != EngineName.PANDAS:
        warnings.append(
            f"Engine '{chosen_engine.value}' selected; Dataset keeps a Pandas cache "
            "for sklearn materialization and attaches a native handle for "
            "project/filter/sample before to_pandas."
        )

    report = _make_report(
        source_type=source_type,
        format_name=format_name,
        schema=schema,
        row_estimate=row_estimate,
        byte_estimate=byte_estimate,
        mode=recommended_mode,
        engine=recommended_engine,
        installed=installed,
        warnings=warnings,
        mock_byte_estimate=mock_byte_estimate,
        path=path,
        chosen_mode=chosen_mode,
        chosen_engine=chosen_engine,
    )

    if dry_run:
        return None, report

    attach = chosen_engine != EngineName.PANDAS
    dataset = Dataset.from_pandas(
        frame,
        schema=schema,
        mode=chosen_mode if attach else DataMode.MEMORY,
        engine=chosen_engine,
        source="dataframe",
        attach_native=attach,
    )
    return dataset, report


def _finalize_path_ingest(
    *,
    source_type: str,
    format_name: str,
    schema: TableSchema,
    row_estimate: int | None,
    byte_estimate: int | None,
    recommended_mode: DataMode,
    recommended_engine: EngineName,
    chosen_mode: DataMode,
    chosen_engine: EngineName,
    installed: tuple[EngineName, ...],
    warnings: list[str],
    mock_byte_estimate: int | None,
    path: Path,
    dry_run: bool,
    frame: pd.DataFrame | None,
    native: Any | None,
    native_details: dict[str, Any],
) -> tuple[Dataset | None, IngestReport]:
    if chosen_mode != DataMode.MEMORY and chosen_engine == EngineName.PANDAS:
        warnings.append(
            "Requested/recommended non-memory mode, but Pandas path loads "
            "materialize an in-memory DataFrame for the returned Dataset."
        )

    report = _make_report(
        source_type=source_type,
        format_name=format_name,
        schema=schema,
        row_estimate=row_estimate,
        byte_estimate=byte_estimate,
        mode=recommended_mode,
        engine=recommended_engine,
        installed=installed,
        warnings=warnings,
        mock_byte_estimate=mock_byte_estimate,
        path=path,
        chosen_mode=chosen_mode,
        chosen_engine=chosen_engine,
        native_details=native_details,
    )
    if dry_run:
        return None, report

    if native is not None and chosen_engine != EngineName.PANDAS:
        # Defer Pandas promotion; keep native handle (eager or LazyFrame).
        materialize = chosen_mode == DataMode.MEMORY
        dataset = Dataset.from_native(
            native,
            engine=chosen_engine,
            schema=schema,
            mode=chosen_mode,
            source=str(path),
            materialize_pandas=materialize,
        )
        return dataset, report

    assert frame is not None
    dataset = Dataset.from_pandas(
        frame,
        schema=schema,
        mode=DataMode.MEMORY,
        engine=EngineName.PANDAS,
        source=str(path),
    )
    return dataset, report


def _load_path(path: Path, *, format_name: str, nrows: int | None) -> pd.DataFrame:
    if format_name in {"csv", "tsv"}:
        return loaders.load_csv(path, nrows=nrows)
    if format_name == "parquet":
        return loaders.load_parquet(path)
    if format_name == "arrow":
        return loaders.load_arrow(path)
    raise IngestError(
        f"Unsupported or unknown file format for '{path}'. "
        "Supported: csv, tsv, parquet, arrow/feather."
    )


def _make_report(
    *,
    source_type: str,
    format_name: str,
    schema: TableSchema,
    row_estimate: int | None,
    byte_estimate: int | None,
    mode: DataMode,
    engine: EngineName,
    installed: tuple[EngineName, ...],
    warnings: list[str],
    mock_byte_estimate: int | None,
    path: Path | None,
    chosen_mode: DataMode | None = None,
    chosen_engine: EngineName | None = None,
    native_details: dict[str, Any] | None = None,
) -> IngestReport:
    details: dict[str, Any] = detect.build_scale_details(
        byte_estimate=byte_estimate,
        row_estimate=row_estimate,
        mocked_bytes=mock_byte_estimate,
    )
    if path is not None:
        details["path"] = str(path)
    if chosen_mode is not None:
        details["chosen_mode"] = chosen_mode.value
    if chosen_engine is not None:
        details["chosen_engine"] = chosen_engine.value
    if native_details:
        details["native_load"] = dict(native_details)
    return IngestReport(
        source_type=source_type,
        format_name=format_name,
        schema=schema,
        row_estimate=row_estimate,
        byte_estimate=byte_estimate,
        recommended_mode=mode,
        recommended_engine=engine,
        available_engines=installed,
        warnings=list(warnings),
        details=details,
    )
