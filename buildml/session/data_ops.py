"""Ingest, roles, splits, engines, and checkpoint orchestration."""

from __future__ import annotations

from buildml.session._imports import *  # noqa: F403


def close_native(session) -> None:
    """Close an owned DuckDB connection on the session dataset, if any.

    Safe to call when no dataset is attached or the engine is not DuckDB.
    Derived Datasets that share a connection are not owners; only the root
    handle closes the connection.
    """
    dataset = session._dataset
    if dataset is None:
        return
    closer = getattr(dataset, "close_native", None)
    if callable(closer):
        closer()


def ingest_session(
    session_cls,
    source: pd.DataFrame | str | Path,
    *,
    mode: DataMode | str | None = None,
    engine: EngineName | str | None = None,
    dry_run: bool = False,
    mock_byte_estimate: int | None = None,
    read_nrows: int | None = None,
) -> Session:
    """Create a session by ingesting a tabular source.

    Parameters
    ----------
    source:
        DataFrame or path to CSV/Parquet/Arrow.
    mode:
        Optional data-mode override.
    engine:
        Optional engine override.
    dry_run:
        If True, build a session with report only (no dataset) when the
        pipeline does not materialize data.
    mock_byte_estimate:
        Optional scale override for tests/heuristics.
    read_nrows:
        Optional CSV row cap.

    Returns
    -------
    Session
        Session containing dataset and/or ingest report.

    Notes
    -----
    **Scale:** Large paths are not silently loaded into Pandas. Use
    ``dry_run=True``, ``read_nrows``, ``mode='memory'`` (force), or engine
    extras.

    **Leakage:** Call :meth:`split` before fit-capable operations. Use
    :meth:`assert_can_fit` to enforce train-only fit scope.
    """
    dataset, report = ingest_source(
        source,
        mode=mode,
        engine=engine,
        dry_run=dry_run,
        mock_byte_estimate=mock_byte_estimate,
        read_nrows=read_nrows,
    )
    session = session_cls(dataset=dataset, ingest_report=report)
    session._record(
        "ingest",
        {
            "source_type": report.source_type,
            "format": report.format_name,
            "mode": report.recommended_mode.value if mode is None else str(mode),
            "engine": report.recommended_engine.value if engine is None else str(engine),
            "dry_run": dry_run,
            "read_nrows": read_nrows,
        },
        decision_origin="automatic" if mode is None and engine is None else "explicit",
        warnings=report.warnings,
    )
    return session


def set_roles(session, mapping: dict[str, str | ColumnRole]) -> Session:
    """Assign column roles on the current dataset.

    Parameters
    ----------
    mapping:
        Column → role mapping.

    Returns
    -------
    Session
        ``self`` for fluent chaining.
    """
    session.dataset.set_roles(mapping)
    session._record(
        "set_roles",
        {
            "mapping": {
                name: role.value if isinstance(role, ColumnRole) else str(role)
                for name, role in mapping.items()
            }
        },
    )
    return session


def split(
    session,
    *,
    test_size: float | int = 0.2,
    validation_size: float | int | None = None,
    random_state: int | None = 42,
    stratify: bool = False,
) -> Session:
    """Create a train/test (optional validation) split.

    Parameters
    ----------
    test_size:
        Test fraction or count.
    validation_size:
        Optional validation fraction/count from the train pool.
    random_state:
        RNG seed.
    stratify:
        If True, stratify on the target role column.

    Notes
    -----
    **Leakage:** After splitting, fit-capable operations must use the train
    partition only (:meth:`assert_can_fit`).
    """
    session._split_plan = create_split(
        session.dataset,
        test_size=test_size,
        validation_size=validation_size,
        random_state=random_state,
        stratify=stratify,
    )
    session._record(
        "split",
        {
            "kind": session._split_plan.kind,
            "test_size": test_size,
            "validation_size": validation_size,
            "stratify": stratify,
        },
    )
    return session


def inject_split(
    session,
    *,
    train_indices: list[int] | tuple[int, ...],
    test_indices: list[int] | tuple[int, ...],
    validation_indices: list[int] | tuple[int, ...] | None = None,
) -> Session:
    """Inject externally defined partition indices.

    Parameters
    ----------
    train_indices / test_indices / validation_indices:
        Positional indices into the current dataset.
    """
    session._split_plan = inject_partitions(
        session.dataset,
        train_indices=train_indices,
        test_indices=test_indices,
        validation_indices=validation_indices,
    )
    session._record(
        "inject_split",
        {
            "train_indices": list(train_indices),
            "test_indices": list(test_indices),
            "validation_indices": None if validation_indices is None else list(validation_indices),
            "kind": "injected",
        },
    )
    return session


def group_split(
    session,
    *,
    test_size: float | int = 0.2,
    validation_size: float | int | None = None,
    random_state: int | None = 42,
    group_column: str | None = None,
) -> Session:
    """Create a group-aware train/test(/validation) split.

    No group identifier appears in more than one partition. Sizes are
    interpreted over groups, not rows.

    Parameters
    ----------
    test_size / validation_size:
        Fraction or count of groups.
    random_state:
        RNG seed.
    group_column:
        Optional override; defaults to the sole ``group`` role column.

    Notes
    -----
    **Leakage:** Prefer this over :meth:`split` when rows share entities
    (customers, sites, documents). Random row splits leak across groups.
    """
    session._split_plan = create_group_split(
        session.dataset,
        test_size=test_size,
        validation_size=validation_size,
        random_state=random_state,
        group_column=group_column,
    )
    session._record(
        "group_split",
        {
            "kind": session._split_plan.kind,
            "test_size": test_size,
            "validation_size": validation_size,
            "group_column": session._split_plan.stratify_column,
        },
    )
    return session


def time_split(
    session,
    *,
    test_size: float | int = 0.2,
    validation_size: float | int | None = None,
    time_column: str | None = None,
) -> Session:
    """Create a chronological train/test(/validation) split.

    Rows are ordered by the time-role column. The latest rows form test;
    optional validation is carved from the end of the remaining pool.

    Parameters
    ----------
    test_size / validation_size:
        Fraction or absolute row count after time ordering.
    time_column:
        Optional override; defaults to the sole ``time`` role column.

    Notes
    -----
    **Leakage:** Prefer this over shuffled splits for temporal processes.
    The splitter does not add a calendar embargo beyond strict ordering.
    """
    session._split_plan = create_time_split(
        session.dataset,
        test_size=test_size,
        validation_size=validation_size,
        time_column=time_column,
    )
    session._record(
        "time_split",
        {
            "kind": session._split_plan.kind,
            "test_size": test_size,
            "validation_size": validation_size,
            "time_column": session._split_plan.stratify_column,
        },
    )
    return session


def partition(
    session, name: PartitionName | Literal['train', 'validation', 'test']
) -> pd.DataFrame:
    """Return a copy of rows for a named partition.

    Raises
    ------
    ValidationError
        If no split exists.
    """
    if session._split_plan is None:
        raise ValidationError("No split defined. Call split(...) or inject_split(...) first.")
    frame = frame_for_partition(session.dataset, session._split_plan, name)
    session._record(
        "partition",
        {"name": str(name)},
        result_summary={"name": str(name), "rows": int(len(frame))},
    )
    return frame


def assert_can_fit(session, partition: PartitionName = "train") -> Session:
    """Enforce leakage-safe fit scope.

    Parameters
    ----------
    partition:
        Partition the caller intends to fit on (must be ``train``).

    Raises
    ------
    LeakageError
        If no split exists or partition is not train.
    """
    assert_fit_partition(session._split_plan, partition)
    return session


def to_engine(session, engine: EngineName | str | None = None) -> Any:
    """Materialize the current dataset in a selected engine's native type.

    Parameters
    ----------
    engine:
        Target engine. Defaults to the dataset's current engine setting.
    """
    native = session.dataset.to_engine(engine)
    selected = session.dataset.engine if engine is None else EngineName(engine)
    session._record(
        "to_engine",
        {"engine": selected.value},
        result_summary={"engine": selected.value, "native_type": type(native).__name__},
    )
    return native


def checkpoint_save(
    session,
    path: str | Path,
    *,
    sidecar_partition_rows: int | None = None,
    sidecar_compression: str | None = None,
    sidecar_layout: str | None = None,
) -> Path:
    """Save a resumable checkpoint bundle for mid-loop exit.

    Parameters
    ----------
    path:
        Destination directory.
    sidecar_partition_rows:
        Optional rows-per-partition for native sidecars (default 25_000).
        Ignored when ``sidecar_layout='single'``.
    sidecar_compression:
        Optional Parquet compression for native sidecars (default ``zstd``).
    sidecar_layout:
        ``'auto'`` (default; partition at ≥50_000 rows), ``'single'``, or
        ``'partitioned'``.
    """
    before = prior_state(session._history)
    sidecar_params = {
        "sidecar_partition_rows": sidecar_partition_rows,
        "sidecar_compression": sidecar_compression,
        "sidecar_layout": sidecar_layout,
    }
    record = make_operation_record(
        sequence=len(session._history) + 1,
        operation_id="checkpoint_save",
        parameters={"path": str(path), **sidecar_params},
        decision_origin="explicit",
        before=before,
        after=session_state(session),
        result_summary={"path": str(Path(path))},
    )
    destination = save_checkpoint(
        path,
        dataset=session.dataset,
        split_plan=session._split_plan,
        history=[*session._history, record],
        plans=session._plan_objects(),
        sidecar_partition_rows=sidecar_partition_rows,
        sidecar_compression=sidecar_compression,
        sidecar_layout=sidecar_layout,
    )
    record["parameters"] = {"path": str(destination), **sidecar_params}
    record["details"] = {"path": str(destination)}
    record["result_summary"] = {
        "path": str(destination),
        "plans_present": [
            key for key, value in session._plan_objects().items() if value is not None
        ],
    }
    session._history.append(record)
    return destination


def checkpoint_load_session(session_cls, path: str | Path, *, data_only: bool = False) -> Session:
    """Load a checkpoint bundle and validate reattach conditions.

    Parameters
    ----------
    path:
        Checkpoint directory.
    data_only:
        If True, ignore metadata and treat data as a fresh ingest.

    Notes
    -----
    When ``plans.joblib`` is present, preprocess plan objects are restored
    for mid-loop resume. Checkpoints still do not embed a fitted estimator;
    use :meth:`load_pipeline` for inference artifacts.
    """
    loaded = load_checkpoint(path, data_only=data_only)
    session = session_cls(
        dataset=loaded.dataset,
        split_plan=loaded.split_plan,
        history=loaded.history,
        reattach_result=loaded.reattach,
    )
    if not data_only:
        session._restore_plans(loaded.plans)
    session._record(
        "checkpoint_load",
        {
            "path": str(path),
            "status": loaded.reattach.status,
            "data_only": data_only,
            "plans_restored": sorted(
                (key for key, value in loaded.plans.items() if value is not None)
            ),
        },
    )
    return session


def reattach(session, path: str | Path, *, data_only: bool = False) -> Session:
    """Replace this session state from a checkpoint path (instance helper)."""
    loaded = load_checkpoint(path, data_only=data_only)
    session.close_native()
    session._dataset = loaded.dataset
    session._split_plan = loaded.split_plan
    session._history = list(loaded.history)
    session._reattach_result = loaded.reattach
    session._ingest_report = None
    if data_only:
        session._clear_plans()
    else:
        session._restore_plans(loaded.plans)
    session._record(
        "reattach",
        {
            "path": str(path),
            "status": loaded.reattach.status,
            "data_only": data_only,
            "plans_restored": sorted(
                (key for key, value in loaded.plans.items() if value is not None)
            ),
        },
    )
    return session


def to_pandas(session) -> pd.DataFrame:
    """Escape hatch: copy the current dataset as a Pandas DataFrame."""
    frame = session.dataset.to_pandas()
    session._record(
        "to_pandas", result_summary={"rows": int(len(frame)), "columns": int(frame.shape[1])}
    )
    return frame


def to_parquet(session, path: str | Path) -> Path:
    """Write the current dataset to Parquet."""
    destination = session.dataset.to_parquet(path)
    session._record(
        "to_parquet", {"path": str(destination)}, result_summary={"path": str(destination)}
    )
    return destination


def head(session, n: int = 5) -> pd.DataFrame:
    """Preview the first rows."""
    frame = session.dataset.head(n)
    session._record(
        "head", {"n": n}, result_summary={"rows": int(len(frame)), "columns": int(frame.shape[1])}
    )
    return frame


def with_mode(session, mode: DataMode | str) -> Session:
    """Record a mode override on the dataset metadata.

    Accepted values are ``memory`` and ``lazy``. Legacy ``out_of_core`` is
    coerced to ``lazy`` (there is no separate out-of-core fit mode).
    """
    session.dataset.mode = coerce_data_mode(mode)
    session._record("with_mode", {"mode": session.dataset.mode.value})
    return session


def with_engine(session, engine: EngineName | str) -> Session:
    """Select a compute engine and attach a native handle when applicable.

    Parameters
    ----------
    engine:
        ``pandas``, ``polars``, or ``duckdb``.

    Notes
    -----
    Polars/DuckDB attach a persistent ``Dataset.native`` table used by
    :meth:`prepare_design_matrix`, :meth:`~buildml.data.dataset.Dataset.project`,
    and sample/filter helpers before Pandas materialization. Sklearn fit
    still requires an in-memory design matrix. Missing extras raise
    :class:`~buildml.core.errors.MissingExtraError`.
    """
    from buildml.data.engines import get_engine

    chosen = EngineName(engine)
    get_engine(chosen)
    session.dataset.engine = chosen
    if chosen == EngineName.PANDAS:
        session.dataset.clear_native()
    else:
        session.dataset.attach_native(rebuild=True)
    session._record(
        "with_engine", {"engine": chosen.value, "has_native": session.dataset.has_native}
    )
    return session


def sync_native(session) -> Session:
    """Rebuild ``Dataset.native`` from the current Pandas frame (eager).

    Session preprocess transforms already sync when ``engine`` is Polars or
    DuckDB. Call this after external Pandas mutation of ``dataset.frame``,
    or after a transform that opted out of sync. This is not a lazy plan
    of prior steps — it converts the full current frame into the engine
    table.
    """
    has_native = False
    if session.dataset.engine != EngineName.PANDAS:
        session.dataset.sync_native()
        has_native = session.dataset.has_native
    session._record(
        "sync_native", {"engine": session.dataset.engine.value, "has_native": has_native}
    )
    return session


def metadata(session) -> dict[str, Any]:
    """Session/dataset metadata snapshot."""
    payload: dict[str, Any] = {
        "has_dataset": session._dataset is not None,
        "ingest_report": None
        if session._ingest_report is None
        else session._ingest_report.to_dict(),
        "split": None if session._split_plan is None else session._split_plan.to_dict(),
        "history": session.history,
        "reattach": None
        if session._reattach_result is None
        else {
            "status": session._reattach_result.status,
            "messages": list(session._reattach_result.messages),
        },
    }
    if session._dataset is not None:
        payload["dataset"] = session._dataset.metadata()
    return payload
