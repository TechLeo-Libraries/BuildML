"""Save a session mid-analysis and pick it up later, or on another machine.

A checkpoint is a directory holding everything needed to resume: the data, the
column roles, the split membership, the operation history, and any preprocessing
plans fitted so far. It exists because analysis is rarely finished in one
sitting, and because the split is the one thing that must not be regenerated —
a fresh split reshuffles which rows are held out, and every score computed
before and after becomes incomparable.

What a checkpoint deliberately does *not* contain is a fitted estimator. That is
the job of a pipeline bundle, and the distinction matters: a checkpoint is for
resuming work, a pipeline bundle is for serving predictions. Save both when you
need both — neither embeds the other.

The layout is a plain directory of Parquet and JSON, readable without BuildML.
Engine-native query plans cannot be serialised, so a Polars or DuckDB table is
snapshotted to Parquet and reattached on load.

See Also
--------
buildml.pipeline : Bundles that carry a fitted model for inference.
buildml.checkpoint.validate : Deciding whether a checkpoint may be reattached.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd

from buildml._version import __version__
from buildml.checkpoint.validate import ReattachResult, validate_reattach
from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole, DataMode, EngineName, TableSchema, coerce_data_mode
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.explain.history import HISTORY_SCHEMA_VERSION, normalize_history
from buildml.ingest.detect import schema_from_dataframe

SidecarLayout = Literal["auto", "single", "partitioned"]

PIPELINE_COMPATIBILITY = (
    "Checkpoints and pipeline bundles are complementary, not interchangeable. "
    "A checkpoint restores data, roles, splits, history, and optional preprocess "
    "plan objects for mid-loop resume. It does not embed a fitted estimator. "
    "A pipeline bundle restores fitted plans plus the estimator and model card "
    "for inference. Neither artifact embeds the other; store them side by side "
    "when both resume and deployment are required. Optional plans.joblib uses "
    "buildml.plans.v2 (flat legacy plan dicts remain readable)."
)

NATIVE_SIDECAR_RELATIVE = "data/native_sidecar.parquet"
NATIVE_SIDECAR_DIR_RELATIVE = "data/native_sidecar"
NATIVE_SIDECAR_LIMITS = (
    "Engine-native query plans are not serialized. A sidecar stores a Parquet "
    "snapshot of the native table at save time (zstd compression by default; "
    "large frames may use a partitioned directory layout). Restore with "
    "lazy_intent reattaches Polars via scan_parquet (a new scan plan over those "
    "bytes) or DuckDB via read_parquet (file-backed relation). Older single-file "
    "sidecars remain readable. Sklearn still requires an in-memory design matrix "
    "at the estimator boundary."
)
# Partition when row count is large enough that a single Parquet blob is awkward.
SIDECAR_PARTITION_ROW_THRESHOLD = 50_000
SIDECAR_ROWS_PER_PARTITION = 25_000
SIDECAR_DEFAULT_COMPRESSION = "zstd"


@dataclass(slots=True)
class LoadedCheckpoint:
    """Everything a checkpoint restored, including how far it could be trusted.

    The important field is ``reattach``. A checkpoint can load cleanly and still
    not be safe to continue from, because the data on disk may no longer match
    what the checkpoint was written against. That verdict travels with the
    contents rather than being raised as an error, so the caller can decide
    whether a downgrade to fresh ingest is acceptable.

    Attributes
    ----------
    dataset:
        The restored data with its roles and engine reattached where possible.
    split_plan:
        The original partition membership, or ``None`` if reattach refused it.
        This is the field a resume exists to preserve.
    history:
        The operation log, normalised to the current schema version.
    reattach:
        The verdict — clean resume, degraded, fresh ingest, or blocked — with the
        messages explaining it.
    meta, manifest:
        The saved metadata and integrity record, or ``None`` under
        ``data_only``.
    plans:
        Fitted preprocessing plans, so imputation values and encodings carry
        over instead of being refitted on different data.

    Notes
    -----
    **A ``None`` split plan after a successful load is the case to handle.** It
    means the checkpoint's partitions could not be applied to the current data,
    so any new split will be a different one and scores will not be comparable
    with those from before the save.

    See Also
    --------
    load_checkpoint : Producing this.
    buildml.checkpoint.validate.ReattachResult : The verdict in detail.
    """

    dataset: Dataset
    split_plan: SplitPlan | None
    history: list[dict[str, Any]]
    reattach: ReattachResult
    meta: dict[str, Any] | None
    manifest: dict[str, Any] | None
    plans: dict[str, Any] = field(default_factory=dict)


def save_checkpoint(
    path: str | Path,
    *,
    dataset: Dataset,
    split_plan: SplitPlan | None = None,
    history: list[dict[str, Any]] | None = None,
    plans: dict[str, Any] | None = None,
    sidecar_partition_rows: int | None = None,
    sidecar_compression: str | None = None,
    sidecar_layout: SidecarLayout | str | None = None,
) -> Path:
    """Write the session to a directory so the same work can be resumed later.

    Saves the data, roles, split membership, history, and any fitted
    preprocessing plans. The split is the reason to bother: regenerating one
    reshuffles the holdout, and scores from before and after a resume then
    describe different experiments.

    Layout
    ------
    ``data/frame.parquet`` (canonical), optional native sidecar
    (``data/native_sidecar.parquet`` or partitioned ``data/native_sidecar/``)
    for Polars/DuckDB reattach, ``meta.json``, ``splits.json``, ``history.json``,
    optional ``plans.joblib``, ``MANIFEST.json``.

    Parameters
    ----------
    path:
        Destination directory (created if missing).
    dataset:
        Current dataset.
    split_plan:
        Optional split membership.
    history:
        Optional operation history entries.
    plans:
        Optional Session preprocess plan objects (impute/encode/scale/dates/
        outliers/binning/feature_select/resample). Stored for mid-loop resume;
        not a substitute for a pipeline bundle's estimator.
    sidecar_partition_rows:
        Rows written per partition file when the sidecar layout is partitioned.
        Defaults to ``SIDECAR_ROWS_PER_PARTITION`` (25_000). Ignored for a
        forced ``single`` layout. Must be a positive integer when provided.
    sidecar_compression:
        Parquet compression codec for the native sidecar. Defaults to
        ``SIDECAR_DEFAULT_COMPRESSION`` (``zstd``). Passed through to the
        engine / PyArrow writers.
    sidecar_layout:
        ``'auto'`` (default): single-file below
        ``SIDECAR_PARTITION_ROW_THRESHOLD`` (50_000 rows), partitioned at or
        above that threshold. ``'single'`` / ``'partitioned'`` force the
        layout. ``None`` means ``'auto'``.

    Returns
    -------
    Path
        The checkpoint directory, so the call can be chained or logged.

    Raises
    ------
    ValidationError
        If ``sidecar_partition_rows`` is not a positive integer, or
        ``sidecar_layout`` is not one of the three accepted values. Checked
        before any writing, including when no sidecar will be produced, so a
        typo fails immediately rather than on the one dataset large enough to
        trigger the partitioned path.

    Notes
    -----
    **No estimator is saved here.** Loading a checkpoint gives back data, roles,
    splits, history, and plans — not a model. Use a pipeline bundle for that,
    and save both when a run needs to be both resumable and deployable.

    **The split is what makes a resume honest.** Everything else could be
    recomputed; the exact partition membership could not.

    **A query plan cannot be saved, only its result.** Canonical
    ``frame.parquet`` stays the interchange source of truth and keeps older
    loaders working. When a Polars or DuckDB handle is attached, a sidecar
    snapshot is written so restore can reattach without rebuilding eagerly from
    the exported Pandas frame. A LazyFrame's *plan* is not persisted — only the
    Parquet bytes and a ``lazy_intent`` flag, so a restored lazy frame is a new
    scan over the snapshot rather than the original pipeline.

    **The sidecar knobs are optional and backward compatible.** Omitting them
    preserves prior defaults, and older single-file sidecars still load.

    Examples
    --------
    Save mid-loop, resume later, and confirm the split survived::

        save_checkpoint(
            "artifacts/run-01",
            dataset=dataset,
            split_plan=split_plan,
            history=session.history(),
            plans=session.plans(),
        )

        restored = load_checkpoint("artifacts/run-01")
        assert restored.split_plan is not None

    See Also
    --------
    load_checkpoint : The other half of this pair.
    """
    # Validate public knobs even when no native sidecar will be written.
    _resolve_sidecar_options(
        n_rows=int(dataset.n_rows) if dataset.has_native else 0,
        partition_rows=sidecar_partition_rows,
        compression=sidecar_compression,
        layout=sidecar_layout,
    )

    root = Path(path)
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    sidecar_meta = _write_native_sidecar(
        dataset,
        data_dir,
        partition_rows=sidecar_partition_rows,
        compression=sidecar_compression,
        layout=sidecar_layout,
    )

    data_path = data_dir / "frame.parquet"
    dataset._ensure_pandas().to_parquet(data_path, index=False)

    plan_payload = dict(plans or {})
    has_plans = any(value is not None for value in plan_payload.values())

    meta = {
        "source": dataset.source,
        "mode": dataset.mode.value,
        "engine": dataset.engine.value,
        "schema": dataset.schema.to_dict(),
        "roles": {k: v.value for k, v in dataset.roles.items()},
        "n_rows": dataset.n_rows,
        "columns": dataset.columns,
        "has_native": dataset.has_native,
        "has_lazy_native": dataset.has_lazy_native,
        "buildml_version": __version__,
        "has_plans": has_plans,
        "pipeline_compatibility": PIPELINE_COMPATIBILITY,
        "native_sidecar_limits": NATIVE_SIDECAR_LIMITS,
    }
    if sidecar_meta is not None:
        meta["native_sidecar"] = sidecar_meta
    splits_payload = None if split_plan is None else split_plan.to_dict()
    history_payload = normalize_history(history)

    _write_json(root / "meta.json", meta)
    _write_json(root / "splits.json", splits_payload)
    _write_json(root / "history.json", history_payload)

    hashes = {
        "data/frame.parquet": _sha256_file(data_path),
        "meta.json": _sha256_file(root / "meta.json"),
        "splits.json": _sha256_file(root / "splits.json"),
        "history.json": _sha256_file(root / "history.json"),
    }
    if sidecar_meta is not None:
        hashes.update(_sidecar_hashes(root, sidecar_meta))
    if has_plans:
        from buildml.pipeline.bundle import pack_plans_payload

        plans_path = root / "plans.joblib"
        joblib.dump(pack_plans_payload(**plan_payload), plans_path)
        hashes["plans.joblib"] = _sha256_file(plans_path)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "buildml_version": __version__,
        "history_schema_version": HISTORY_SCHEMA_VERSION,
        "hashes": hashes,
        "pipeline_compatibility": PIPELINE_COMPATIBILITY,
    }
    _write_json(root / "MANIFEST.json", manifest)
    return root


def load_checkpoint(path: str | Path, *, data_only: bool = False) -> LoadedCheckpoint:
    """Restore a checkpoint, and say how much of it could safely be trusted.

    Reading the files back is the easy part. The question this answers is
    whether the saved roles and split still apply — if the data on disk has
    changed since the save, reusing the old partition membership would assign
    rows to partitions they were never in, and every score from before and after
    the resume would describe different experiments.

    Rather than choosing for you, the verdict is returned alongside the
    contents, downgrading what cannot be trusted and blocking only when nothing
    can.

    Parameters
    ----------
    path:
        The checkpoint directory. A bare ``.parquet`` path is also accepted and
        treated as data alone.
    data_only:
        Ignore all metadata and treat the checkpoint as a fresh ingest of its
        data file. Useful when the roles or splits are known to be stale and you
        intend to redefine them, but it discards the split, so scores after this
        are not comparable with scores from before.

    Returns
    -------
    LoadedCheckpoint
        The dataset, split plan, history, plans, and the reattach verdict.

    Raises
    ------
    ValidationError
        If no data file is found at the path, or if reattach is blocked —
        meaning the current data is incompatible enough that resuming would be
        misleading. The messages name the specific mismatches.

    Notes
    -----
    **Check ``reattach.status`` before trusting the split.** A clean load with a
    degraded status means some of what you saved did not survive, and the split
    is usually the casualty.

    **The native engine is reattached when it can be.** A sidecar, single-file
    or partitioned, is preferred when present and the engine extra is installed.
    Checkpoints holding only ``data/frame.parquet``, or a legacy single-file
    sidecar, still restore; the layout is detected from ``meta.native_sidecar``.

    **A restored lazy frame is not the original plan.** It is a new scan over
    the snapshot taken at save time, so anything that depended on the upstream
    query is gone.

    See Also
    --------
    save_checkpoint : Writing what this reads.
    buildml.checkpoint.validate.validate_reattach : How the verdict is decided.
    """
    root = Path(path)
    data_path = root / "data" / "frame.parquet"
    if not data_path.exists():
        # Allow a bare parquet path for flexibility.
        alt = root if root.suffix.lower() == ".parquet" else None
        if alt is None or not alt.exists():
            raise ValidationError(f"Checkpoint data not found at '{data_path}'")
        data_path = alt

    frame = pd.read_parquet(data_path)
    schema = schema_from_dataframe(frame)

    meta = None if data_only else _read_json(root / "meta.json")
    splits_payload = None if data_only else _read_json(root / "splits.json")
    history_payload = None if data_only else _read_json(root / "history.json")
    manifest = None if data_only else _read_json(root / "MANIFEST.json")

    reattach = validate_reattach(
        current_schema=schema,
        current_columns=[str(c) for c in frame.columns],
        current_n_rows=len(frame),
        meta=None if data_only else meta,
        splits_payload=None if data_only else splits_payload,
    )

    roles: dict[str, ColumnRole] = {}
    mode = DataMode.MEMORY
    engine = EngineName.PANDAS
    source = str(data_path)
    if meta and reattach.status not in {"fresh_ingest", "blocked"}:
        roles = {k: ColumnRole(v) for k, v in meta.get("roles", {}).items() if k in frame.columns}
        mode = coerce_data_mode(meta.get("mode", DataMode.MEMORY.value))
        engine = EngineName(meta.get("engine", EngineName.PANDAS.value))
        source = str(meta.get("source", source))

    if reattach.status == "blocked":
        raise ValidationError("; ".join(reattach.messages))

    if meta is None:
        resolved_schema = schema
    else:
        resolved_schema = TableSchema.from_dict(meta.get("schema", schema.to_dict()))
    dataset = Dataset.from_pandas(
        frame,
        schema=resolved_schema,
        mode=mode,
        engine=engine,
        source=source,
        roles=roles,
    )
    # Prefer live schema from frame when columns were added.
    dataset.schema = schema

    native_messages = _reattach_native_handle(
        dataset,
        engine=engine,
        root=root,
        meta=None if data_only else meta,
    )
    if native_messages:
        reattach.messages.extend(native_messages)
        reattach.details["has_native"] = dataset.has_native
        reattach.details["has_lazy_native"] = dataset.has_lazy_native
        reattach.details["engine"] = dataset.engine.value
        sidecar = (meta or {}).get("native_sidecar") if meta else None
        if isinstance(sidecar, dict):
            reattach.details["native_sidecar"] = dict(sidecar)

    split_plan = reattach.split_plan
    history = normalize_history(history_payload)

    plans: dict[str, Any] = {}
    plans_path = root / "plans.joblib"
    if not data_only and plans_path.exists():
        from buildml.pipeline.bundle import unpack_plans_payload

        loaded = joblib.load(plans_path)
        plans, _plans_format = unpack_plans_payload(loaded)

    return LoadedCheckpoint(
        dataset=dataset,
        split_plan=split_plan,
        history=history,
        reattach=reattach,
        meta=meta,
        manifest=manifest,
        plans=plans,
    )


def _resolve_sidecar_options(
    *,
    n_rows: int,
    partition_rows: int | None,
    compression: str | None,
    layout: SidecarLayout | str | None,
) -> tuple[bool, int, str]:
    """Return ``(partitioned, rows_per_partition, compression)``."""
    resolved_compression = (
        SIDECAR_DEFAULT_COMPRESSION if compression is None else str(compression).strip()
    )
    if not resolved_compression:
        raise ValidationError("sidecar_compression must be a non-empty codec name")

    rows_per = (
        SIDECAR_ROWS_PER_PARTITION if partition_rows is None else int(partition_rows)
    )
    if rows_per < 1:
        raise ValidationError("sidecar_partition_rows must be a positive integer")

    resolved_layout = "auto" if layout is None else str(layout).strip().lower()
    if resolved_layout not in {"auto", "single", "partitioned"}:
        raise ValidationError(
            "sidecar_layout must be one of 'auto', 'single', or 'partitioned'"
        )
    if resolved_layout == "single":
        partitioned = False
    elif resolved_layout == "partitioned":
        partitioned = True
    else:
        partitioned = n_rows >= SIDECAR_PARTITION_ROW_THRESHOLD
    return partitioned, rows_per, resolved_compression


def _write_native_sidecar(
    dataset: Dataset,
    data_dir: Path,
    *,
    partition_rows: int | None = None,
    compression: str | None = None,
    layout: SidecarLayout | str | None = None,
) -> dict[str, Any] | None:
    """Persist an optional engine-native Parquet sidecar; return metadata or None."""
    if not dataset.has_native or dataset.engine == EngineName.PANDAS:
        return None
    single_path = data_dir / "native_sidecar.parquet"
    part_dir = data_dir / "native_sidecar"
    # Clear prior sidecar artifacts from a reused checkpoint directory.
    if single_path.exists():
        single_path.unlink(missing_ok=True)
    if part_dir.exists():
        for child in part_dir.glob("*.parquet"):
            child.unlink(missing_ok=True)
        try:
            part_dir.rmdir()
        except OSError:
            pass

    lazy_intent = bool(dataset.has_lazy_native or dataset.mode == DataMode.LAZY)
    n_rows = int(dataset.n_rows)
    partitioned, rows_per, resolved_compression = _resolve_sidecar_options(
        n_rows=n_rows,
        partition_rows=partition_rows,
        compression=compression,
        layout=layout,
    )
    try:
        from buildml.data.engines import get_engine

        engine = get_engine(dataset.engine)
        writer = getattr(engine, "write_parquet", None)
        if not callable(writer):
            return None
        if partitioned:
            part_dir.mkdir(parents=True, exist_ok=True)
            parts = _write_partitioned_sidecar(
                dataset,
                engine=engine,
                writer=writer,
                part_dir=part_dir,
                compression=resolved_compression,
                rows_per_partition=rows_per,
            )
            if not parts:
                return None
            return {
                "relative_path": NATIVE_SIDECAR_DIR_RELATIVE,
                "format": "parquet",
                "layout": "partitioned",
                "compression": resolved_compression,
                "n_partitions": len(parts),
                "rows_per_partition": rows_per,
                "n_rows": n_rows,
                "parts": parts,
                "engine": dataset.engine.value,
                "lazy_intent": lazy_intent,
                "limits": NATIVE_SIDECAR_LIMITS,
            }
        writer(
            dataset.native,
            single_path,
            compression=resolved_compression,
        )
    except Exception:  # noqa: BLE001
        if single_path.exists():
            single_path.unlink(missing_ok=True)
        if part_dir.exists():
            for child in part_dir.glob("*.parquet"):
                child.unlink(missing_ok=True)
        return None
    if not single_path.exists():
        return None
    return {
        "relative_path": NATIVE_SIDECAR_RELATIVE,
        "format": "parquet",
        "layout": "single",
        "compression": resolved_compression,
        "n_partitions": 1,
        "n_rows": n_rows,
        "engine": dataset.engine.value,
        "lazy_intent": lazy_intent,
        "limits": NATIVE_SIDECAR_LIMITS,
    }


def _write_partitioned_sidecar(
    dataset: Dataset,
    *,
    engine: Any,
    writer: Any,
    part_dir: Path,
    compression: str,
    rows_per_partition: int = SIDECAR_ROWS_PER_PARTITION,
) -> list[str]:
    """Write row-sliced Parquet parts; return relative part file names."""
    n_rows = int(dataset.n_rows)
    parts: list[str] = []
    step = max(1, int(rows_per_partition))
    # Materialize once for slicing when engines lack native offset writes.
    # Prefer Arrow for DuckDB; Polars collect; else Pandas.
    table = dataset.native
    arrow_table = None
    polars_df = None
    if dataset.engine == EngineName.DUCKDB:
        to_arrow = getattr(engine, "to_arrow", None)
        if callable(to_arrow):
            arrow_table = to_arrow(table)
    elif dataset.engine == EngineName.POLARS:
        collect = getattr(engine, "_collect", None)
        polars_df = collect(table) if callable(collect) else table

    for start in range(0, n_rows, step):
        end = min(start + step, n_rows)
        name = f"part-{start:08d}-{end:08d}.parquet"
        dest = part_dir / name
        if arrow_table is not None:
            import pyarrow.parquet as pq

            pq.write_table(
                arrow_table.slice(start, end - start),
                dest,
                compression=compression,
            )
        elif polars_df is not None:
            chunk = polars_df.slice(start, end - start)
            writer(chunk, dest, compression=compression)
        else:
            frame = dataset._ensure_pandas().iloc[start:end]
            native_chunk = engine.from_pandas(frame)
            writer(native_chunk, dest, compression=compression)
            if dataset.engine == EngineName.DUCKDB:
                from buildml.data.engines.duckdb_engine import DuckDBTable, close_duckdb_connection

                if isinstance(native_chunk, DuckDBTable):
                    close_duckdb_connection(native_chunk.connection)
        parts.append(name)
    return parts


def _sidecar_hashes(root: Path, sidecar_meta: dict[str, Any]) -> dict[str, str]:
    """Hash single-file or partitioned sidecar artifacts for the manifest."""
    rel = str(sidecar_meta.get("relative_path") or NATIVE_SIDECAR_RELATIVE)
    target = root / rel
    out: dict[str, str] = {}
    if sidecar_meta.get("layout") == "partitioned" and target.is_dir():
        for part in sorted(target.glob("*.parquet")):
            out[f"{rel}/{part.name}"] = _sha256_file(part)
        return out
    if target.is_file():
        out[rel] = _sha256_file(target)
    return out


def _resolve_sidecar_path(root: Path, sidecar_info: dict[str, Any]) -> Path | None:
    """Locate a single-file or partitioned sidecar; None if missing."""
    rel = str(sidecar_info.get("relative_path") or NATIVE_SIDECAR_RELATIVE)
    candidate = root / rel
    if candidate.exists():
        return candidate
    # Backward-compatible default when meta omits relative_path.
    legacy = root / NATIVE_SIDECAR_RELATIVE
    if legacy.exists():
        return legacy
    partitioned = root / NATIVE_SIDECAR_DIR_RELATIVE
    if partitioned.is_dir() and any(partitioned.glob("*.parquet")):
        return partitioned
    return None


def _reattach_native_handle(
    dataset: Dataset,
    *,
    engine: EngineName,
    root: Path,
    meta: dict[str, Any] | None,
) -> list[str]:
    """Try to rebuild Dataset.native after checkpoint load; return messages."""
    if engine == EngineName.PANDAS:
        return []

    sidecar_info = (meta or {}).get("native_sidecar") if meta else None
    if isinstance(sidecar_info, dict):
        sidecar_path = _resolve_sidecar_path(root, sidecar_info)
        if sidecar_path is not None:
            messages = _reattach_from_sidecar(
                dataset,
                engine=engine,
                sidecar_path=sidecar_path,
                sidecar_info=sidecar_info,
            )
            if messages is not None:
                return messages

    # Backward-compatible path: eager rebuild from canonical Parquet/Pandas.
    try:
        dataset.attach_native(rebuild=True)
    except Exception as exc:  # noqa: BLE001
        from buildml.core.errors import MissingExtraError

        if isinstance(exc, MissingExtraError):
            return [
                f"Could not restore {engine.value} native handle: {exc}. "
                f"Install with pip install 'buildml[{engine.value}]'. "
                "Dataset remains Pandas-backed; engine metadata is preserved so "
                "sync_native() can retry after install."
            ]
        return [
            f"Could not restore {engine.value} native handle from checkpoint "
            f"Parquet payload ({type(exc).__name__}: {exc}). "
            "Dataset remains Pandas-backed; call sync_native() after fixing the "
            "engine environment."
        ]
    return [
        f"Restored {engine.value} native handle from checkpoint Parquet payload "
        "(eager rebuild; no native sidecar was available). "
        "LazyFrame plans are not persisted across checkpoints."
    ]


def _reattach_from_sidecar(
    dataset: Dataset,
    *,
    engine: EngineName,
    sidecar_path: Path,
    sidecar_info: dict[str, Any],
) -> list[str] | None:
    """Attach from sidecar. Return messages, or None to fall back to frame.parquet."""
    lazy_intent = bool(sidecar_info.get("lazy_intent"))
    default_layout = "partitioned" if sidecar_path.is_dir() else "single"
    layout = str(sidecar_info.get("layout") or default_layout)
    compression = sidecar_info.get("compression")
    rel = str(sidecar_info.get("relative_path") or sidecar_path.name)
    try:
        from buildml.data.engines import get_engine

        adapter = get_engine(engine)
        reader = getattr(adapter, "from_parquet", None)
        if not callable(reader):
            return None
        # Prefer lazy reattach only when the original handle was lazy (Polars)
        # or the session mode was lazy (DuckDB file-backed relation).
        use_lazy = lazy_intent and (engine == EngineName.POLARS or dataset.mode == DataMode.LAZY)
        native = reader(sidecar_path, lazy=use_lazy)
        dataset.native = native
        dataset._pandas_stale = True
        dataset._owns_native_connection = engine == EngineName.DUCKDB
    except Exception as exc:  # noqa: BLE001
        from buildml.core.errors import MissingExtraError

        if isinstance(exc, MissingExtraError):
            return [
                f"Native sidecar present but {engine.value} is unavailable: {exc}. "
                f"Install with pip install 'buildml[{engine.value}]'. "
                "Falling back to Pandas frame; engine metadata is preserved."
            ]
        # Fall back to canonical frame rebuild.
        return None

    layout_note = (
        f"layout={layout}"
        + (f"; compression={compression}" if compression else "")
        + (
            f"; n_partitions={sidecar_info.get('n_partitions')}"
            if sidecar_info.get("n_partitions")
            else ""
        )
    )
    if use_lazy and engine == EngineName.POLARS and dataset.has_lazy_native:
        return [
            f"Restored {engine.value} native handle from checkpoint sidecar "
            f"({rel}; {layout_note}) via scan_parquet (lazy_intent=True). "
            "This is a new scan plan over the sidecar bytes, not the original "
            "LazyFrame graph. Collect-on-promote still applies; sklearn needs an "
            "in-memory design matrix."
        ]
    if use_lazy and engine == EngineName.DUCKDB:
        return [
            f"Restored {engine.value} native handle from checkpoint sidecar "
            f"({rel}; {layout_note}) via read_parquet (file-backed relation; "
            "lazy_intent metadata preserved). Sklearn still materializes at fit."
        ]
    return [
        f"Restored {engine.value} native handle from checkpoint sidecar "
        f"({rel}; {layout_note}) without rebuilding from the Pandas-exported "
        "frame.parquet. Engine-native query plans are not persisted."
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
