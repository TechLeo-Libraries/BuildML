"""Engine-aware column projection and sampling before sklearn materialize."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from buildml.core.errors import ValidationError
from buildml.core.types import EngineName
from buildml.data.dataset import Dataset
from buildml.ingest.detect import check_materialization


@dataclass(slots=True)
class MaterializePrepResult:
    """Outcome of projecting/sampling before Pandas materialization for fit."""

    frame: pd.DataFrame
    columns_requested: tuple[str, ...]
    columns_materialized: tuple[str, ...]
    n_rows_source: int
    n_rows_materialized: int
    sampled: bool
    engine: str
    used_native_handle: bool = False
    disclosures: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns_requested": list(self.columns_requested),
            "columns_materialized": list(self.columns_materialized),
            "n_rows_source": self.n_rows_source,
            "n_rows_materialized": self.n_rows_materialized,
            "sampled": self.sampled,
            "engine": self.engine,
            "used_native_handle": self.used_native_handle,
            "disclosures": list(self.disclosures),
            "limitations": list(self.limitations),
        }


def prepare_design_frame(
    dataset: Dataset,
    columns: list[str] | tuple[str, ...],
    *,
    sample_rows: int | None = None,
    random_state: int | None = 0,
    hard_limit_bytes: int | None = None,
    context: str = "estimator design matrix",
) -> MaterializePrepResult:
    """Project (and optionally sample) columns via the active engine, then materialize.

    Parameters
    ----------
    dataset:
        Source dataset. When a Polars/DuckDB ``native`` handle is attached,
        projection and sampling run on that handle without converting the
        full-width frame first. Otherwise the Pandas ``frame`` is used, with
        an optional one-shot engine conversion of the projected columns only.
    columns:
        Columns required for the design matrix (features and/or target).
    sample_rows:
        Optional row cap applied before materialize. Disclose-only sampling —
        not a substitute for out-of-core training.
    random_state:
        Seed for sampling when supported.
    hard_limit_bytes:
        Optional hard materialization gate forwarded to
        :func:`~buildml.ingest.detect.check_materialization`.
    context:
        Label for materialization gate messages.

    Notes
    -----
    Sklearn estimators still require an in-memory design matrix. This helper
    narrows columns (and optionally rows) before that boundary; it does not
    provide true out-of-core fitting.
    """
    cols = [str(c) for c in columns]
    if not cols:
        raise ValidationError("prepare_design_frame requires at least one column")
    missing = [c for c in cols if c not in dataset.columns]
    if missing:
        raise ValidationError(f"Design-matrix columns missing from dataset: {missing}")

    from buildml.data.engines import get_engine

    engine_name = EngineName(dataset.engine)
    engine = get_engine(engine_name)
    n_source = int(dataset.n_rows)
    disclosures: list[str] = []
    limitations: list[str] = [
        "Sklearn fit still consumes an in-memory Pandas/NumPy design matrix; "
        "engine prep does not enable out-of-core training.",
    ]
    used_native = False
    sampled = False

    if engine_name == EngineName.PANDAS:
        work = dataset._ensure_pandas().loc[:, cols]
        disclosures.append(
            f"Projected to {len(cols)} requested column(s) before materialization."
        )
        if sample_rows is not None:
            if sample_rows < 1:
                raise ValidationError("sample_rows must be >= 1 when provided")
            if sample_rows < len(work):
                work = work.sample(n=sample_rows, random_state=random_state)
                sampled = True
                disclosures.append(
                    f"Sampled {sample_rows} of {n_source} rows before materialize "
                    f"(random_state={random_state})."
                )
        frame = work.copy()
    elif dataset.has_native:
        used_native = True
        table = engine.select_columns(dataset.native, cols)
        lazy_checker = getattr(engine, "is_lazy_handle", None)
        if callable(lazy_checker) and lazy_checker(dataset.native):
            disclosures.append(
                f"Projected to {len(cols)} column(s) on an attached Polars LazyFrame; "
                "collect runs at Pandas promotion (not out-of-core sklearn)."
            )
        else:
            disclosures.append(
                f"Projected to {len(cols)} column(s) on the attached {engine_name.value} "
                "native handle (no full-width Pandas conversion)."
            )
        if sample_rows is not None:
            if sample_rows < 1:
                raise ValidationError("sample_rows must be >= 1 when provided")
            n_projected = engine.n_rows(table)
            if sample_rows < n_projected:
                table = engine.sample_rows(table, sample_rows, random_state=random_state)
                sampled = True
                disclosures.append(
                    f"Sampled {sample_rows} of {n_projected} rows on the "
                    f"{engine_name.value} native handle before materialize "
                    f"(random_state={random_state})."
                )
        frame = engine.to_pandas(table)
        frame = frame.loc[:, [c for c in cols if c in frame.columns]]
    else:
        # Engine selected but no persistent native handle — convert projected cols only.
        narrow = dataset._ensure_pandas().loc[:, cols]
        disclosures.append(
            f"Projected to {len(cols)} requested column(s) before materialization."
        )
        native = engine.from_pandas(narrow)
        disclosures.append(
            f"{engine_name.value} engine: converted only the projected columns "
            "(not the full-width frame) for prep-time ops. Call with_engine(...) "
            "to attach a persistent native handle for repeated prep."
        )
        table = native
        if sample_rows is not None:
            if sample_rows < 1:
                raise ValidationError("sample_rows must be >= 1 when provided")
            n_projected = engine.n_rows(native)
            if sample_rows < n_projected:
                table = engine.sample_rows(native, sample_rows, random_state=random_state)
                sampled = True
                disclosures.append(
                    f"Sampled {sample_rows} of {n_projected} rows on the "
                    f"{engine_name.value} engine before materialize "
                    f"(random_state={random_state})."
                )
        frame = engine.to_pandas(table)
        frame = frame.loc[:, [c for c in cols if c in frame.columns]]
        limitations.append(
            f"Dataset.native was unset; {engine_name.value} projection used a "
            "one-shot conversion of the projected columns."
        )

    check_materialization(frame, context=context, hard_limit_bytes=hard_limit_bytes)
    if sampled:
        limitations.append(
            "Row sampling changes the empirical distribution seen by the estimator; "
            "do not treat sampled fits as full-population estimates."
        )

    return MaterializePrepResult(
        frame=frame,
        columns_requested=tuple(cols),
        columns_materialized=tuple(str(c) for c in frame.columns),
        n_rows_source=n_source,
        n_rows_materialized=int(len(frame)),
        sampled=sampled,
        engine=engine_name.value,
        used_native_handle=used_native,
        disclosures=disclosures,
        limitations=limitations,
    )
