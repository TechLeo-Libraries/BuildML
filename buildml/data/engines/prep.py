"""Build the smallest frame an estimator can be fitted on, and say what it cost.

Fitting is where deferral ends. Whatever engine did the prep, sklearn needs a
real array in memory, and the size of that array is set by how much narrowing
happened first.

This module does the narrowing at that boundary. It projects to exactly the
columns the model needs, optionally caps rows, and then materialises: using the
native handle where one exists so the full-width frame is never built. It also
records what it did, because "the model was fitted on 50,000 sampled rows of 4
million" is a fact that has to survive into the run history rather than being
inferred later from a suspiciously fast fit.

See Also
--------
buildml.data.dataset.Dataset.project : The narrowing operation.
buildml.ingest.detect.check_materialization : The size gate applied here.
"""

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
    """The design matrix, together with an account of how it was produced.

    The frame is what gets fitted. The rest is the record: which columns were
    asked for and which arrived, how many rows existed versus how many were
    kept, whether the native handle was used, and what the caller should know
    about the result.

    Attributes
    ----------
    frame:
        The materialised design matrix.
    columns_requested:
        What was asked for.
    columns_materialized:
        What actually came back. A difference means the engine dropped
        something.
    n_rows_source:
        Rows in the source before any capping.
    n_rows_materialized:
        Rows in the frame.
    sampled:
        Whether rows were dropped to meet a cap. **The single most important
        field**: a fit on a sample is not a fit on the population.
    engine:
        Which engine did the prep.
    used_native_handle:
        Whether an attached native handle was used, or a one-shot conversion.
    disclosures:
        What happened, in plain language, for history and walkthroughs.
    limitations:
        What the result cannot be used for.

    Notes
    -----
    **``sampled`` being true changes what the metrics mean.** Scores from a
    sampled fit describe the sample; treating them as population estimates is
    the error this field exists to prevent.

    See Also
    --------
    prepare_design_frame : What produces this.
    """

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
        """Return the account without the data.

        Everything except ``frame``, so a run record can say how the design
        matrix was built without carrying the matrix itself.

        Returns
        -------
        dict
            JSON-safe metadata: column lists, row counts, sampling flag,
            engine, native-handle flag, disclosures, and limitations.

        Notes
        -----
        **``frame`` is deliberately excluded.** It is the payload, and it does
        not belong in a log.
        """
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
    """Narrow to the modelled columns, then materialise, recording what it cost.

    Called at the estimator boundary. Projection happens first and, where a
    native handle is attached, entirely inside the engine: so a table with
    hundreds of columns is never widened into pandas just to fit on twelve of
    them.

    Three paths exist, and which one runs determines how much is avoided. With
    pandas, projection is a slice of an already-loaded frame. With an attached
    native handle, projection and sampling both run in the engine. With an
    engine selected but no handle, only the projected columns are converted :
    better than nothing, and a sign that ``with_engine`` should be called to
    attach one.

    Parameters
    ----------
    dataset:
        The source.
    columns:
        Everything the design matrix needs, features and target.
    sample_rows:
        Cap rows before materialising. Changes what a fit means: see the notes.
    random_state:
        Seed for the cap, so the same rows are drawn across runs.
    hard_limit_bytes:
        Refuse above this estimated size, forwarded to
        :func:`~buildml.ingest.detect.check_materialization`.
    context:
        Label used in size-gate messages.

    Returns
    -------
    MaterializePrepResult
        The frame plus the account of how it was produced.

    Raises
    ------
    ValidationError
        If no columns are given, a column does not exist, ``sample_rows`` is
        less than one, or the frame exceeds a configured hard limit.

    Notes
    -----
    **Sampling is a disclosure, not an out-of-core strategy.** A capped fit
    sees a different empirical distribution, so its scores describe the sample.
    That fact is recorded in ``limitations`` precisely because it is easy to
    forget between the fit and the report.

    **This does not make fitting out-of-core.** It narrows what has to be in
    memory; the estimator boundary is unchanged.

    **Projection is by column, not by role.** Pass the target too if the caller
    needs it: it is not added automatically.

    Examples
    --------
    Prepare a design matrix for two features and a target::

        prep = prepare_design_frame(dataset, ["age", "region", "outcome"])
        model.fit(prep.frame[["age", "region"]], prep.frame["outcome"])

    See Also
    --------
    MaterializePrepResult : What comes back.
    buildml.data.dataset.Dataset.project : The narrowing this performs.
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
        # Engine selected but no persistent native handle: convert projected cols only.
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
