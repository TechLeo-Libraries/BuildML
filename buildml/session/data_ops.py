"""Ingest, roles, splits, engines, and checkpoint orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from buildml.session.session import Session

import pandas as pd

from buildml.session._imports import (
    ColumnRole,
    DataMode,
    EngineName,
    PartitionName,
    ValidationError,
    assert_fit_partition,
    coerce_data_mode,
    create_group_split,
    create_split,
    create_time_split,
    frame_for_partition,
    ingest_source,
    inject_partitions,
    load_checkpoint,
    make_operation_record,
    prior_state,
    save_checkpoint,
    session_state,
)


def close_native(session) -> None:
    """Close the DuckDB connection this session owns, if it has one.

    :meth:`with_engine` with ``'duckdb'`` opens a connection that stays
    alive so later queries can reuse it. That connection holds an operating
    system handle, so it should be released when you are finished: on
    Windows especially, an open handle can block deleting or overwriting
    the underlying file.

    Calling this is always safe. It does nothing when no dataset is
    attached, when the engine is Pandas or Polars, or when the connection
    has already been closed. Datasets derived from a parent share the
    parent's connection and are not owners, so closing a derived handle
    does not pull the connection out from under the original.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.

    Notes
    -----
    You rarely need to call this by hand: prefer ``with session:``, which
    calls it for you on exit.
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
) -> "Session":
    """Create a session by loading a table, and inspect it while loading.

    This is where every BuildML workflow starts. Give it a DataFrame you
    already have or a path to a file on disk, and you get back a
    :class:`Session` holding the data plus an
    :class:`~buildml.core.results.IngestReport` describing what was found:
    the detected file format, the column schema, row and byte estimates,
    which compute engines are installed, and any warnings worth reading.

    Ingest does not just read the file: it decides *how* to read it. A
    small CSV loads straight into Pandas. A large one does not, because
    quietly pulling a multi-gigabyte file into memory is how notebooks die.
    Instead BuildML refuses and tells you the four ways forward: force it
    with ``mode='memory'``, look before you leap with ``dry_run=True``,
    sample with ``read_nrows``, or install ``buildml[engines]`` and load
    natively through Polars or DuckDB.

    Parameters
    ----------
    session_cls:
        The :class:`~buildml.session.session.Session` class (classmethod
        receiver) used to construct the new session instance.
    source:
        The data to load: a :class:`pandas.DataFrame` you already hold in
        memory, or a path to a ``.csv``, ``.tsv``, ``.parquet``, or
        ``.arrow``/``.feather`` file. Format is detected from the file, not
        assumed from the extension alone.
    mode:
        How the data should live in memory: ``'memory'`` for a fully
        materialised frame, ``'lazy'`` to keep an engine handle and defer
        work until something forces materialisation. Leave as ``None`` to
        let the size heuristic decide, and pass ``'memory'`` explicitly to
        override a refusal on a large file.
    engine:
        Which compute engine backs the data: ``'pandas'``, ``'polars'``, or
        ``'duckdb'``. ``None`` picks the best available for the estimated
        size. Polars and DuckDB read the file natively with no Pandas-first
        pass, which is what makes large sources tractable, but they require
        ``pip install 'buildml[engines]'``.
    dry_run:
        When True, inspect without loading: you get a session carrying the
        report but no dataset. Use this to see the schema, size estimate,
        and warnings before committing memory to a file you are unsure
        about.
    mock_byte_estimate:
        Pretend the source is this many bytes when applying the size
        heuristics. Exists so tests and demonstrations can trigger the
        large-file path without producing a large file.
    read_nrows:
        Read at most this many rows from a CSV. A quick way to work on a
        representative slice of something too big to load whole: but note
        that statistics from a truncated read describe the slice, not the
        file.

    Returns
    -------
    Session
        A new session. It carries a dataset unless ``dry_run=True`` (or a
        large-source refusal under dry run), and always carries
        :attr:`ingest_report`. Read that report's ``warnings`` before
        continuing; it is where scale and engine advice appears.

    Raises
    ------
    ~buildml.core.errors.IngestError
        The path does not exist, the format is not one BuildML reads, or
        the source is large enough that loading it blindly into Pandas
        would be reckless. The message names the specific way out.
    ~buildml.core.errors.MissingExtraError
        You asked for ``engine='polars'`` or ``'duckdb'`` without the
        matching extra installed.

    Notes
    -----
    **Scale:** Large paths are not silently loaded into Pandas. Use
    ``dry_run=True``, ``read_nrows``, ``mode='memory'`` (force), or engine
    extras.

    **Leakage:** Call :meth:`split` before fit-capable operations. Use
    :meth:`assert_can_fit` to enforce train-only fit scope.

    Examples
    --------
    The ordinary case: a DataFrame already in hand:

    >>> import pandas as pd
    >>> from buildml import Session
    >>> session = Session.ingest(pd.DataFrame({"a": [1, 2], "y": [0, 1]}))
    >>> session.dataset.frame.shape
    (2, 2)

    Inspect a file before deciding how to load it:

    >>> probe = Session.ingest("events.parquet", dry_run=True)  # doctest: +SKIP
    >>> probe.ingest_report.row_estimate  # doctest: +SKIP
    4210332
    >>> probe.ingest_report.warnings  # doctest: +SKIP
    ['Source looks large (812993024 bytes estimated). ...']

    Then load it natively rather than through Pandas:

    >>> session = Session.ingest(
    ...     "events.parquet", engine="duckdb", mode="lazy"
    ... )  # doctest: +SKIP

    See Also
    --------
    Session.set_roles : The next step: tell BuildML what the columns mean.
    Session.with_engine : Switch engines after ingest.
    Session.checkpoint_load : Resume a saved session instead of re-reading.
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
    return cast("Session", session)
def set_roles(session, mapping: dict[str, str | ColumnRole]) -> "Session":
    """Declare what each column means, so later steps can act on it.

    A role is BuildML's answer to "which column is the answer, and which
    ones are allowed to help predict it?". Assigning roles once removes the
    need to pass column lists into every later call: :meth:`scale` knows
    to leave your identifier alone, :meth:`split` knows what to stratify
    on, and :meth:`fit` knows what it is predicting.

    The roles are:

    ``target``
        The column being predicted. Supervised methods require exactly one.
    ``feature``
        An input the model may learn from. Columns default to this.
    ``id``
        A row identifier. Carried through but never used as a predictor,
        and never modified by preprocessing.
    ``group``
        An entity that owns several rows: a customer, a patient, a
        document. :meth:`group_split` keeps all of a group's rows on the
        same side of the split.
    ``time``
        The timestamp that orders the data. Used by :meth:`time_split` and
        the forecasting methods.
    ``weight``
        Per-row importance passed to estimators that accept sample weights.
    ``ignore``
        Kept in the table, excluded from everything.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    mapping:
        Column name to role. Roles may be given as strings (``'target'``)
        or as :class:`~buildml.core.types.ColumnRole` members. Only the
        columns you name change; anything you leave out keeps its current
        role.

    Returns
    -------
    Session
        ``self``, so this call chains into the next step.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        A named column is not in the dataset, or the role is not one of the
        values above.

    Examples
    --------
    >>> import pandas as pd
    >>> from buildml import Session
    >>> frame = pd.DataFrame({"id": [1, 2], "x": [0.5, 0.9], "y": [0, 1]})
    >>> session = Session.ingest(frame)
    >>> _ = session.set_roles({"id": "id", "y": "target"})
    >>> session.dataset.roles["y"].value
    'target'

    Marking a grouping column is what makes a leakage-safe split possible:

    >>> _ = session.set_roles({"customer_id": "group"})  # doctest: +SKIP
    >>> _ = session.group_split(test_size=0.2)  # doctest: +SKIP

    See Also
    --------
    Session.split : The next step for independent rows.
    Session.group_split : Use when rows share an entity.
    Session.time_split : Use when rows are ordered in time.
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
    return cast("Session", session)
def split(
    session,
    *,
    test_size: float | int = 0.2,
    validation_size: float | int | None = None,
    random_state: int | None = 42,
    stratify: bool = False,
) -> "Session":
    """Randomly hold back rows so you can measure honest performance.

    A model that has seen a row can usually predict it. To find out whether
    it learned anything general, you must score it on rows it never saw.
    This method decides, once, which rows those are, and records the
    decision on the session so every later step respects it.

    Rows are shuffled and cut into a train partition (the model learns
    here), an optional validation partition (you tune here), and a test
    partition (you measure here, once, at the end). Nothing is copied :
    only row positions are stored: so the split costs almost nothing and
    stays consistent no matter how the data is transformed afterwards.

    Use this when rows are independent. If several rows describe the same
    customer or patient, use :meth:`group_split`; if the rows form a time
    series, use :meth:`time_split`. Random splitting in either of those
    cases leaks information and produces scores you cannot trust.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    test_size:
        How much data to hold back for the final measurement. A float is a
        proportion (``0.2`` means 20% of rows); an integer is an absolute
        row count. Larger test sets give a more stable estimate of
        performance but leave less to learn from.
    validation_size:
        How much to carve out of the remaining rows for tuning: again a
        proportion or a count. Set this when you plan to compare models or
        search hyperparameters, so the test set stays untouched until the
        end. ``None`` produces just train and test.
    random_state:
        Seed for the shuffle. Keeping the default ``42`` means the same
        rows land in the same partitions on every run, so your results are
        reproducible. Pass ``None`` for a different split each time.
    stratify:
        When True, preserve the target's class proportions in every
        partition. Turn this on for classification, particularly when one
        class is rare: an unstratified split can leave a rare class almost
        absent from test, making the score meaningless.

    Returns
    -------
    Session
        ``self``, so this call chains into preprocessing.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        No dataset is attached, the requested sizes leave a partition
        empty, or ``stratify=True`` was passed without exactly one target
        column (or with a class too rare to appear in every partition).

    Notes
    -----
    **Leakage:** After splitting, fit-capable operations must use the train
    partition only (:meth:`assert_can_fit`).

    Splitting before preprocessing is deliberate. BuildML's transforms fit
    their statistics on train rows alone, which is only possible if the
    split already exists: so ordering the calls this way is what makes the
    leakage guarantee real rather than aspirational.

    Examples
    --------
    >>> import pandas as pd
    >>> from buildml import Session
    >>> frame = pd.DataFrame({"a": [1, 2, 3, 4], "y": [0, 1, 0, 1]})
    >>> session = Session.ingest(frame).set_roles({"y": "target"})
    >>> _ = session.split(test_size=0.5, stratify=True)
    >>> len(session.partition("train")), len(session.partition("test"))
    (2, 2)

    Reserve a validation partition when you intend to tune:

    >>> _ = session.split(test_size=0.2, validation_size=0.2)  # doctest: +SKIP

    See Also
    --------
    Session.group_split : Keep an entity's rows together.
    Session.time_split : Respect chronological order.
    Session.inject_split : Reuse a split decided elsewhere.
    Session.cv_score : Rotate the holdout instead of fixing it.
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
    return cast("Session", session)
def inject_split(
    session,
    *,
    train_indices: list[int] | tuple[int, ...],
    test_indices: list[int] | tuple[int, ...],
    validation_indices: list[int] | tuple[int, ...] | None = None,
) -> "Session":
    """Adopt a split that was decided outside BuildML.

    Sometimes the partitioning is not yours to choose: a benchmark ships
    with an official train/test division, a colleague's split must be
    reproduced exactly, or the boundary follows domain logic no generic
    splitter encodes (everything before the regulation changed is train,
    everything after is test). Pass the row positions directly and BuildML
    treats them exactly as it would treat a split it generated: the
    leakage guards, partition accessors, and history record all apply.

    The plan is recorded with kind ``'injected'``, so :meth:`walkthrough`
    and the model card report honestly that the split was supplied rather
    than derived.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    train_indices:
        Positional row numbers (``0`` to ``n_rows - 1``, not DataFrame
        index labels) the model may learn from.
    test_indices:
        Positional row numbers reserved for the final measurement.
    validation_indices:
        Optional positional row numbers for tuning. ``None`` produces just
        train and test.

    Returns
    -------
    Session
        ``self``, so this call chains into preprocessing.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        The partitions overlap, an index falls outside the dataset, or
        train or test would be empty. Overlap is rejected rather than
        silently deduplicated, because a row in both train and test defeats
        the entire point of the split.

    Examples
    --------
    Reproduce a chronological cut-off decided by domain knowledge:

    >>> import pandas as pd
    >>> from buildml import Session
    >>> frame = pd.DataFrame({"a": [1, 2, 3, 4], "y": [0, 1, 0, 1]})
    >>> session = Session.ingest(frame).set_roles({"y": "target"})
    >>> _ = session.inject_split(train_indices=[0, 1], test_indices=[2, 3])
    >>> session.split_plan.kind
    'injected'

    See Also
    --------
    Session.split : Let BuildML choose the rows.
    Session.split_plan : Inspect the resulting membership.
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
    return cast("Session", session)
def group_split(
    session,
    *,
    test_size: float | int = 0.2,
    validation_size: float | int | None = None,
    random_state: int | None = 42,
    group_column: str | None = None,
) -> "Session":
    """Split by entity, so no customer appears on both sides.

    When several rows describe the same thing: twelve monthly records for
    one customer, forty sensor readings from one machine, every sentence of
    one document: a random row split scatters that entity across train and
    test. The model then sees eleven of the customer's months in training
    and is asked to predict the twelfth. It does well, and the score is a
    lie: in production the customer is entirely new.

    This method splits whole groups instead of individual rows. Every row
    belonging to a group lands in exactly one partition, so a test score
    answers the question you actually care about: how well does this work
    on someone we have never seen?

    Because groups are the unit, ``test_size`` counts groups rather than
    rows. Partitions therefore rarely come out at exactly the requested row
    proportion, and that is expected: groups differ in size.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    test_size:
        Proportion (float) or number (int) of *groups* held back for the
        final measurement.
    validation_size:
        Optional proportion or count of groups for tuning, taken from the
        groups not already assigned to test. ``None`` produces just train
        and test.
    random_state:
        Seed controlling which groups go where. The default keeps the
        assignment stable across runs.
    group_column:
        Which column identifies the entity. ``None`` uses the single column
        you marked with the ``group`` role via :meth:`set_roles`, which is
        the usual path; name a column here to override it for one call.

    Returns
    -------
    Session
        ``self``, so this call chains into preprocessing.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        No group column was resolved (none assigned and none passed, or
        several assigned), or there are too few distinct groups to fill the
        requested partitions.

    Notes
    -----
    **Leakage:** Prefer this over :meth:`split` when rows share entities
    (customers, sites, documents). Random row splits leak across groups.

    A useful check: if you can imagine two rows in your data that a model
    could match to each other by memorising an identity rather than by
    learning a pattern, you need this method.

    Examples
    --------
    >>> import pandas as pd
    >>> from buildml import Session
    >>> frame = pd.DataFrame(
    ...     {
    ...         "customer": ["a", "a", "b", "b", "c", "c"],
    ...         "spend": [10, 12, 90, 84, 41, 38],
    ...         "churn": [0, 0, 1, 1, 0, 0],
    ...     }
    ... )
    >>> session = Session.ingest(frame).set_roles(
    ...     {"customer": "group", "churn": "target"}
    ... )
    >>> _ = session.group_split(test_size=1)
    >>> train = set(session.partition("train")["customer"])
    >>> test = set(session.partition("test")["customer"])
    >>> train & test
    set()

    See Also
    --------
    Session.split : Use when rows are independent.
    Session.cv_score : Cross-validation with the same group awareness.
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
    return cast("Session", session)
def time_split(
    session,
    *,
    test_size: float | int = 0.2,
    validation_size: float | int | None = None,
    time_column: str | None = None,
) -> "Session":
    """Train on the past and test on the future, as deployment will.

    Shuffling a time series lets the model learn from Thursday to predict
    Wednesday. Nothing in the mathematics objects, and the score comes out
    excellent, but the arrangement is impossible in production: you never
    have next month's data when making this month's prediction. Models
    validated that way routinely collapse on release.

    This method sorts rows by their timestamp and cuts chronologically. The
    most recent rows become test, an optional validation block is taken
    from the end of what remains, and the earliest rows are train. The
    result mirrors reality: every evaluation row is later than every row
    the model learned from.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    test_size:
        Proportion (float) or number of rows (int) at the end of the
        timeline to hold back. Make this long enough to span the seasonal
        cycle you care about: a two-week test set says little about a
        model with yearly seasonality.
    validation_size:
        Optional proportion or row count for tuning, taken from the end of
        the remaining rows so it still sits before test in time. ``None``
        produces just train and test.
    time_column:
        Which column orders the data. ``None`` uses the single column you
        marked with the ``time`` role via :meth:`set_roles`.

    Returns
    -------
    Session
        ``self``, so this call chains into preprocessing.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        No time column was resolved (none assigned and none passed, or
        several assigned), or the requested sizes leave a partition empty.

    Notes
    -----
    **Leakage:** Prefer this over shuffled splits for temporal processes.
    The splitter does not add a calendar embargo beyond strict ordering.

    The embargo point matters if your target is measured over a window. A
    label like "churned within 30 days" computed for the last training row
    depends on outcomes that fall inside the test period. Strict ordering
    does not catch that; drop a gap of rows yourself, or build the label so
    its window closes before the boundary.

    Examples
    --------
    >>> import pandas as pd
    >>> from buildml import Session
    >>> frame = pd.DataFrame(
    ...     {
    ...         "day": pd.date_range("2024-01-01", periods=6, freq="D"),
    ...         "demand": [10, 12, 15, 14, 19, 21],
    ...     }
    ... )
    >>> session = Session.ingest(frame).set_roles(
    ...     {"day": "time", "demand": "target"}
    ... )
    >>> _ = session.time_split(test_size=2)
    >>> session.partition("train")["day"].max() < session.partition("test")["day"].min()
    True

    See Also
    --------
    Session.fit_forecast : Forecasting models for temporal targets.
    Session.analyze_timeseries : Inspect trend and seasonality first.
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
    return cast("Session", session)
def partition(
    session, name: PartitionName | Literal['train', 'validation', 'test']
) -> pd.DataFrame:
    """Pull out the rows belonging to one partition, as a DataFrame.

    The split stores row positions, not data. This materialises one of
    those partitions so you can look at it: check class balance, sanity
    check a transform, or hand the frame to code outside BuildML.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    name:
        Which partition to extract: ``'train'``, ``'validation'``, or
        ``'test'``.

    Returns
    -------
    pandas.DataFrame
        A copy of those rows with all current columns, reflecting every
        transform applied so far. Because it is a copy, editing it does not
        change the session's data.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        No split has been created yet, or ``name`` is not one of the three
        partition names. Asking for ``'validation'`` when the split was
        made without one also raises.

    Examples
    --------
    >>> import pandas as pd
    >>> from buildml import Session
    >>> frame = pd.DataFrame({"a": [1, 2, 3, 4], "y": [0, 1, 0, 1]})
    >>> session = Session.ingest(frame).set_roles({"y": "target"})
    >>> _ = session.split(test_size=0.5)
    >>> session.partition("test").shape[0]
    2

    Check that stratification did what you asked:

    >>> session.partition("train")["y"].value_counts(normalize=True)  # doctest: +SKIP
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


def assert_can_fit(session, partition: PartitionName = "train") -> "Session":
    """Refuse to continue unless fitting is confined to the train rows.

    BuildML's own transforms already fit on train alone. This method is for
    the code you write around them: drop it in front of a custom fit step
    and it turns "we should only fit on train" from a comment into a
    runtime guarantee. It raises if no split exists, and it raises if the
    partition you name is anything other than train.

    It is a checkpoint, not a transform. Nothing about the data changes.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    partition:
        The partition you are about to fit on. Only ``'train'`` is
        permitted; naming ``'validation'`` or ``'test'`` is precisely the
        mistake this call exists to catch.

    Returns
    -------
    Session
        ``self``, so the guard sits inline in a chain.

    Raises
    ------
    ~buildml.core.errors.LeakageError
        No split has been created, or ``partition`` is not ``'train'``.

    Examples
    --------
    >>> import pandas as pd
    >>> from buildml import Session
    >>> frame = pd.DataFrame({"a": [1, 2, 3, 4], "y": [0, 1, 0, 1]})
    >>> session = Session.ingest(frame).set_roles({"y": "target"})
    >>> _ = session.split(test_size=0.5)
    >>> train = session.assert_can_fit().partition("train")

    Fitting anything of your own on a holdout is stopped outright:

    >>> session.assert_can_fit("test")
    Traceback (most recent call last):
        ...
    buildml.core.errors.LeakageError: ...

    See Also
    --------
    Session.split : Create the split this guard checks for.
    """
    assert_fit_partition(session._split_plan, partition)
    return cast("Session", session)
def to_engine(session, engine: EngineName | str | None = None) -> Any:
    """Hand back the data as the chosen engine's own object.

    An escape hatch. When you need a real ``polars.DataFrame`` or a DuckDB
    relation to run something BuildML does not expose, this converts the
    current data and returns the native object directly.

    Unlike :meth:`with_engine`, this does not change what the session uses;
    it produces a value for you to work with.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    engine:
        Which engine's type to produce: ``'pandas'``, ``'polars'``, or
        ``'duckdb'``. ``None`` uses the engine the dataset is already set
        to.

    Returns
    -------
    object
        A ``pandas.DataFrame``, ``polars.DataFrame``, or DuckDB relation,
        depending on the engine.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        No dataset is attached.
    ~buildml.core.errors.MissingExtraError
        The requested engine's package is not installed.

    Notes
    -----
    The returned object is detached from the session. Changes you make to
    it do not flow back: call :meth:`sync_native` if you have mutated the
    session's own frame and need the engine table rebuilt.

    See Also
    --------
    Session.with_engine : Change the engine the session itself uses.
    Session.to_pandas : The common case, spelled directly.
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
    """Save the whole session so you can stop and pick up where you left off.

    Long workflows get interrupted: a laptop closes, a job hits its time
    limit, a notebook kernel dies three hours into feature engineering.
    A checkpoint writes the current data, the split membership, the fitted
    preprocessing plans, and the full operation history to disk so
    :meth:`checkpoint_load` can restore all of it later.

    This is for work in progress, not for deployment. It deliberately does
    not embed a fitted estimator; models are inference artefacts and belong
    in a :meth:`save_pipeline` bundle. Think of a checkpoint as saving your
    place, and a pipeline as shipping the result.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    path:
        Destination directory, created if needed.
    sidecar_partition_rows:
        How many rows go in each Parquet file when the data is split across
        several. Defaults to 25,000. Ignored under
        ``sidecar_layout='single'``.
    sidecar_compression:
        Parquet compression codec. Defaults to ``zstd``, which compresses
        well without being slow to read back.
    sidecar_layout:
        How the data files are arranged. ``'auto'`` (the default) writes a
        single file for small data and partitions at 50,000 rows or more.
        ``'single'`` always writes one file, simpler to move around.
        ``'partitioned'`` always splits, which reads back faster for large
        data and lets a reader skip parts of it.

    Returns
    -------
    pathlib.Path
        The checkpoint directory that was written.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        No dataset is attached, or the destination cannot be written.

    Notes
    -----
    The checkpoint records a fingerprint of the data. When it is loaded
    back, that fingerprint is re-checked and the outcome lands on
    :attr:`reattach_result`, so a checkpoint whose underlying data has
    shifted announces itself instead of quietly resuming against something
    different.

    Examples
    --------
    >>> path = session.checkpoint_save("checkpoints/step_3")  # doctest: +SKIP

    Later, in a new process:

    >>> from buildml import Session
    >>> session = Session.checkpoint_load("checkpoints/step_3")  # doctest: +SKIP
    >>> session.reattach_result.status  # doctest: +SKIP
    'clean'

    See Also
    --------
    Session.checkpoint_load : Restore what this writes.
    Session.save_pipeline : Save a deployable model instead.
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


def checkpoint_load_session(session_cls, path: str | Path, *, data_only: bool = False, trusted: bool = False) -> "Session":
    """Restore a saved session and check the data still matches.

    Rebuilds a session from a :meth:`checkpoint_save` bundle: the data, the
    split membership, the fitted preprocessing plans, and the operation
    history all come back, so the audit trail spans the interruption rather
    than restarting at it.

    Restoration is verified, not assumed. The data is re-checked against
    the fingerprint recorded when the checkpoint was written, and the
    outcome lands on :attr:`reattach_result`. Read it before continuing :
    plans fitted against data that has since changed are no longer the
    right plans.

    Parameters
    ----------
    session_cls:
        The :class:`~buildml.session.session.Session` class (classmethod
        receiver) used to construct the restored session instance.
    path:
        The checkpoint directory to restore from.
    data_only:
        Load only the rows and discard the rest: no split, no plans, no
        history. Use this when you want the stored data as a starting point
        for something new, and the previous session's decisions would only
        get in the way. When ``True``, ``plans.joblib`` is skipped and
        ``trusted`` is not required.
    trusted:
        Must be ``True`` to deserialize ``plans.joblib`` (pickle/joblib).
        Pass only for checkpoints you created or fully trust. Defaults to
        ``False``.

    Returns
    -------
    Session
        A new session holding the restored state.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        The directory is not a readable checkpoint, its contents are
        incomplete, or ``plans.joblib`` is present without ``trusted=True``.

    Notes
    -----
    Checkpoints do not embed a fitted estimator. Use :meth:`load_pipeline`
    for inference artefacts; the two are complementary, and a workflow that
    both resumes and ships will use each for its own purpose.

    **Security:** pickle/joblib loads are opt-in via ``trusted=True``. Prefer
    ``data_only=True`` when provenance is unclear.

    Examples
    --------
    >>> from buildml import Session
    >>> session = Session.checkpoint_load("checkpoints/step_3")  # doctest: +SKIP
    >>> session.reattach_result.status  # doctest: +SKIP
    'clean'
    >>> len(session.history)  # doctest: +SKIP
    14

    See Also
    --------
    Session.checkpoint_save : Write the bundle this reads.
    Session.reattach : Restore into an existing session instead.
    Session.reattach_result : The verification outcome to check.
    """
    loaded = load_checkpoint(path, data_only=data_only, trusted=trusted)
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
    return cast("Session", session)
def reattach(
    session, path: str | Path, *, data_only: bool = False, trusted: bool = False
) -> "Session":
    """Replace this session's state from a checkpoint, in place.

    Does what :meth:`checkpoint_load` does, except it overwrites the
    current session rather than returning a new one. Useful in a loop or a
    long-lived process where you want to keep the same session object
    while swapping what it holds.

    Any native engine connection this session owns is closed first, so
    resources are not leaked across the swap.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    path:
        The checkpoint directory to restore from.
    data_only:
        Restore only the rows, clearing the split, plans, and history.
        When ``True``, ``plans.joblib`` is skipped and ``trusted`` is not
        required.
    trusted:
        Must be ``True`` to deserialize ``plans.joblib`` (pickle/joblib).
        Pass only for checkpoints you created or fully trust. Defaults to
        ``False``.

    Returns
    -------
    Session
        ``self``, now holding the restored state.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        The directory is not a readable checkpoint, or ``plans.joblib`` is
        present without ``trusted=True``.

    Notes
    -----
    Everything the session currently holds is discarded. Save first if the
    current state matters.

    See Also
    --------
    Session.checkpoint_load : The classmethod form, returning a new session.
    """
    loaded = load_checkpoint(path, data_only=data_only, trusted=trusted)
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
    return cast("Session", session)
def to_pandas(session) -> pd.DataFrame:
    """Take the data out as a plain Pandas DataFrame.

    The escape hatch. When you need to do something BuildML does not cover,
    this hands you an ordinary DataFrame to work with. If the data is
    currently held by Polars or DuckDB, it is materialised into memory
    here.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.

    Returns
    -------
    pandas.DataFrame
        A copy of the current data with every transform applied so far.
        Because it is a copy, editing it does not affect the session.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        No dataset is attached.

    Notes
    -----
    **Scale:** On a lazy or engine-backed dataset this forces full
    materialisation. That is precisely what the engine paths exist to
    avoid, so on a large table it may be slow or exhaust memory. Reach for
    :meth:`head` to look, or :meth:`prepare_design_matrix` to narrow first.

    See Also
    --------
    Session.head : A small preview instead of the whole table.
    Session.partition : One partition rather than everything.
    """
    frame = session.dataset.to_pandas()
    session._record(
        "to_pandas", result_summary={"rows": int(len(frame)), "columns": int(frame.shape[1])}
    )
    return cast(pd.DataFrame, frame)


def to_parquet(session, path: str | Path) -> Path:
    """Write the current data to a Parquet file.

    Parquet stores columns rather than rows, which makes it much smaller
    than CSV and much faster to read back: and unlike CSV it preserves
    dtypes, so a datetime column returns as a datetime rather than as text
    you have to re-parse.

    Use this to hand transformed data to another tool, or to save an
    intermediate result you do not want to recompute.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    path:
        Destination file path.

    Returns
    -------
    pathlib.Path
        Where the file was written.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        No dataset is attached, or the destination cannot be written.

    Notes
    -----
    This writes the data only. Roles, split membership, and fitted plans
    are not included: :meth:`checkpoint_save` is the option that preserves
    those.

    See Also
    --------
    Session.checkpoint_save : Save the session, not just the table.
    Session.ingest : Read a Parquet file back in.
    """
    destination = session.dataset.to_parquet(path)
    session._record(
        "to_parquet", {"path": str(destination)}, result_summary={"path": str(destination)}
    )
    return cast(Path, destination)


def head(session, n: int = 5) -> pd.DataFrame:
    """Look at the first few rows.

    The quickest way to see what you are working with, and worth doing
    after every transform: a column that has become all zeros or all
    ``NaN`` shows up immediately here and can otherwise go unnoticed until
    the model underperforms for no visible reason.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    n:
        How many rows to return.

    Returns
    -------
    pandas.DataFrame
        The first ``n`` rows with all current columns.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        No dataset is attached.

    Notes
    -----
    Only the requested rows are materialised, so this stays cheap on
    engine-backed data where :meth:`to_pandas` would not be.

    These are the first rows in storage order, not a random sample. If the
    file is sorted, they are not representative: use :meth:`eda` for a
    picture of the whole table.

    Examples
    --------
    >>> import pandas as pd
    >>> from buildml import Session
    >>> session = Session.ingest(pd.DataFrame({"a": range(100)}))
    >>> session.head(3).shape
    (3, 1)

    See Also
    --------
    Session.eda : A full profile rather than a glance.
    """
    frame = session.dataset.head(n)
    session._record(
        "head", {"n": n}, result_summary={"rows": int(len(frame)), "columns": int(frame.shape[1])}
    )
    return cast(pd.DataFrame, frame)


def with_mode(session, mode: DataMode | str) -> "Session":
    """Set whether data is held in memory or kept lazy.

    ``'memory'`` means the rows are fully materialised and every operation
    works on them directly. ``'lazy'`` means the dataset keeps an engine
    handle and defers materialising until something genuinely requires it :
    which is how a table larger than memory stays workable.

    This records the intent on the dataset. Whether laziness actually
    happens depends on the engine: it is real for Polars and DuckDB and
    cannot apply to a Pandas-backed frame, which is already in memory by
    definition.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    mode:
        ``'memory'`` or ``'lazy'``. The historical value ``'out_of_core'``
        is accepted and coerced to ``'lazy'``; there is no separate
        out-of-core fit path.

    Returns
    -------
    Session
        ``self``, so this call chains.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        No dataset is attached, or the mode is not a recognised value.

    Notes
    -----
    Lazy mode defers work; it does not make scikit-learn out-of-core. The
    estimator still needs an in-memory matrix at fit time. What laziness
    buys you is the chance to filter and project first, so that only the
    reduced result has to fit.

    See Also
    --------
    Session.with_engine : Choose the engine that makes lazy meaningful.
    Session.prepare_design_matrix : Narrow the data before materialising.
    """
    session.dataset.mode = coerce_data_mode(mode)
    session._record("with_mode", {"mode": session.dataset.mode.value})
    return cast("Session", session)
def with_engine(session, engine: EngineName | str) -> "Session":
    """Switch the compute engine backing the data.

    Pandas is the default and the right choice for anything that
    comfortably fits in memory. Polars and DuckDB exist for when it does
    not: both hold the data in their own columnar format and can filter,
    project, and aggregate over it far faster and with less memory than
    Pandas.

    Choosing between them is mostly about how you like to express things.
    Polars offers a DataFrame API with strong lazy evaluation; DuckDB lets
    you write SQL against the table. Either way, BuildML attaches a native
    handle that :meth:`prepare_design_matrix` and the filter and sample
    helpers use to reduce the data before anything crosses into Pandas.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    engine:
        ``'pandas'``, ``'polars'``, or ``'duckdb'``. The latter two require
        ``pip install 'buildml[engines]'``.

    Returns
    -------
    Session
        ``self``, so this call chains.

    Raises
    ------
    ~buildml.core.errors.MissingExtraError
        The requested engine's package is not installed.
    ~buildml.core.errors.ValidationError
        No dataset is attached, or the engine name is not recognised.

    Notes
    -----
    Switching to Pandas releases any native handle; switching to Polars or
    DuckDB builds one. DuckDB's handle holds a connection, so close it with
    :meth:`close_native` or use ``with session:``.

    The estimator boundary is unchanged. scikit-learn still requires an
    in-memory matrix, so the engine's value lies in everything that happens
    before the fit.

    Examples
    --------
    >>> session = Session.ingest("events.parquet")  # doctest: +SKIP
    >>> with session.with_engine("duckdb") as s:  # doctest: +SKIP
    ...     prepared = s.prepare_design_matrix(sample_rows=500_000)

    See Also
    --------
    Session.to_engine : Get a native object without switching.
    Session.close_native : Release a DuckDB connection.
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
    return cast("Session", session)
def sync_native(session) -> "Session":
    """Rebuild the engine's table from the current Pandas frame.

    With a Polars or DuckDB engine attached, the data exists in two places:
    the engine's native table and a Pandas cache. BuildML's own transforms
    keep them in step. Code outside BuildML that reaches in and edits
    ``dataset.frame`` directly does not, leaving the engine table stale.

    This resynchronises them, converting the current frame into a fresh
    engine table. On a Pandas-backed dataset there is nothing to sync and
    the call simply records that.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.

    Returns
    -------
    Session
        ``self``, so this call chains.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        No dataset is attached.

    Notes
    -----
    This is eager and total: the whole current frame is converted. It does
    not replay earlier steps as a lazy plan, so on a large table it costs
    what a full conversion costs.

    See Also
    --------
    Session.with_engine : Attach the engine this keeps in step.
    """
    has_native = False
    if session.dataset.engine != EngineName.PANDAS:
        session.dataset.sync_native()
        has_native = session.dataset.has_native
    session._record(
        "sync_native", {"engine": session.dataset.engine.value, "has_native": has_native}
    )
    return cast("Session", session)
def metadata(session) -> dict[str, Any]:
    """Take a serialisable snapshot of everything the session knows.

    Returns the session's state as plain dictionaries and lists: no
    BuildML objects: so it can be written to JSON, logged, compared
    between runs, or attached to an experiment tracker.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.

    Returns
    -------
    dict
        Whether a dataset is attached, the ingest report, the split plan,
        the full operation history, the checkpoint reattach outcome, and
        the dataset's own metadata (schema, roles, row count, engine).
        Contains no row data, so it is safe to log.

    Notes
    -----
    Useful as a run fingerprint. Diffing two runs' metadata is a fast way
    to find why yesterday's numbers and today's disagree.

    See Also
    --------
    Session.history : The operation record on its own.
    Session.summarize_history : A readable summary rather than raw state.
    """
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
