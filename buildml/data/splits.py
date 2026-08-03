"""Decide which rows a model may learn from, and prove it never saw the rest.

A model evaluated on data it was trained on will report a score it cannot
reproduce on anything new. Splitting is how that is avoided, and the whole of
this module exists to make the split correct and to keep it correct.

A :class:`SplitPlan` is membership, not data — positional indices into the
frame. Nothing is copied, so a plan is cheap to carry and can be serialised
alongside a model to record exactly which rows it was fitted on.

Four strategies, because "correct" depends on the data. :func:`create_split`
shuffles rows at random, optionally stratified so class proportions hold in each
partition. :func:`create_group_split` keeps every row of a group together, which
matters whenever rows are not independent — repeated measurements of one
patient, several orders from one customer. :func:`create_time_split` cuts
chronologically, because predicting the past from the future is not a problem
anyone has. :func:`inject_partitions` accepts membership you determined
elsewhere.

The invariants are checked rather than assumed. Every plan asserts disjoint
partitions; group splits verify no group crosses a boundary; time splits verify
no training timestamp lands after a test one. Violations raise
:class:`~buildml.core.errors.LeakageError`, and :func:`assert_fit_partition`
refuses any fit that has no split at all.

Notes
-----
**The strategy matters more than the fraction.** A random split of grouped data
gives an inflated score that looks fine, and choosing 80/20 over 75/25 will not
save it.

See Also
--------
buildml.data.dataset.Dataset : What gets split.
buildml.core.errors.LeakageError : What a violation raises.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from buildml.core.errors import LeakageError, ValidationError
from buildml.core.types import ColumnRole
from buildml.data.dataset import Dataset

PartitionName = Literal["train", "validation", "test"]


@dataclass(slots=True)
class SplitPlan:
    """Which rows belong to which partition, and how that was decided.

    Membership, not data. The indices are positions in the frame, so a plan
    stays small and can be stored beside a model as a record of exactly what it
    was fitted on.

    Attributes
    ----------
    kind:
        How it was made — ``'random'``, ``'stratified'``, ``'group'``,
        ``'time'``, or ``'injected'``. **Read this before trusting a score**: a
        random split of grouped data is the most common cause of an optimistic
        result.
    test_size:
        The fraction or count requested for test.
    validation_size:
        The same for validation, or ``None`` when there is no validation
        partition.
    random_state:
        The seed. ``None`` for time splits, which do not shuffle, and for
        injected plans.
    stratify_column:
        The column that shaped the split — the stratification target, the group
        column, or the time column, depending on ``kind``.
    train_indices:
        Positions a model may learn from.
    validation_indices:
        Positions for tuning. Empty when none was carved.
    test_indices:
        Positions reserved for the final estimate.

    Notes
    -----
    **Indices are positional and tied to the current frame.** Reordering or
    filtering rows after building a plan silently invalidates it — the indices
    will still resolve, and will point at different rows. Split after the frame
    is settled.

    **Every partition a model tunes against stops being a clean estimate.**
    Choosing between models on validation is why a separate test partition
    exists; consulting test repeatedly turns it into validation.

    See Also
    --------
    create_split : Random and stratified.
    create_group_split : Keeping groups intact.
    create_time_split : Chronological.
    """

    kind: str
    test_size: float | int | None
    validation_size: float | int | None
    random_state: int | None
    stratify_column: str | None
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    test_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the split as JSON-safe values, indices included.

        Complete and round-trippable. The indices are written in full rather
        than summarised, because a plan without them is a description of a
        split rather than the split itself.

        Returns
        -------
        dict
            Kind, sizes, seed, the shaping column, and all three index lists.

        Notes
        -----
        **This grows with the dataset.** A million-row split serialises a
        million indices. That is the cost of being able to reproduce a fit
        exactly.

        See Also
        --------
        from_dict : The inverse.
        """
        return {
            "kind": self.kind,
            "test_size": self.test_size,
            "validation_size": self.validation_size,
            "random_state": self.random_state,
            "stratify_column": self.stratify_column,
            "train_indices": list(self.train_indices),
            "validation_indices": list(self.validation_indices),
            "test_indices": list(self.test_indices),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SplitPlan:
        """Rebuild a split from its serialised form.

        Restores the exact membership, which is how a reloaded checkpoint
        continues against the same partitions rather than a fresh split that
        happens to be the same size.

        Parameters
        ----------
        payload:
            A mapping from :meth:`to_dict`.

        Returns
        -------
        SplitPlan
            The reconstructed plan.

        Raises
        ------
        KeyError
            If ``kind`` is absent. Everything else defaults, since a plan with
            no validation partition legitimately has none.

        Notes
        -----
        **Nothing is validated against a dataset here.** A plan restored
        against a differently ordered or differently sized frame will resolve
        to the wrong rows. Restore the data and the plan together.
        """
        return cls(
            kind=str(payload["kind"]),
            test_size=payload.get("test_size"),
            validation_size=payload.get("validation_size"),
            random_state=payload.get("random_state"),
            stratify_column=payload.get("stratify_column"),
            train_indices=tuple(int(i) for i in payload.get("train_indices", [])),
            validation_indices=tuple(int(i) for i in payload.get("validation_indices", [])),
            test_indices=tuple(int(i) for i in payload.get("test_indices", [])),
        )

    def indices_for(self, partition: PartitionName) -> tuple[int, ...]:
        """Return the row positions belonging to one partition.

        The accessor everything else uses to slice a frame, so that no caller
        has to know which attribute holds which partition.

        Parameters
        ----------
        partition:
            ``'train'``, ``'validation'``, or ``'test'``.

        Returns
        -------
        tuple of int
            Positions in the frame. Empty when the partition was never carved.

        Raises
        ------
        ValidationError
            If the name is not one of the three.

        Notes
        -----
        **An empty result is not an error.** A plan built without a validation
        partition returns an empty tuple for it, and slicing by that yields an
        empty frame rather than raising.

        See Also
        --------
        frame_for_partition : The rows themselves.
        """
        if partition == "train":
            return self.train_indices
        if partition == "validation":
            return self.validation_indices
        if partition == "test":
            return self.test_indices
        raise ValidationError(f"Unknown partition '{partition}'")

    def assert_disjoint(self) -> None:
        """Verify no row belongs to two partitions, and that the split is usable.

        Called by every constructor in this module. A row appearing in both
        train and test is the purest form of leakage — the model has memorised
        the answer — and it is cheap enough to check that there is no reason
        to assume it away.

        Returns
        -------
        None
            Returns nothing on success; the value is the absence of an
            exception.

        Raises
        ------
        ValidationError
            If any two partitions share a row, or if train or test is empty.

        Notes
        -----
        **Disjoint indices are not the same as disjoint information.**
        Duplicate rows with different positions pass this check and still leak.
        Deduplicate before splitting, or split on a group.

        See Also
        --------
        assert_fit_partition : The guard on fitting.
        """
        train, valid, test = (
            set(self.train_indices),
            set(self.validation_indices),
            set(self.test_indices),
        )
        if train & valid or train & test or valid & test:
            raise ValidationError("Split partitions are not disjoint")
        if not train or not test:
            raise ValidationError("Split must include non-empty train and test partitions")


def create_split(
    dataset: Dataset,
    *,
    test_size: float | int = 0.2,
    validation_size: float | int | None = None,
    random_state: int | None = 42,
    stratify: bool = False,
) -> SplitPlan:
    """Split rows at random, optionally preserving class balance.

    The default strategy, and the right one when rows are independent — each
    row a separate observation, with no shared subject, session, or entity tying
    any two together.

    With ``stratify``, class proportions are held roughly constant across
    partitions. This matters most when a class is rare: an unstratified split of
    a 2% positive rate can leave a test partition with almost none, making the
    score mostly noise.

    Parameters
    ----------
    dataset:
        What to split.
    test_size:
        A fraction in (0, 1), or an absolute row count.
    validation_size:
        Carved from what remains after test, so a 0.2 test and 0.2 validation
        leaves 64% for training rather than 60%.
    random_state:
        The seed. Fixed by default, because a split that changes between runs
        makes every comparison meaningless.
    stratify:
        Preserve the target's class proportions. Classification only.

    Returns
    -------
    SplitPlan
        Verified disjoint membership.

    Raises
    ------
    ValidationError
        If the dataset has fewer than two rows, if stratification is requested
        with no target role, or if the sizes are out of range.

    Notes
    -----
    **Random splitting assumes independent rows, and quietly gives a wrong
    answer when they are not.** If the same patient, customer, or device appears
    in several rows, use :func:`create_group_split`. If rows are ordered in
    time, use :func:`create_time_split`. The inflated score a random split
    produces in those cases looks entirely reasonable.

    **Stratification needs enough of every class.** Scikit-learn refuses when a
    class has fewer members than partitions; the error names the class.

    Examples
    --------
    A stratified split with a validation partition::

        plan = create_split(
            dataset, test_size=0.2, validation_size=0.2, stratify=True,
        )
        len(plan.train_indices), len(plan.test_indices)

    See Also
    --------
    create_group_split : When rows share an entity.
    create_time_split : When rows are ordered.
    assert_fit_partition : The guard that uses this.
    """
    n_rows = dataset.n_rows
    if n_rows < 2:
        raise ValidationError("Need at least 2 rows to create a train/test split")

    indices = np.arange(n_rows)
    stratify_column: str | None = None
    stratify_values: pd.Series | None = None
    kind = "random"

    if stratify:
        stratify_column = dataset.require_target()
        stratify_values = dataset._ensure_pandas()[stratify_column]
        kind = "stratified"

    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_values,
    )

    validation_indices: tuple[int, ...] = ()
    if validation_size is not None:
        y_train = None if stratify_values is None else stratify_values.iloc[train_idx]
        train_idx, valid_idx = train_test_split(
            train_idx,
            test_size=validation_size,
            random_state=random_state,
            stratify=y_train,
        )
        validation_indices = tuple(int(i) for i in valid_idx)

    plan = SplitPlan(
        kind=kind,
        test_size=test_size,
        validation_size=validation_size,
        random_state=random_state,
        stratify_column=stratify_column,
        train_indices=tuple(int(i) for i in train_idx),
        validation_indices=validation_indices,
        test_indices=tuple(int(i) for i in test_idx),
    )
    plan.assert_disjoint()
    return plan


def create_group_split(
    dataset: Dataset,
    *,
    test_size: float | int = 0.2,
    validation_size: float | int | None = None,
    random_state: int | None = 42,
    group_column: str | None = None,
) -> SplitPlan:
    """Split so that no group is ever split, keeping related rows together.

    When several rows describe the same entity — visits by one patient, orders
    by one customer, readings from one sensor — a random split puts some of that
    entity's rows in train and some in test. The model then recognises the
    entity rather than learning the pattern, and reports a score it will not
    reproduce on anyone new.

    This assigns whole groups to partitions, so a group appears in exactly one.

    Parameters
    ----------
    dataset:
        Data with a ``group`` role, or use ``group_column``.
    test_size:
        A fraction or count **of groups, not rows**.
    validation_size:
        The same, carved from the training groups.
    random_state:
        The seed.
    group_column:
        Which column identifies the group. Defaults to the sole ``group`` role.

    Returns
    -------
    SplitPlan
        Membership, verified both disjoint and group-clean.

    Raises
    ------
    ValidationError
        If no group column can be resolved, if there are fewer than two
        distinct groups, or if too few training groups remain to carve a
        validation partition.
    LeakageError
        If a group nonetheless appears in two partitions. A defensive check on
        the splitter's output.

    Notes
    -----
    **Row counts will not match your fractions, and that is correct.** Groups
    vary in size, so requesting 20% of groups might yield 12% or 31% of rows.
    Grouping is the thing being controlled; row balance is what gets traded for
    it.

    **Class balance is not preserved.** Stratified grouping is a harder problem
    and is not attempted here. Check the class distribution in each partition
    afterwards if a class is rare.

    Examples
    --------
    Hold out whole patients::

        plan = create_group_split(
            dataset, test_size=0.2, group_column="patient_id",
        )

    See Also
    --------
    create_split : When rows are independent.
    """
    column = _resolve_group_column(dataset, group_column)
    groups = dataset._ensure_pandas()[column]
    n_groups = int(groups.nunique(dropna=False))
    if n_groups < 2:
        raise ValidationError("Group split requires at least 2 distinct groups")

    indices = np.arange(dataset.n_rows)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_pool, test_idx = next(splitter.split(indices, groups=groups))

    validation_indices: tuple[int, ...] = ()
    train_idx = np.asarray(train_pool)
    if validation_size is not None:
        train_groups = groups.iloc[train_idx]
        if int(train_groups.nunique(dropna=False)) < 2:
            raise ValidationError("Not enough training groups to carve a validation partition")
        valid_splitter = GroupShuffleSplit(
            n_splits=1, test_size=validation_size, random_state=random_state
        )
        inner_train_pos, inner_valid_pos = next(
            valid_splitter.split(np.arange(len(train_idx)), groups=train_groups)
        )
        validation_indices = tuple(int(i) for i in train_idx[inner_valid_pos])
        train_idx = train_idx[inner_train_pos]

    plan = SplitPlan(
        kind="group",
        test_size=test_size,
        validation_size=validation_size,
        random_state=random_state,
        stratify_column=column,
        train_indices=tuple(int(i) for i in train_idx),
        validation_indices=validation_indices,
        test_indices=tuple(int(i) for i in test_idx),
    )
    plan.assert_disjoint()
    _assert_groups_disjoint(dataset, plan, column)
    return plan


def create_time_split(
    dataset: Dataset,
    *,
    test_size: float | int = 0.2,
    validation_size: float | int | None = None,
    time_column: str | None = None,
) -> SplitPlan:
    """Split chronologically: train on the past, test on the future.

    Sorts by timestamp and cuts. The earliest rows train, the latest test, and
    a validation partition sits between them. Nothing is shuffled.

    This mirrors how a deployed model actually works — it will only ever see
    data from before the moment it predicts. A random split on time-ordered data
    lets the model train on Thursday and predict Tuesday, which inflates the
    score by an amount nobody can estimate afterwards.

    Parameters
    ----------
    dataset:
        Data with a ``time`` role, or use ``time_column``.
    test_size:
        A fraction or count of rows, taken from the most recent end.
    validation_size:
        The same, taken from the most recent end of what remains — so the
        ordering is train, then validation, then test.
    time_column:
        Which column holds the timestamps. Defaults to the sole ``time`` role.

    Returns
    -------
    SplitPlan
        Membership, verified disjoint and chronologically ordered. Its
        ``random_state`` is ``None``, since nothing was randomised.

    Raises
    ------
    ValidationError
        If no time column can be resolved, if any timestamp cannot be parsed,
        or if the sizes would leave a partition empty.
    LeakageError
        If a training timestamp lands after a validation or test one. A
        defensive check on the sort.

    Notes
    -----
    **Every timestamp must parse.** Unparseable values are refused rather than
    dropped or sorted arbitrarily, since either would put rows on the wrong side
    of the cut without saying so.

    **Ties are broken stably.** Rows sharing a timestamp keep their original
    relative order, so the split is reproducible — but rows on the boundary
    could have gone either way, and if many rows share the cut timestamp that is
    worth knowing about.

    **The test partition is one period, not a sample of periods.** A model
    scored against a single unusual month is scored against that month. Rolling
    evaluation, in :mod:`buildml.timeseries`, addresses this.

    Examples
    --------
    Hold out the most recent fifth::

        plan = create_time_split(
            dataset, test_size=0.2, time_column="order_date",
        )

    See Also
    --------
    create_split : When order does not matter.
    buildml.timeseries : Rolling-origin evaluation.
    """
    column = _resolve_time_column(dataset, time_column)
    stamps = pd.to_datetime(dataset._ensure_pandas()[column], errors="coerce")
    if stamps.isna().any():
        raise ValidationError(
            f"Time split requires parseable timestamps in '{column}'; "
            f"found {int(stamps.isna().sum())} invalid value(s)"
        )

    order = np.argsort(stamps.to_numpy(), kind="mergesort")
    n_rows = len(order)
    n_test = _absolute_size(test_size, n_rows)
    if n_test < 1 or n_test >= n_rows:
        raise ValidationError("test_size must leave at least one train row and one test row")

    test_idx = order[-n_test:]
    remaining = order[:-n_test]
    validation_indices: tuple[int, ...] = ()
    if validation_size is not None:
        n_valid = _absolute_size(validation_size, len(remaining))
        if n_valid < 1 or n_valid >= len(remaining):
            raise ValidationError(
                "validation_size must leave at least one train row when carving from the train pool"
            )
        validation_indices = tuple(int(i) for i in remaining[-n_valid:])
        train_idx = remaining[:-n_valid]
    else:
        train_idx = remaining

    plan = SplitPlan(
        kind="time",
        test_size=test_size,
        validation_size=validation_size,
        random_state=None,
        stratify_column=column,
        train_indices=tuple(int(i) for i in train_idx),
        validation_indices=validation_indices,
        test_indices=tuple(int(i) for i in test_idx),
    )
    plan.assert_disjoint()
    _assert_time_order(dataset, plan, column)
    return plan


def inject_partitions(
    dataset: Dataset,
    *,
    train_indices: list[int] | tuple[int, ...],
    test_indices: list[int] | tuple[int, ...],
    validation_indices: list[int] | tuple[int, ...] | None = None,
) -> SplitPlan:
    """Adopt a split you determined elsewhere.

    For membership that comes from outside BuildML — a competition's official
    split, a partition your team agreed on, or a scheme none of the built-in
    strategies expresses.

    The escape hatch is real but not unguarded. Indices are bounds-checked and
    the result must still be disjoint, so an injected plan cannot smuggle in an
    overlap the built-in strategies would have refused.

    Parameters
    ----------
    dataset:
        The data the indices refer to.
    train_indices:
        Positions a model may learn from.
    test_indices:
        Positions reserved for evaluation.
    validation_indices:
        Optional positions for tuning.

    Returns
    -------
    SplitPlan
        Membership with ``kind='injected'``, verified disjoint.

    Raises
    ------
    ValidationError
        If any index is out of range, if the partitions overlap, or if train or
        test is empty.

    Notes
    -----
    **Only disjointness is checked.** Group and time invariants are not — the
    plan does not record what scheme you intended, so there is nothing to verify
    against. An injected split that leaks groups will be accepted.

    **Rows in no partition are silently excluded.** The indices need not cover
    the frame, and any row you omit takes no part in anything.

    **Positions, not labels.** These are positions in the frame, so a plan built
    against a DataFrame index rather than its position will point at the wrong
    rows.

    See Also
    --------
    create_split : Letting BuildML decide.
    SplitPlan.assert_disjoint : The check applied here.
    """
    n_rows = dataset.n_rows
    for name, values in {
        "train": train_indices,
        "test": test_indices,
        "validation": validation_indices or (),
    }.items():
        for idx in values:
            if idx < 0 or idx >= n_rows:
                raise ValidationError(f"{name} index {idx} out of range for {n_rows} rows")

    plan = SplitPlan(
        kind="injected",
        test_size=None,
        validation_size=None,
        random_state=None,
        stratify_column=None,
        train_indices=tuple(int(i) for i in train_indices),
        validation_indices=tuple(int(i) for i in (validation_indices or ())),
        test_indices=tuple(int(i) for i in test_indices),
    )
    plan.assert_disjoint()
    return plan


def frame_for_partition(
    dataset: Dataset,
    plan: SplitPlan,
    partition: PartitionName,
) -> pd.DataFrame:
    """Return the rows of one partition, as an independent copy.

    Turns membership into data. A copy is returned so that modifying the
    partition cannot reach back into the dataset, which would be a leak in the
    other direction.

    Parameters
    ----------
    dataset:
        The data.
    plan:
        Which rows belong where.
    partition:
        ``'train'``, ``'validation'``, or ``'test'``.

    Returns
    -------
    pandas.DataFrame
        The rows, in plan order.

    Raises
    ------
    ValidationError
        If the partition name is unrecognised, or if ``'validation'`` is
        requested from a plan that has none — an empty frame there would look
        like an empty split rather than an absent one.

    Notes
    -----
    **This copies.** On a large partition that is real memory. Where a view
    would do, slice with :meth:`SplitPlan.indices_for` instead.

    See Also
    --------
    SplitPlan.indices_for : The positions without the copy.
    """
    indices = plan.indices_for(partition)
    if not indices and partition == "validation":
        raise ValidationError("No validation partition exists on this split plan")
    return dataset._ensure_pandas().iloc[list(indices)].copy()


def assert_fit_partition(plan: SplitPlan | None, partition: PartitionName = "train") -> None:
    """Refuse to fit on anything except a real training partition.

    Called at the top of every operation that learns something — imputation
    statistics, encoder vocabularies, scaler parameters, model weights. It
    enforces two rules: a split must exist, and the fit must be on train.

    The first rule is the one that matters most. Fitting on full data before
    splitting is the leak that produces the most convincing wrong number,
    because nothing about it looks unusual — the code runs, the score is good,
    and the model fails in production for reasons nobody can reconstruct.

    Parameters
    ----------
    plan:
        The current split. ``None`` means none was made.
    partition:
        Which partition the caller intends to fit on.

    Returns
    -------
    None
        Returns nothing on success; the value is the absence of an exception.

    Raises
    ------
    LeakageError
        If there is no split, if the partition is not ``'train'``, or if the
        training partition is empty.

    Notes
    -----
    **The refusal is deliberate friction.** Every alternative — a warning, a
    default, an inferred split — leaves a path to a wrong number that looks
    right. :func:`inject_partitions` is available when you genuinely own the
    split.

    **Cross-validation refits per fold** and calls this against each fold's
    training rows, so the guard holds there too.

    See Also
    --------
    create_split : Making a plan to satisfy this.
    inject_partitions : Supplying your own.
    """
    if plan is None:
        raise LeakageError(
            "Refusing fit-capable operation on full data. "
            "Create a split with session.split(...) first, or inject partitions explicitly."
        )
    if partition != "train":
        raise LeakageError(
            f"Refusing to fit on partition '{partition}'. Fit only on 'train' "
            "(or use CV helpers that refit per fold)."
        )
    if not plan.train_indices:
        raise LeakageError("Train partition is empty; cannot fit.")


def _resolve_group_column(dataset: Dataset, group_column: str | None) -> str:
    if group_column is not None:
        if group_column not in dataset.columns:
            raise ValidationError(f"Group column '{group_column}' not found in dataset")
        return group_column
    group_cols = dataset.role_columns(ColumnRole.GROUP)
    if not group_cols:
        raise ValidationError(
            "Group split requires a column with role 'group' "
            "(or pass group_column=... explicitly)"
        )
    if len(group_cols) != 1:
        raise ValidationError(
            "Group split expects exactly one group-role column; "
            f"found {group_cols}. Pass group_column=... to select one."
        )
    return group_cols[0]


def _resolve_time_column(dataset: Dataset, time_column: str | None) -> str:
    if time_column is not None:
        if time_column not in dataset.columns:
            raise ValidationError(f"Time column '{time_column}' not found in dataset")
        return time_column
    time_cols = dataset.role_columns(ColumnRole.TIME)
    if not time_cols:
        raise ValidationError(
            "Time split requires a column with role 'time' "
            "(or pass time_column=... explicitly)"
        )
    if len(time_cols) != 1:
        raise ValidationError(
            "Time split expects exactly one time-role column; "
            f"found {time_cols}. Pass time_column=... to select one."
        )
    return time_cols[0]


def _absolute_size(size: float | int, n_rows: int) -> int:
    if isinstance(size, float):
        if not 0.0 < size < 1.0:
            raise ValidationError("Fractional sizes must be in (0, 1)")
        return max(1, int(round(size * n_rows)))
    if isinstance(size, int):
        if size < 1:
            raise ValidationError("Absolute sizes must be >= 1")
        return int(size)
    raise ValidationError(f"Unsupported size type: {type(size).__name__}")


def _assert_groups_disjoint(dataset: Dataset, plan: SplitPlan, column: str) -> None:
    frame = dataset._ensure_pandas()
    partitions = {
        "train": set(plan.train_indices),
        "validation": set(plan.validation_indices),
        "test": set(plan.test_indices),
    }
    group_sets = {
        name: set(frame.iloc[list(indices)][column].tolist()) if indices else set()
        for name, indices in partitions.items()
    }
    for left, right in (("train", "test"), ("train", "validation"), ("validation", "test")):
        overlap = group_sets[left] & group_sets[right]
        if overlap:
            raise LeakageError(
                f"Group leakage between {left} and {right} on '{column}': "
                f"{sorted(overlap)[:5]}"
            )


def _assert_time_order(dataset: Dataset, plan: SplitPlan, column: str) -> None:
    stamps = pd.to_datetime(dataset._ensure_pandas()[column], errors="coerce")
    train_max = stamps.iloc[list(plan.train_indices)].max()
    test_min = stamps.iloc[list(plan.test_indices)].min()
    if plan.validation_indices:
        valid_stamps = stamps.iloc[list(plan.validation_indices)]
        valid_min = valid_stamps.min()
        valid_max = valid_stamps.max()
        if train_max > valid_min:
            raise LeakageError(
                "Time split invariant failed: a training timestamp is later than validation"
            )
        if valid_max > test_min:
            raise LeakageError(
                "Time split invariant failed: a validation timestamp is later than test"
            )
    elif train_max > test_min:
        raise LeakageError(
            "Time split invariant failed: a training timestamp is later than test"
        )

