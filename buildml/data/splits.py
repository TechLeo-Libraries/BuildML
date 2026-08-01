"""Train/validation/test split planning and membership."""

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
    """Split recipe and row-index membership for a dataset.

    Parameters
    ----------
    kind:
        Split strategy name (``random``, ``stratified``, ``group``, ``time``,
        or ``injected``).
    test_size:
        Fraction or count used for the test partition when created by BuildML.
    validation_size:
        Optional fraction or count for a validation partition.
    random_state:
        Seed used for reproducible splitting.
    stratify_column:
        Column used for stratification, if any.
    train_indices / validation_indices / test_indices:
        Positional indices into the current dataset frame.
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
        if partition == "train":
            return self.train_indices
        if partition == "validation":
            return self.validation_indices
        if partition == "test":
            return self.test_indices
        raise ValidationError(f"Unknown partition '{partition}'")

    def assert_disjoint(self) -> None:
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
    """Create a random or stratified split over dataset row positions.

    Parameters
    ----------
    dataset:
        Dataset to split.
    test_size:
        Test fraction or absolute count.
    validation_size:
        Optional validation fraction/count taken from the remaining train pool.
    random_state:
        RNG seed.
    stratify:
        If True, stratify using the dataset target column.

    Returns
    -------
    SplitPlan
        Disjoint membership plan.

    Raises
    ------
    ValidationError
        If stratification is requested without a target role, or sizes are invalid.

    Notes
    -----
    **Leakage:** Modeling fits must use the train partition only after a split
    exists. Use :func:`assert_fit_partition` before fit-capable operations.
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
    """Create a group-aware split so no group appears in more than one partition.

    Parameters
    ----------
    dataset:
        Dataset with a ``group`` role column (or ``group_column`` override).
    test_size / validation_size:
        Fractions or counts interpreted over **groups**, not rows.
    random_state:
        RNG seed.
    group_column:
        Optional explicit group column. Defaults to the sole ``group`` role.

    Raises
    ------
    ValidationError
        If no group column exists, groups are too few, or sizes are invalid.
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
    """Create a chronological split ordered by a time-role column.

    Earlier rows form train (and optional validation); the latest rows form
    test. No random shuffle is applied.

    Parameters
    ----------
    dataset:
        Dataset with a ``time`` role column (or ``time_column`` override).
    test_size / validation_size:
        Fraction or absolute count of rows (after time ordering).
    time_column:
        Optional explicit time column. Defaults to the sole ``time`` role.

    Raises
    ------
    ValidationError
        If the time column is missing, unparseable, or sizes leave empty
        partitions.
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
    """Inject externally owned partition membership.

    Parameters
    ----------
    dataset:
        Dataset whose positional indices are referenced.
    train_indices / test_indices / validation_indices:
        Positional indices into ``dataset.frame``.

    Notes
    -----
    Professional escape hatch. BuildML still enforces disjoint membership and
    fit-scope guards; it does not silently allow fit-on-full-data.
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
    """Return a copy of the frame rows belonging to ``partition``."""
    indices = plan.indices_for(partition)
    if not indices and partition == "validation":
        raise ValidationError("No validation partition exists on this split plan")
    return dataset._ensure_pandas().iloc[list(indices)].copy()


def assert_fit_partition(plan: SplitPlan | None, partition: PartitionName = "train") -> None:
    """Guard fit-capable operations against full-data / wrong-partition use.

    Parameters
    ----------
    plan:
        Current split plan. ``None`` means no split has been created.
    partition:
        Partition the caller intends to fit on.

    Raises
    ------
    LeakageError
        If there is no split, or the requested partition is not ``train``.
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

