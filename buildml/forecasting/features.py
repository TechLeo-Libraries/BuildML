"""Time-order helpers and lag/window feature construction for forecasting."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from buildml.core.errors import LeakageError, ValidationError
from buildml.core.types import ColumnRole
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, frame_for_partition

_FORBIDDEN_SPLIT_KINDS = frozenset({"random", "stratified", "group"})


def resolve_time_column(dataset: Dataset, time_column: str | None = None) -> str:
    """Resolve the sole time-role column or an explicit override.

    Forecasting APIs require exactly one chronological index column unless
    the caller passes an explicit ``time_column`` name.

    Parameters
    ----------
    dataset:
        Session dataset with assigned column roles.
    time_column:
        Optional explicit time column name overriding role resolution.

    Returns
    -------
    str
        Resolved time column name.

    Raises
    ------
    ValidationError
        When the override is missing from the dataset or role resolution finds
        zero or multiple time-role columns.
    """
    if time_column is not None:
        if time_column not in dataset.columns:
            raise ValidationError(f"time_column '{time_column}' is not in the dataset")
        return time_column
    times = dataset.role_columns(ColumnRole.TIME)
    if len(times) != 1:
        raise ValidationError(
            "Forecasting requires exactly one time-role column "
            f"(found {len(times)}: {times}). Assign roles with a 'time' column "
            "or pass time_column=..."
        )
    return times[0]


def resolve_target_column(dataset: Dataset, target_column: str | None = None) -> str:
    """Resolve the forecasting target column from roles or an override.

    Uses the dataset target role when ``target_column`` is ``None``; otherwise
    validates the explicit column name.

    Parameters
    ----------
    dataset:
        Session dataset with an assigned target role or explicit columns.
    target_column:
        Optional explicit target column name.

    Returns
    -------
    str
        Resolved target column name.

    Raises
    ------
    ValidationError
        When the override is missing from the dataset.
    """
    if target_column is not None:
        if target_column not in dataset.columns:
            raise ValidationError(f"target_column '{target_column}' is not in the dataset")
        return target_column
    return dataset.require_target()


def assert_temporal_split(split_plan: SplitPlan | None) -> None:
    """Refuse shuffled or entity-random splits for forecasting APIs.

    ``time`` splits are preferred. ``injected`` is allowed as an escape hatch
    when the caller owns chronological membership (order is still verified
    against the time column at fit time).

    Parameters
    ----------
    split_plan:
        Session split plan to validate for temporal suitability.

    Raises
    ------
    LeakageError
        When ``split_plan`` is ``None`` or uses random, stratified, or group
        split kinds that leak future rows into train features.
    ValidationError
        When the split kind is not ``time`` or ``injected``.
    """
    if split_plan is None:
        raise LeakageError(
            "Forecasting requires a temporal SplitPlan. Call session.time_split(...) "
            "after assigning a time role (random split is refused)."
        )
    kind = str(split_plan.kind)
    if kind in _FORBIDDEN_SPLIT_KINDS:
        raise LeakageError(
            f"Forecasting refuses split kind={kind!r}. Use time_split "
            "(or inject_split with chronologically ordered partitions). "
            "Shuffled random/stratified/group splits leak future rows into train "
            "features and metrics."
        )
    if kind not in {"time", "injected"}:
        raise ValidationError(
            f"Unsupported split kind for forecasting: {kind!r}. "
            "Expected 'time' or 'injected'."
        )


def ordered_frame(
    dataset: Dataset,
    split_plan: SplitPlan,
    partition: str,
    *,
    time_column: str,
) -> pd.DataFrame:
    """Return a partition frame sorted by the time column.

    Parses timestamps, refuses non-parseable values, and applies a stable
    mergesort so lag features respect chronological order.

    Parameters
    ----------
    dataset:
        Session dataset containing the requested partition.
    split_plan:
        Split plan defining partition membership.
    partition:
        Partition name such as ``train``, ``validation``, ``test``, or ``all``.
    time_column:
        Chronological index column used for sorting.

    Returns
    -------
    pandas.DataFrame
        Partition rows sorted ascending by ``time_column``.

    Raises
    ------
    ValidationError
        When timestamps in ``time_column`` cannot be parsed.
    """
    if partition == "all":
        frame = dataset._ensure_pandas().copy()
    else:
        frame = frame_for_partition(dataset, split_plan, partition).copy()  # type: ignore[arg-type]
    stamps = pd.to_datetime(frame[time_column], errors="coerce")
    if stamps.isna().any():
        raise ValidationError(
            f"Forecasting requires parseable timestamps in '{time_column}'; "
            f"found {int(stamps.isna().sum())} invalid value(s)"
        )
    frame = frame.assign(**{time_column: stamps}).sort_values(
        time_column, kind="mergesort"
    )
    return frame.reset_index(drop=True)


def assert_partition_time_order(
    dataset: Dataset,
    split_plan: SplitPlan,
    *,
    time_column: str,
) -> None:
    """Ensure train ends before validation or test in clock time.

    Detects temporal leakage when a holdout partition starts before the last
    train timestamp after chronological ordering.

    Parameters
    ----------
    dataset:
        Session dataset containing all partitions.
    split_plan:
        Temporal split plan to validate.
    time_column:
        Chronological index column used for ordering checks.

    Raises
    ------
    ValidationError
        When the train partition is empty after ordering.
    LeakageError
        When validation or test starts before train ends in clock time.
    """
    train = ordered_frame(dataset, split_plan, "train", time_column=time_column)
    if train.empty:
        raise ValidationError("Train partition is empty after time ordering")
    train_end = train[time_column].iloc[-1]
    for name in ("validation", "test"):
        indices = split_plan.indices_for(name)  # type: ignore[arg-type]
        if not indices:
            continue
        part = ordered_frame(dataset, split_plan, name, time_column=time_column)
        if part.empty:
            continue
        part_start = part[time_column].iloc[0]
        if part_start < train_end:
            raise LeakageError(
                f"Temporal leakage: {name} starts at {part_start} before "
                f"train ends at {train_end}. Forecasting requires "
                "chronological partition order (use time_split)."
            )


def normalize_lags(lags: list[int] | tuple[int, ...] | None) -> tuple[int, ...]:
    """Validate and normalize positive lag orders for lag-model features.

    Defaults to ``(1, 2, 3, 7)`` when ``lags`` is ``None`` and deduplicates
    caller-supplied values in ascending order.

    Parameters
    ----------
    lags:
        Positive integer lag orders, or ``None`` for the default set.

    Returns
    -------
    tuple[int, ...]
        Sorted unique lag orders, each at least ``1``.

    Raises
    ------
    ValidationError
        When ``lags`` is empty or contains values below ``1``.
    """
    if lags is None:
        lags = (1, 2, 3, 7)
    cleaned = tuple(sorted({int(v) for v in lags}))
    if not cleaned or any(v < 1 for v in cleaned):
        raise ValidationError("lags must be a non-empty collection of integers >= 1")
    return cleaned


def resolve_exog_columns(
    dataset: Dataset,
    exog_columns: list[str] | tuple[str, ...] | None,
    *,
    target_column: str,
    time_column: str,
) -> tuple[str, ...]:
    """Resolve optional exogenous feature columns for lag forecasting.

    Accepts only numeric columns distinct from the target and time columns.

    Parameters
    ----------
    dataset:
        Session dataset containing candidate exogenous columns.
    exog_columns:
        Optional list of exogenous column names.
    target_column:
        Target column name excluded from exog resolution.
    time_column:
        Time column name excluded from exog resolution.

    Returns
    -------
    tuple[str, ...]
        Resolved exogenous column names, or an empty tuple when none requested.

    Raises
    ------
    ValidationError
        When a column is the target or time column, is missing, or is not
        numeric.
    """
    if not exog_columns:
        return ()
    frame = dataset._ensure_pandas()
    resolved: list[str] = []
    for name in exog_columns:
        if name in {target_column, time_column}:
            raise ValidationError(
                f"exog column '{name}' cannot be the target or time column"
            )
        if name not in frame.columns:
            raise ValidationError(f"exog column '{name}' is not in the dataset")
        if not pd.api.types.is_numeric_dtype(frame[name]):
            raise ValidationError(
                f"exog column '{name}' must be numeric for classical lag forecasting"
            )
        resolved.append(name)
    return tuple(resolved)


def target_series(frame: pd.DataFrame, target_column: str) -> np.ndarray:
    """Extract a numeric target series from an ordered partition frame.

    Coerces values to float and refuses null or non-numeric targets before
    lag-matrix construction or evaluation.

    Parameters
    ----------
    frame:
        Partition frame containing the target column.
    target_column:
        Name of the forecasting target column.

    Returns
    -------
    numpy.ndarray
        One-dimensional float target array aligned with ``frame`` row order.

    Raises
    ------
    ValidationError
        When the column is missing or contains null/non-numeric values.
    """
    if target_column not in frame.columns:
        raise ValidationError(f"Target '{target_column}' missing from frame")
    series = pd.to_numeric(frame[target_column], errors="coerce")
    if series.isna().any():
        raise ValidationError(
            f"Target '{target_column}' has {int(series.isna().sum())} null/non-numeric "
            "value(s); impute or drop before forecasting"
        )
    return series.to_numpy(dtype=float)


def build_lag_matrix(
    y: np.ndarray,
    lags: tuple[int, ...],
    *,
    exog: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build supervised lag rows with no future target leakage.

    Row *i* predicts ``y[i]`` using ``y[i-lag]`` for each lag and optional
    contemporaneous exogenous values at *i*. Rows lacking full lag history are
    dropped.

    Parameters
    ----------
    y:
        One-dimensional target series in chronological order.
    lags:
        Positive lag orders used as autoregressive features.
    exog:
        Optional exogenous matrix with one row per ``y`` index.

    Returns
    -------
    X : numpy.ndarray
        Feature matrix with one row per usable lag row.
    y_out : numpy.ndarray
        Aligned target values for each feature row.
    start_index : int
        First absolute index in ``y`` represented by ``X[0]``.

    Raises
    ------
    ValidationError
        When ``y`` has fewer rows than ``max(lags)``.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    n = int(y.shape[0])
    max_lag = max(lags) if lags else 0
    if n <= max_lag:
        raise ValidationError(
            f"Need more than max(lags)={max_lag} rows to build lag features "
            f"(have n={n})"
        )
    start = max_lag
    rows: list[np.ndarray] = []
    targets: list[float] = []
    for i in range(start, n):
        feats = [float(y[i - lag]) for lag in lags]
        if exog is not None:
            feats.extend(float(v) for v in np.asarray(exog[i], dtype=float).reshape(-1))
        rows.append(np.asarray(feats, dtype=float))
        targets.append(float(y[i]))
    return np.vstack(rows), np.asarray(targets, dtype=float), start


def lag_feature_row(
    history: np.ndarray,
    lags: tuple[int, ...],
    *,
    exog_row: np.ndarray | None = None,
) -> np.ndarray:
    """Build one lag feature vector from the end of history.

    Uses only past target values at the requested lag offsets plus an optional
    contemporaneous exogenous row for one-step prediction.

    Parameters
    ----------
    history:
        Chronologically ordered target history ending at the forecast origin.
    lags:
        Positive lag orders defining autoregressive features.
    exog_row:
        Optional contemporaneous exogenous values for the scored step.

    Returns
    -------
    numpy.ndarray
        One-dimensional feature vector ready for estimator ``predict``.

    Raises
    ------
    ValidationError
        When ``history`` is shorter than ``max(lags)``.
    """
    history = np.asarray(history, dtype=float).reshape(-1)
    max_lag = max(lags)
    if history.shape[0] < max_lag:
        raise ValidationError(
            f"History length {history.shape[0]} is shorter than max lag {max_lag}"
        )
    feats = [float(history[-lag]) for lag in lags]
    if exog_row is not None:
        feats.extend(float(v) for v in np.asarray(exog_row, dtype=float).reshape(-1))
    return np.asarray(feats, dtype=float)


def stamp_strings(stamps: Any) -> tuple[str, ...]:
    """Serialize timestamps to ISO strings for results and bundles.

    Converts parseable values to ISO-8601 strings and uses empty strings for
    values that cannot be parsed as timestamps.

    Parameters
    ----------
    stamps:
        Iterable of timestamp-like values from a partition frame.

    Returns
    -------
    tuple[str, ...]
        ISO-formatted timestamp strings aligned with the input order.
    """
    values = pd.to_datetime(pd.Series(list(stamps)), errors="coerce")
    out: list[str] = []
    for value in values:
        if pd.isna(value):
            out.append("")
        else:
            out.append(pd.Timestamp(value).isoformat())
    return tuple(out)
