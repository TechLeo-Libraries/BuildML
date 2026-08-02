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
    """Resolve the sole time-role column (or explicit override)."""
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
    """Resolve the forecasting target column."""
    if target_column is not None:
        if target_column not in dataset.columns:
            raise ValidationError(f"target_column '{target_column}' is not in the dataset")
        return target_column
    return dataset.require_target()


def assert_temporal_split(split_plan: SplitPlan | None) -> None:
    """Refuse shuffled / entity-random splits for forecasting APIs.

    ``time`` splits are preferred. ``injected`` is allowed as an escape hatch
    when the caller owns chronological membership (order is still verified
    against the time column at fit time).
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
    """Return a partition frame sorted by the time column (stable)."""
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
    """Ensure train ends before validation/test in clock time."""
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
    """Validate and normalize positive lag orders."""
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
    """Resolve optional exogenous feature columns (numeric only)."""
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
    """Extract a numeric target series; refuse non-numeric / null targets."""
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
    """Build supervised lag rows ``(X, y)`` with no future leakage.

    Row *i* (absolute index) predicts ``y[i]`` using ``y[i-lag]`` for each lag
    and optional contemporaneous exogenous values at *i*. Rows that lack full
    lag history are dropped.

    Returns
    -------
    X, y_out, start_index
        Feature matrix, aligned targets, and the first absolute index used.
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
    """Build one lag feature vector from the end of ``history`` (past only)."""
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
    """Serialize timestamps for results / bundles."""
    values = pd.to_datetime(pd.Series(list(stamps)), errors="coerce")
    out: list[str] = []
    for value in values:
        if pd.isna(value):
            out.append("")
        else:
            out.append(pd.Timestamp(value).isoformat())
    return tuple(out)
