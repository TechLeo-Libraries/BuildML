"""Series extraction and temporal scope helpers for analysis."""

from __future__ import annotations

import numpy as np

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.forecasting.features import (
    assert_temporal_split,
    ordered_frame,
    resolve_target_column,
    resolve_time_column,
    stamp_strings,
    target_series,
)
from buildml.timeseries.types import AnalysisScope


def analysis_frame(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    scope: AnalysisScope = "train",
    time_column: str | None = None,
    target_column: str | None = None,
) -> tuple[np.ndarray, tuple[str, ...], str, str]:
    """Return ordered target series, timestamps, target_col, time_col."""
    assert_temporal_split(split_plan)
    if split_plan is None:
        raise ValidationError(
            "Time-series analysis requires a temporal SplitPlan. "
            "Call session.time_split(...) first."
        )
    time_col = resolve_time_column(dataset, time_column)
    target_col = resolve_target_column(dataset, target_column)
    partition = "train" if scope == "train" else "all"
    frame = ordered_frame(dataset, split_plan, partition, time_column=time_col)
    if frame.empty:
        raise ValidationError(f"Analysis scope={scope!r} produced an empty frame")
    y = target_series(frame, target_col)
    stamps = stamp_strings(frame[time_col].tolist())
    return y, stamps, target_col, time_col


def infer_seasonal_period(
    y: np.ndarray,
    *,
    seasonal_period: int | None = None,
    default: int = 7,
) -> int:
    """Resolve seasonal period with a sensible default."""
    if seasonal_period is not None:
        period = int(seasonal_period)
        if period < 2:
            raise ValidationError("seasonal_period must be >= 2")
        return period
    n = int(y.shape[0])
    if n >= 2 * default:
        return default
    if n >= 4:
        return max(2, n // 2)
    raise ValidationError(
        f"Need at least 4 points for decomposition (have n={n}); "
        "pass seasonal_period explicitly for short series."
    )
