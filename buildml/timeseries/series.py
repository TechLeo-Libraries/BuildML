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
    """Extract an ordered target series and timestamps for time-series analysis.

    Validates that the split is temporal (not random), resolves column names,
    and returns the numeric target vector plus string timestamps for the
    requested scope. Used internally by :func:`analyze_timeseries`.

    Parameters
    ----------
    dataset:
        Tabular frame holding time and target columns.
    split_plan:
        Temporal split from Session ``time_split``. ``None`` is refused.
    scope:
        ``train`` uses only the train partition; ``all`` uses every row in order.
    time_column:
        Sort key column. Defaults to the dataset's resolved time column.
    target_column:
        Numeric series to analyze. Defaults to the dataset target.

    Returns
    -------
    tuple[np.ndarray, tuple[str, ...], str, str]
        Target vector, timestamp strings, resolved target column name, and time
        column name.

    Raises
    ------
    ValidationError
        When the split is missing, not temporal, or the scoped frame is empty.
    """
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
    """Resolve the seasonal period for decomposition with sensible defaults.

    When the caller passes ``seasonal_period``, it is validated and returned.
    Otherwise picks ``default`` (7) when the series is long enough, or half the
    series length for very short windows.

    Parameters
    ----------
    y:
        Target vector whose length informs the default period.
    seasonal_period:
        Explicit cycle length. When ``None``, inferred from ``y`` and ``default``.
    default:
        Preferred period when ``n >= 2 * default`` (weekly seasonality on daily
        data, for example).

    Returns
    -------
    int
        Seasonal period >= 2 suitable for STL or moving-average decomposition.

    Raises
    ------
    ValidationError
        When an explicit period is < 2 or the series is too short to infer one.
    """
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
