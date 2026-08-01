"""Datetime parsing and feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.core.validation import validate_column_names
from buildml.data.dataset import Dataset
from buildml.ingest.detect import schema_from_dataframe


@dataclass(slots=True)
class DateFeaturePlan:
    """Record of datetime feature engineering applied to a dataset."""

    columns: tuple[str, ...]
    include_time: bool
    created_columns: tuple[str, ...]
    drop_original: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "include_time": self.include_time,
            "created_columns": list(self.created_columns),
            "drop_original": self.drop_original,
        }


def extract_date_features(
    dataset: Dataset,
    columns: list[str] | tuple[str, ...] | None = None,
    *,
    include_time: bool = False,
    drop_original: bool = False,
) -> tuple[Dataset, DateFeaturePlan]:
    """Parse datetime columns and expand calendar/time parts.

    Parameters
    ----------
    dataset:
        Source dataset.
    columns:
        Datetime-like columns. Defaults to datetime dtypes and columns with
        role ``time``.
    include_time:
        Also extract hour/minute/second when available.
    drop_original:
        Drop source datetime columns after expansion.

    Notes
    -----
    Uses pandas ``.dt`` accessors (correctness fix vs BuildML 1.x).
    """
    base = dataset._ensure_pandas()
    if columns is None:
        inferred = list(base.select_dtypes(include=["datetime", "datetimetz"]).columns)
        inferred.extend(dataset.role_columns(ColumnRole.TIME))
        cols = validate_column_names(sorted(set(map(str, inferred))), dataset.columns)
    else:
        cols = validate_column_names(columns, dataset.columns)
    if not cols:
        raise ValidationError("No datetime columns available for date feature extraction")

    frame = base.copy()
    created: list[str] = []
    roles = dict(dataset.roles)

    for col in cols:
        parsed = pd.to_datetime(frame[col], errors="coerce", utc=False)
        frame[col] = parsed
        parts = {
            f"{col}_year": parsed.dt.year,
            f"{col}_month": parsed.dt.month,
            f"{col}_day": parsed.dt.day,
            f"{col}_dayofweek": parsed.dt.dayofweek,
            f"{col}_dayofyear": parsed.dt.dayofyear,
            f"{col}_quarter": parsed.dt.quarter,
            f"{col}_is_month_start": parsed.dt.is_month_start.astype("Int64"),
            f"{col}_is_month_end": parsed.dt.is_month_end.astype("Int64"),
        }
        if include_time:
            parts.update(
                {
                    f"{col}_hour": parsed.dt.hour,
                    f"{col}_minute": parsed.dt.minute,
                    f"{col}_second": parsed.dt.second,
                }
            )
        for name, series in parts.items():
            frame[name] = series
            created.append(name)
            roles.setdefault(name, ColumnRole.FEATURE)
        if drop_original:
            frame = frame.drop(columns=[col])
            roles.pop(col, None)
        else:
            roles[col] = ColumnRole.TIME

    plan = DateFeaturePlan(
        columns=tuple(cols),
        include_time=include_time,
        created_columns=tuple(created),
        drop_original=drop_original,
    )
    out = Dataset.from_transformed(
        dataset,
        frame,
        schema=schema_from_dataframe(frame),
        roles=roles,
    )
    return out, plan
