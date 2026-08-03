"""Turn timestamps into features a model can actually use.

A raw timestamp is close to useless to most estimators. Treated as a number it
becomes "seconds since 1970", which grows monotonically and tells the model
almost nothing beyond "later": and worse, guarantees that every future row
falls outside the training range. Treated as a category, every timestamp is
unique and the column carries no signal at all.

What is actually predictive lives in the *parts*: the day of the week, because
weekends behave differently; the month, because demand is seasonal; the hour,
because mornings are not evenings; whether the date is the start or end of a
month, because billing cycles are real. Splitting one timestamp into those
components gives the model something it can learn a pattern from.

This module does that expansion. It is deliberately not fitted: the calendar
is the same for training and test rows, so there is no statistic to learn and
no way for this step to leak. For lag features and rolling windows, which
absolutely can leak, see :mod:`buildml.forecasting.features`.
"""

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
    """A record of which timestamps were expanded and what came out.

    There is nothing learned from the data here: unlike a scaler or an
    imputer, this plan holds no statistics. It exists so the expansion can be
    replayed identically at inference time, and so a model card can state
    exactly which columns the model expects.

    Attributes
    ----------
    columns:
        The source timestamp columns that were expanded.
    include_time:
        Whether hour, minute, and second were extracted alongside the calendar
        parts.
    created_columns:
        Every column the expansion added, named ``<source>_<part>``. This is
        the list to check against an incoming frame at inference time.
    drop_original:
        Whether the source timestamps were removed after expansion.
    """

    columns: tuple[str, ...]
    include_time: bool
    created_columns: tuple[str, ...]
    drop_original: bool

    def to_dict(self) -> dict[str, Any]:
        """Return the plan as plain JSON-safe values.

        Used by model cards and checkpoints, where the set of generated
        columns needs to be readable outside Python.

        Returns
        -------
        dict
            Keys ``columns``, ``include_time``, ``created_columns``, and
            ``drop_original``.
        """
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
    """Split each timestamp into the calendar parts a model can learn from.

    Every named column is parsed to a proper datetime and then expanded into
    year, month, day, day of week, day of year, quarter, and two flags for
    whether the date is the first or last day of its month. With
    ``include_time`` the clock parts are added too. The new columns are named
    ``<source>_<part>`` and given the ``feature`` role automatically, so they
    are picked up by later steps without extra wiring.

    Parameters
    ----------
    dataset:
        The dataset holding the timestamps.
    columns:
        Which columns to expand. Left as ``None``, this picks up anything
        already stored as a datetime plus anything you assigned the ``time``
        role. Name columns explicitly when your dates are still strings and
        pandas has not recognised them.
    include_time:
        Also extract hour, minute, and second. Leave this off for daily or
        coarser data, where those parts are all zero and only add noise and
        width. Turn it on for anything with intraday rhythm: web traffic,
        transactions, sensor readings.
    drop_original:
        Remove the source timestamp after expanding it. Useful because the raw
        column is rarely usable as a feature, but keep it when you still need
        it for a time-ordered split or for joining.

    Returns
    -------
    tuple of (~buildml.data.dataset.Dataset, DateFeaturePlan)
        The expanded dataset, and the record of what was created so the same
        expansion can be replayed at inference time.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        No datetime columns were found, or a named column is not in the
        dataset.

    Notes
    -----
    **Unparseable values become missing.** Parsing is lenient: a string that is
    not a date is turned into "not a time" rather than raising, and every part
    extracted from it is missing. That keeps one malformed row from stopping
    the pipeline, but check the resulting missing-value counts: a column that
    is suddenly half empty means the format was not what you assumed.

    **Cyclical parts are left as integers.** Month is 1 through 12, which
    tells a linear model that December and January are eleven apart rather than
    adjacent. Tree models handle this fine. For linear models and neural
    networks, consider a sine and cosine encoding of the part instead.

    **This step cannot leak**: the calendar does not depend on your data: so
    it is safe to run before splitting, unlike almost everything else in this
    package.

    Examples
    --------
    >>> data, plan = extract_date_features(  # doctest: +SKIP
    ...     dataset, ["order_date"], drop_original=True
    ... )
    >>> plan.created_columns[:3]  # doctest: +SKIP
    ('order_date_year', 'order_date_month', 'order_date_day')

    See Also
    --------
    buildml.session.Session.extract_dates : The session-level entry point.
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
