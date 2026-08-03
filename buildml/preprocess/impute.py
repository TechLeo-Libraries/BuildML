"""Fill in missing values with a constant learned from the training rows.

Most estimators cannot accept a missing value at all — scikit-learn will raise
rather than guess. So before modelling you have to decide what a gap means and
what to put there.

Simple imputation replaces every gap in a column with one number: the column's
median, its mean, its most frequent value, or a constant you choose. It is
crude by design. It does not model the relationship between columns, so it will
happily fill a missing income with the population median even when the row's
other fields make that implausible. What it buys you is predictability and
speed, and it is often enough — particularly when only a small fraction of
values are missing.

Two things are worth knowing before reaching for it. First, the fill value is
learned from training rows only; a median computed over the whole dataset leaks
the test distribution into training. Second, imputation destroys the
information that a value *was* missing, which is sometimes the most predictive
signal in the data — a blank income field on a loan application may say more
than any number would. If you suspect that, add an indicator column before
filling.

For gaps that depend on other columns, see
:mod:`buildml.preprocess.custom` for iterative and model-based alternatives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition, frame_for_partition
from buildml.ingest.detect import schema_from_dataframe
from buildml.preprocess.columns import resolve_transform_columns

Strategy = Literal["mean", "median", "most_frequent", "constant"]


@dataclass(slots=True)
class SimpleImputePlan:
    """The fill values learned from training rows, ready to replay anywhere.

    Keeping the learned numbers in an object rather than filling immediately is
    what makes inference correct: a single row arriving in production gets the
    same median the model was trained against, not the median of whatever batch
    it happened to arrive in.

    Attributes
    ----------
    columns:
        Columns this plan fills, in fit order.
    strategy:
        Which statistic was computed — ``'mean'``, ``'median'``,
        ``'most_frequent'``, or ``'constant'``.
    fill_value:
        The literal used when ``strategy`` is ``'constant'``; ``None``
        otherwise.
    statistics_:
        The actual number chosen per column. Worth reading before you trust the
        plan: a median of 0 on a column that should be positive usually means
        the column is mostly missing, and imputing it is papering over a data
        problem.
    """

    columns: tuple[str, ...]
    strategy: Strategy
    fill_value: Any | None
    statistics_: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return the plan as plain JSON-safe values.

        Used by model cards, checkpoints, and audit trails, where the fill
        values need to be readable outside Python. NumPy scalars are converted
        to built-in ``int`` and ``float``, and not-a-number becomes ``None``.

        Returns
        -------
        dict
            Keys ``columns``, ``strategy``, ``fill_value``, and
            ``statistics_``.
        """
        return {
            "columns": list(self.columns),
            "strategy": self.strategy,
            "fill_value": self.fill_value,
            "statistics_": dict(self.statistics_),
        }


def fit_simple_imputer(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    columns: list[str] | None = None,
    strategy: Strategy = "median",
    fill_value: Any | None = None,
) -> SimpleImputePlan:
    """Learn a fill value per column from the training rows.

    Computes the chosen statistic over training rows and returns it as a plan.
    Nothing is filled here — pass the plan to
    :func:`transform_simple_imputer` to apply it.

    Parameters
    ----------
    dataset:
        The full dataset. Only rows the split assigns to ``train`` are read.
    split_plan:
        The split that defines the training rows. Required, with no "fit on
        everything" fallback, because a median computed across the test rows
        quietly inflates every score you subsequently report.
    columns:
        Columns to fill. By default this covers numeric ``feature`` columns and
        skips ``ignore``, ``id``, ``target``, ``group``, ``time``, and
        ``weight`` — imputing a target would fabricate labels. Pass an explicit
        list to override, which is how you fill a categorical column with
        ``'most_frequent'`` or a constant.
    strategy:
        How the fill value is chosen.

        ``'median'`` (the default) takes the middle value, which ignores
        extremes — a handful of implausible salaries will not drag it.
        ``'mean'`` takes the average, which preserves the column total but
        moves with outliers. ``'most_frequent'`` takes the mode and is the only
        one of the three that works on text or categories.
        ``'constant'`` uses ``fill_value`` verbatim, which is the honest option
        when missing genuinely means something specific — zero purchases,
        "unknown", a sentinel your downstream code recognises.
    fill_value:
        The literal to insert when ``strategy`` is ``'constant'``. Ignored
        otherwise.

    Returns
    -------
    SimpleImputePlan
        The chosen fill value per column, ready to apply.

    Raises
    ------
    ~buildml.core.errors.LeakageError
        No split plan was supplied.
    ~buildml.core.errors.ValidationError
        No eligible columns were found and none were named explicitly.

    Notes
    -----
    Imputing distorts the distribution. Filling twenty percent of a column with
    its median creates an artificial spike at the median, which shrinks the
    apparent variance and can weaken a genuine relationship. When a column is
    mostly missing, dropping it usually beats filling it.

    A column that is entirely missing in the training rows has no statistic to
    learn, and its fill will be recorded as ``None``.

    Examples
    --------
    >>> plan = fit_simple_imputer(dataset, split_plan, strategy="median")  # doctest: +SKIP
    >>> plan.statistics_  # doctest: +SKIP
    {'age': 38.0, 'income': 52000.0}

    See Also
    --------
    transform_simple_imputer : Applies the plan produced here.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None

    train = frame_for_partition(dataset, split_plan, "train")
    cols = resolve_transform_columns(
        dataset,
        train,
        columns,
        kind="numeric",
        require_dtype=False,
        empty_message=(
            "No numeric feature columns available for simple imputation. "
            "Pass columns=... explicitly to include ignore/id roles."
        ),
    )
    imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
    imputer.fit(train[list(cols)])
    stats = {
        col: _jsonable_stat(value) for col, value in zip(cols, imputer.statistics_, strict=True)
    }
    return SimpleImputePlan(
        columns=tuple(cols),
        strategy=strategy,
        fill_value=fill_value,
        statistics_=stats,
    )


def transform_simple_imputer(dataset: Dataset, plan: SimpleImputePlan) -> Dataset:
    """Fill missing values everywhere using an already-learned plan.

    Runs over every row, training and test alike. That is the intended
    behaviour: the fill values came from training only, so applying them across
    all partitions makes the representation consistent without letting test
    rows influence what the representation is.

    Parameters
    ----------
    dataset:
        The dataset to fill. Every column the plan names must be present.
    plan:
        A plan from :func:`fit_simple_imputer`, or one restored from a saved
        pipeline.

    Returns
    -------
    ~buildml.data.dataset.Dataset
        A new dataset with the gaps filled and the schema refreshed. The input
        is unchanged.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        A column the plan expects is missing from the dataset, which usually
        means the plan and the data have drifted apart.

    Notes
    -----
    **Leakage:** this function trusts the plan. It has no way to tell whether
    the statistics inside were learned from training rows or from everything,
    so build plans through :func:`fit_simple_imputer` rather than by hand.

    Columns absent from the plan are passed through untouched, missing values
    and all — which will fail later at fit time if an estimator cannot accept
    them. That is deliberate: silently filling a column nobody asked about is
    worse than a clear error.

    See Also
    --------
    fit_simple_imputer : Produces the plan this consumes.
    """
    missing = [c for c in plan.columns if c not in dataset.columns]
    if missing:
        raise ValidationError(f"Impute plan columns missing from dataset: {missing}")

    frame = dataset._ensure_pandas().copy()
    for column in plan.columns:
        fill = plan.statistics_[column]
        frame[column] = frame[column].fillna(fill)

    return Dataset.from_transformed(
        dataset,
        frame,
        schema=schema_from_dataframe(frame),
    )


def _jsonable_stat(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value
