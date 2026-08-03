"""Put numeric features on a comparable scale, learning the scale from train only.

Many algorithms treat "distance" or "size" as meaningful. A k-nearest-neighbour
model measuring distance across an income column in the tens of thousands and an
age column in the tens will be deciding almost entirely on income, not because
income matters more but because its numbers are bigger. Regularised linear
models have the same problem: the penalty applies to coefficients, and a feature
measured in small units needs a large coefficient to have any effect, so it gets
penalised for its units.

Scaling removes the unit from the comparison. Tree-based models are immune —
they split on ordering, which scaling preserves — so this is optional for random
forests and gradient boosting, and close to mandatory for linear models, SVMs,
k-nearest neighbours, and neural networks.

The fit/transform split matters here. The mean and standard deviation are
learned from training rows only and then applied everywhere, because computing
them over the whole dataset lets the test rows influence how training rows are
represented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition, frame_for_partition
from buildml.ingest.detect import schema_from_dataframe
from buildml.preprocess.columns import resolve_transform_columns

ScaleMethod = Literal["standard", "minmax"]


@dataclass(slots=True)
class ScalePlan:
    """The scaling learned from the training rows, ready to replay anywhere.

    Holding the fitted scaler as an object rather than applying it immediately
    is what makes inference correct. The same shifts and divisors that were
    learned at training time get applied to tomorrow's single incoming row, so
    that row is represented the way the model expects. Recomputing statistics
    on new data would silently change the meaning of every feature.

    Attributes
    ----------
    columns:
        The columns this plan scales, in the order the fitted scaler expects
        them. Applying the plan to a frame missing any of them is an error
        rather than a silent skip.
    method:
        ``'standard'`` or ``'minmax'`` — which transform was fitted.
    scaler:
        The fitted scikit-learn scaler. Exposed for inspection; prefer
        :func:`transform_scaler` over calling it directly, since that path
        also runs the memory checks.
    """

    columns: tuple[str, ...]
    method: ScaleMethod
    scaler: Any

    def to_dict(self) -> dict[str, Any]:
        """Return the learned constants as plain JSON-safe values.

        Used for model cards, checkpoints, and audit trails — anywhere the
        numbers need to be read by something that is not Python. Which keys
        appear depends on the method: ``'standard'`` contributes ``mean_`` and
        ``scale_``, ``'minmax'`` contributes ``data_min_`` and ``data_max_``.

        Returns
        -------
        dict
            ``columns`` and ``method`` always, plus the fitted constants as
            lists of floats aligned to ``columns``.
        """
        payload: dict[str, Any] = {
            "columns": list(self.columns),
            "method": self.method,
        }
        if hasattr(self.scaler, "mean_"):
            payload["mean_"] = [float(x) for x in np.asarray(self.scaler.mean_)]
        if hasattr(self.scaler, "scale_"):
            payload["scale_"] = [float(x) for x in np.asarray(self.scaler.scale_)]
        if hasattr(self.scaler, "data_min_"):
            payload["data_min_"] = [float(x) for x in np.asarray(self.scaler.data_min_)]
        if hasattr(self.scaler, "data_max_"):
            payload["data_max_"] = [float(x) for x in np.asarray(self.scaler.data_max_)]
        return payload


def fit_scaler(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    columns: list[str] | None = None,
    method: ScaleMethod = "standard",
) -> ScalePlan:
    """Learn scaling constants from the training rows.

    Reads the training partition, measures each column's centre and spread, and
    returns those constants as a plan. Nothing is transformed yet — pass the
    plan to :func:`transform_scaler` to apply it.

    Requiring a split is deliberate. Fitting a scaler is the most common way
    people leak test data without realising: the standard deviation computed
    over the full dataset carries information about the test rows into the
    training representation, and the resulting holdout score is optimistic.

    Parameters
    ----------
    dataset:
        The full dataset. Only rows the split assigns to ``train`` are read.
    split_plan:
        The split that defines which rows are training rows. Required — there
        is no safe default for "fit on everything".
    columns:
        Columns to scale. By default this is the numeric ``feature`` columns,
        skipping ``ignore``, ``id``, ``target``, ``group``, ``time``, and
        ``weight`` roles, since scaling an identifier or a target is almost
        always a mistake. Pass an explicit list to override that judgement.
    method:
        ``'standard'`` subtracts the mean and divides by the standard
        deviation, giving each column mean zero and unit variance. This is the
        default and the right choice for roughly bell-shaped data and for
        regularised linear models.

        ``'minmax'`` rescales linearly so the training minimum becomes 0 and
        the maximum becomes 1. Useful when an algorithm needs bounded inputs,
        but it is sensitive to outliers: one extreme training value compresses
        everything else into a narrow band. Consider handling outliers first.

    Returns
    -------
    ScalePlan
        The fitted constants, ready to apply to any partition.

    Raises
    ------
    ~buildml.core.errors.LeakageError
        No split plan was supplied.
    ~buildml.core.errors.ValidationError
        ``method`` is not recognised, or no numeric feature columns were found
        and none were named explicitly.

    Notes
    -----
    Values outside the training range are not clipped. With ``'minmax'``, a test
    row beyond the training minimum or maximum lands outside ``[0, 1]``, which
    is correct behaviour — it reflects genuinely unseen data — but can surprise
    a downstream model that assumes bounded inputs.

    See Also
    --------
    transform_scaler : Apply the plan produced here.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None

    train = frame_for_partition(dataset, split_plan, "train")
    cols = resolve_transform_columns(
        dataset,
        train,
        columns,
        kind="numeric",
        empty_message=(
            "No numeric feature columns available for scaling. "
            "Pass columns=... explicitly to include ignore/id roles."
        ),
    )
    scaler: Any
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValidationError(f"Unsupported scale method '{method}'")

    scaler.fit(train[list(cols)])
    return ScalePlan(columns=tuple(cols), method=method, scaler=scaler)


def transform_scaler(
    dataset: Dataset,
    plan: ScalePlan,
    *,
    hard_limit_bytes: int | None = None,
) -> Dataset:
    """Apply a fitted scale plan to every row of a dataset.

    This runs over the whole frame, training and test rows alike, and that is
    correct: the constants came from training only, so applying them everywhere
    puts all partitions into the same representation without any test
    information having influenced what that representation is.

    Parameters
    ----------
    dataset:
        The dataset to transform. Every column named in the plan must be
        present.
    plan:
        A plan from :func:`fit_scaler`, or one restored from a saved pipeline.
    hard_limit_bytes:
        Ceiling on the estimated memory the scaled columns will occupy. Scaling
        forces float columns into a dense in-memory frame, which can be a large
        jump for a dataset that was lazily backed. Left as ``None``, the
        library-wide default applies; raise it deliberately when you know the
        machine can take it.

    Returns
    -------
    ~buildml.data.dataset.Dataset
        A new dataset with the named columns rescaled and their schema updated
        to float. The input is not modified.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        A column the plan expects is missing from the dataset — usually a sign
        the plan and the data have drifted out of step, for example because a
        column was dropped after fitting.
    ~buildml.core.errors.MemoryLimitError
        The dense result would exceed ``hard_limit_bytes``.

    Notes
    -----
    Column order matters to the underlying scaler, so the plan's own ordering is
    used rather than the dataset's. That means a reordered frame still scales
    correctly.

    See Also
    --------
    fit_scaler : Produces the plan this consumes.
    """
    from buildml.ingest.detect import MEMORY_HARD_LIMIT, check_materialization

    missing = [c for c in plan.columns if c not in dataset.columns]
    if missing:
        raise ValidationError(f"Scale plan columns missing from dataset: {missing}")

    base = dataset._ensure_pandas()
    scale_frame = base[list(plan.columns)]
    check_materialization(
        scale_frame,
        context="transform_scaler (design columns)",
        hard_limit_bytes=(
            MEMORY_HARD_LIMIT if hard_limit_bytes is None else hard_limit_bytes
        ),
    )
    frame = base.copy()
    scaled = plan.scaler.transform(frame[list(plan.columns)])
    scaled_df = pd.DataFrame(scaled, columns=list(plan.columns), index=frame.index)
    for column in plan.columns:
        frame[column] = scaled_df[column].astype(float)
    return Dataset.from_transformed(
        dataset,
        frame,
        schema=schema_from_dataframe(frame),
    )


