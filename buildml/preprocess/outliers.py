"""Find values far outside the normal range, and decide what to do about them.

An outlier is a value so far from the rest that it distorts what the model
learns. A salary recorded as 9,999,999 will drag a mean, inflate a standard
deviation, and pull a regression line toward itself hard enough to worsen
predictions for every ordinary row.

But "far from the rest" is a statistical judgement, not a verdict on
correctness. Some outliers are data errors: a decimal point in the wrong
place, a sentinel value nobody documented. Others are the most important rows
you have: in fraud detection and equipment failure, the extreme *is* the
signal, and removing it removes the thing you were trying to predict. Nothing
here can tell the two apart. That is your call, and it is why the default
action caps rather than deletes.

Two detection methods are available. The **interquartile range** approach marks
anything more than a multiple of the middle-50% spread beyond the quartiles; it
makes no assumption about the shape of the distribution and is not itself
distorted by the extremes it is looking for. The **z-score** approach marks
anything more than a number of standard deviations from the mean, which is
cleaner when data really is bell-shaped but has a circularity problem: the
outliers inflate the standard deviation that is supposed to detect them, so
severe ones can hide.

Fences are learned from training rows only. Applying them to test rows is the
point: the definition of "extreme" must not shift because the test set happened
to contain one enormous value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition, frame_for_partition
from buildml.explain.schemas import (
    Action,
    ActionPriority,
    Evidence,
    EvidenceKind,
    Finding,
    FindingSeverity,
    Recommendation,
)
from buildml.ingest.detect import schema_from_dataframe
from buildml.preprocess.columns import resolve_transform_columns
from buildml.preprocess.result import PreprocessResult

OutlierMethod = Literal["iqr", "zscore"]
OutlierAction = Literal["detect", "cap", "drop"]


@dataclass(slots=True)
class OutlierPlan:
    """The boundaries of "normal" learned from training rows, and what to do outside them.

    Attributes
    ----------
    columns:
        The numeric columns this plan covers.
    method:
        ``'iqr'`` or ``'zscore'``: how the fences were derived.
    action:
        ``'detect'``, ``'cap'``, or ``'drop'``: what happens to a flagged
        value.
    lower_:
        Lower fence per column. Anything below it is flagged.
    upper_:
        Upper fence per column. Anything above it is flagged. Read these
        against your domain knowledge before applying the plan: a fence that
        excludes physically plausible values is usually a sign the method or
        threshold is wrong for this column.
    n_flagged_train:
        How many training rows had at least one value outside its fences. A
        large number relative to the dataset means the fences are too tight, or
        the column is genuinely heavy-tailed and should be transformed rather
        than clipped.
    n_dropped:
        Rows actually removed. Zero unless the plan has been applied with
        ``action='drop'``.
    iqr_multiplier:
        The multiple of the interquartile range used, when the method is
        ``'iqr'``.
    zscore_threshold:
        The number of standard deviations used, when the method is
        ``'zscore'``.
    """

    columns: tuple[str, ...]
    method: OutlierMethod
    action: OutlierAction
    lower_: dict[str, float]
    upper_: dict[str, float]
    n_flagged_train: int
    n_dropped: int
    iqr_multiplier: float
    zscore_threshold: float

    def to_dict(self) -> dict[str, Any]:
        """Return the fences and settings as plain JSON-safe values.

        Used by model cards and checkpoints, and worth writing down: the
        fences are a documented statement about what range the model was built
        to handle.

        Returns
        -------
        dict
            Every attribute in plain-data form.
        """
        return {
            "columns": list(self.columns),
            "method": self.method,
            "action": self.action,
            "lower_": dict(self.lower_),
            "upper_": dict(self.upper_),
            "n_flagged_train": self.n_flagged_train,
            "n_dropped": self.n_dropped,
            "iqr_multiplier": self.iqr_multiplier,
            "zscore_threshold": self.zscore_threshold,
        }


def fit_outlier_plan(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    columns: list[str] | None = None,
    method: OutlierMethod = "iqr",
    action: OutlierAction = "cap",
    iqr_multiplier: float = 1.5,
    zscore_threshold: float = 3.0,
) -> OutlierPlan:
    """Work out where "normal" ends for each column, using the training rows.

    Computes a lower and upper fence per column and counts how many training
    rows fall outside them. Nothing is changed yet: pass the plan to
    :func:`apply_outlier_plan` to act on it, which is deliberately a separate
    step so you can inspect the fences first.

    Parameters
    ----------
    dataset:
        The full dataset. Only rows the split assigns to ``train`` are read.
    split_plan:
        The split defining the training rows. Required, because fences derived
        from all rows let the test set's extremes decide what counts as
        extreme.
    columns:
        Which numeric columns to examine. Defaults to numeric ``feature``
        columns. Naming columns explicitly is often wiser here: a column of
        counts where most values are zero will look full of outliers under any
        general rule.
    method:
        ``'iqr'`` (the default) places fences a multiple of the interquartile
        range beyond the first and third quartiles. It assumes nothing about
        the distribution's shape and the quartiles themselves are not moved by
        the extremes, so it is the safer general choice.

        ``'zscore'`` places fences a number of standard deviations either side
        of the mean. Tighter and more interpretable on genuinely bell-shaped
        data, but both the mean and the standard deviation are pulled by the
        very values it is meant to catch.
    action:
        What :func:`apply_outlier_plan` will do, recorded now so the plan is
        self-describing.

        ``'detect'`` changes nothing and only reports: start here.
        ``'cap'`` (the default) pulls flagged values back to the fence, which
        keeps the row and its other columns while removing the distortion.
        This is winsorising, and it is usually the right answer.
        ``'drop'`` removes the row entirely. Only reasonable when you are
        confident the row is erroneous, since it discards every other field
        too and shrinks your data.
    iqr_multiplier:
        How far past the quartiles the fences sit, in interquartile ranges.
        The conventional 1.5 flags roughly the outer 0.7% of a normal
        distribution; 3.0 is a common stricter setting that catches only
        extreme cases. Ignored unless the method is ``'iqr'``.
    zscore_threshold:
        How many standard deviations from the mean the fences sit. 3.0 flags
        about 0.3% of a normal distribution. Ignored unless the method is
        ``'zscore'``.

    Returns
    -------
    OutlierPlan
        The fences, the recorded action, and the training flag count.

    Raises
    ------
    ~buildml.core.errors.LeakageError
        No split plan was supplied.
    ~buildml.core.errors.ValidationError
        ``method`` or ``action`` is unrecognised, a threshold is not positive,
        no numeric columns resolved, or a column has no finite training values.

    Notes
    -----
    **Check the count before acting.** If ``n_flagged_train`` is a large
    fraction of the training rows, the fences are not describing outliers, they
    are describing the distribution. A heavily skewed column is usually better
    served by a log transform or by :mod:`buildml.preprocess.binning` than by
    clipping.

    **A constant column produces degenerate fences** where the lower and upper
    bound coincide, under the z-score method. Nothing will be flagged, which is
    correct, but such a column carries no information and is better dropped.

    Examples
    --------
    >>> plan = fit_outlier_plan(  # doctest: +SKIP
    ...     dataset, split_plan, method="iqr", action="detect"
    ... )
    >>> plan.n_flagged_train  # doctest: +SKIP
    17

    See Also
    --------
    apply_outlier_plan : Carries out the recorded action.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None
    if method not in {"iqr", "zscore"}:
        raise ValidationError(f"Unsupported outlier method '{method}'")
    if action not in {"detect", "cap", "drop"}:
        raise ValidationError(f"Unsupported outlier action '{action}'")
    if iqr_multiplier <= 0:
        raise ValidationError("iqr_multiplier must be positive")
    if zscore_threshold <= 0:
        raise ValidationError("zscore_threshold must be positive")

    train = frame_for_partition(dataset, split_plan, "train")
    cols = resolve_transform_columns(
        dataset,
        train,
        columns,
        kind="numeric",
        empty_message=(
            "No numeric feature columns available for outlier handling. "
            "Pass columns=... explicitly to include ignore/id roles."
        ),
    )
    lower: dict[str, float] = {}
    upper: dict[str, float] = {}
    for column in cols:
        series = pd.to_numeric(train[column], errors="coerce").dropna()
        if series.empty:
            raise ValidationError(
                f"Column '{column}' has no finite train values for outlier fences"
            )
        if method == "iqr":
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            lower[column] = q1 - iqr_multiplier * iqr
            upper[column] = q3 + iqr_multiplier * iqr
        else:
            mean = float(series.mean())
            std = float(series.std(ddof=0))
            if std == 0.0:
                lower[column] = mean
                upper[column] = mean
            else:
                lower[column] = mean - zscore_threshold * std
                upper[column] = mean + zscore_threshold * std

    mask = _flag_mask(train, cols, lower, upper)
    return OutlierPlan(
        columns=tuple(cols),
        method=method,
        action=action,
        lower_=lower,
        upper_=upper,
        n_flagged_train=int(mask.sum()),
        n_dropped=0,
        iqr_multiplier=float(iqr_multiplier),
        zscore_threshold=float(zscore_threshold),
    )


def apply_outlier_plan(
    dataset: Dataset,
    split_plan: SplitPlan,
    plan: OutlierPlan,
) -> tuple[Dataset, SplitPlan, OutlierPlan, PreprocessResult]:
    """Carry out the plan's action against every row.

    What happens depends on ``plan.action``. Detecting changes nothing and just
    reports. Capping clips flagged values to their fence. Dropping removes the
    flagged rows: and because that renumbers everything, the split plan has to
    be rebuilt so its partitions still point at the right rows, which is why a
    new split plan comes back rather than the one you passed in.

    Parameters
    ----------
    dataset:
        The dataset to act on. Every column the plan names must be present.
    split_plan:
        The current split. Returned unchanged for the detect and cap actions;
        rebuilt against the surviving rows when dropping.
    plan:
        A plan from :func:`fit_outlier_plan`.

    Returns
    -------
    tuple
        ``(dataset, split_plan, outlier_plan, result)`` :
        the dataset after the action; the split plan to use from now on; an
        updated copy of the plan with ``n_dropped`` filled in; and a narrated
        record of what was flagged and what was done.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        A column the plan expects is missing, or dropping would empty the
        training or test partition entirely. The latter is a hard stop rather
        than a warning, since it means the fences are catastrophically tight.

    Notes
    -----
    **Use the returned split plan.** After a drop, the one you passed in refers
    to row positions that no longer exist. Continuing with the old plan is a
    silent correctness bug, so replace your reference with the returned one :
    the session-level API does this for you.

    **Capping creates spikes.** Every clipped value lands exactly on the fence,
    producing a pile-up at that number. Harmless for most models, but it makes
    the column's distribution look artificial in later diagnostics.

    **Dropping affects test rows too.** The fences are applied to every
    partition, so a genuinely extreme test row disappears from your evaluation.
    That flatters the reported score relative to production, where no such
    filter exists.

    See Also
    --------
    fit_outlier_plan : Produces the plan this consumes.
    """
    missing = [c for c in plan.columns if c not in dataset.columns]
    if missing:
        raise ValidationError(f"Outlier plan columns missing from dataset: {missing}")

    if plan.action == "detect":
        result = _build_result(plan, mutated=False)
        return dataset, split_plan, plan, result

    frame = dataset._ensure_pandas().copy()
    if plan.action == "cap":
        for column in plan.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            capped = values.clip(lower=plan.lower_[column], upper=plan.upper_[column])
            frame[column] = capped
        new_dataset = Dataset.from_transformed(
            dataset,
            frame,
            schema=schema_from_dataframe(frame),
        )
        updated = OutlierPlan(
            columns=plan.columns,
            method=plan.method,
            action=plan.action,
            lower_=dict(plan.lower_),
            upper_=dict(plan.upper_),
            n_flagged_train=plan.n_flagged_train,
            n_dropped=0,
            iqr_multiplier=plan.iqr_multiplier,
            zscore_threshold=plan.zscore_threshold,
        )
        return new_dataset, split_plan, updated, _build_result(updated, mutated=True)

    # drop: remove flagged rows using train-learned fences; rebuild partitions.
    keep_mask = ~_flag_mask(frame, list(plan.columns), plan.lower_, plan.upper_)
    kept_positions = np.flatnonzero(keep_mask.to_numpy())
    old_to_new = {int(old): int(new) for new, old in enumerate(kept_positions)}

    def _remap(indices: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(old_to_new[i] for i in indices if i in old_to_new)

    new_split = SplitPlan(
        kind=f"outlier_drop_{split_plan.kind}",
        test_size=split_plan.test_size,
        validation_size=split_plan.validation_size,
        random_state=split_plan.random_state,
        stratify_column=split_plan.stratify_column,
        train_indices=_remap(split_plan.train_indices),
        validation_indices=_remap(split_plan.validation_indices),
        test_indices=_remap(split_plan.test_indices),
    )
    if not new_split.train_indices or not new_split.test_indices:
        raise ValidationError(
            "Outlier drop removed an entire train or test partition. "
            "Widen fences, switch to action='cap', or review columns."
        )
    new_split.assert_disjoint()

    new_frame = frame.iloc[list(kept_positions)].reset_index(drop=True)
    new_dataset = Dataset.from_transformed(
        dataset,
        new_frame,
        schema=schema_from_dataframe(new_frame),
    )
    n_dropped = int(len(frame) - len(new_frame))
    updated = OutlierPlan(
        columns=plan.columns,
        method=plan.method,
        action=plan.action,
        lower_=dict(plan.lower_),
        upper_=dict(plan.upper_),
        n_flagged_train=plan.n_flagged_train,
        n_dropped=n_dropped,
        iqr_multiplier=plan.iqr_multiplier,
        zscore_threshold=plan.zscore_threshold,
    )
    return new_dataset, new_split, updated, _build_result(updated, mutated=True)


def _flag_mask(
    frame: pd.DataFrame,
    columns: list[str],
    lower: dict[str, float],
    upper: dict[str, float],
) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        mask = mask | (values < lower[column]) | (values > upper[column])
    return mask


def _build_result(plan: OutlierPlan, *, mutated: bool) -> PreprocessResult:
    evidence = [
        Evidence(
            key="outlier.train_flagged",
            kind=EvidenceKind.METRIC,
            summary="Train rows outside train-fitted fences.",
            value={"n_flagged_train": plan.n_flagged_train, "method": plan.method},
            source="train.outlier_fences",
            limitations=(
                "Fence rules are heuristic screens, not proof of error or contamination.",
            ),
        )
    ]
    if plan.action == "drop":
        evidence.append(
            Evidence(
                key="outlier.dropped",
                kind=EvidenceKind.METRIC,
                summary="Rows removed after applying train-fitted fences.",
                value={"n_dropped": plan.n_dropped},
                source="dataset.outlier_drop",
                limitations=("Dropped holdout rows change evaluation support.",),
            )
        )
    severity = FindingSeverity.MEDIUM if plan.n_flagged_train > 0 else FindingSeverity.INFO
    findings = [
        Finding(
            key="outlier.screen",
            title="Train-fitted outlier screen",
            detail=(
                f"Method '{plan.method}' flagged {plan.n_flagged_train} train row(s); "
                f"action='{plan.action}'."
            ),
            severity=severity,
            evidence=tuple(evidence),
            affected_columns=plan.columns,
        )
    ]
    recommendations: list[Recommendation] = []
    if plan.action == "detect" and plan.n_flagged_train > 0:
        recommendations.append(
            Recommendation(
                key="outlier.consider-cap",
                title="Consider capping instead of silent deletion",
                rationale=(
                    "Capping preserves row membership while limiting extreme magnitudes "
                    "using the same train-fitted fences."
                ),
                priority=ActionPriority.NEXT,
                action=Action(
                    key="outlier.consider-cap-action",
                    label="Session.handle_outliers(action='cap')",
                    operation="handle_outliers",
                    parameters={"action": "cap", "method": plan.method},
                ),
                based_on=("outlier.screen",),
                caveats=(
                    "Capping assumes extremes are measurement noise rather than rare valid events.",
                ),
            )
        )
    interpretation = [
        (f"Fences were learned on train with method '{plan.method}' and action '{plan.action}'."),
        (
            "Dataset values were left unchanged."
            if not mutated
            else (
                f"Applied '{plan.action}' using frozen train fences"
                + (f"; dropped {plan.n_dropped} row(s)." if plan.n_dropped else ".")
            )
        ),
    ]
    limitations = [
        "IQR and z-score fences assume roughly unimodal numeric features.",
        "Flagged points may be valid rare events; domain review remains required.",
        "Fences must not be re-fit on validation or test rows.",
    ]
    methods = [
        f"Train-only {plan.method} fences; action={plan.action}.",
        (
            f"IQR multiplier={plan.iqr_multiplier}."
            if plan.method == "iqr"
            else f"Z-score threshold={plan.zscore_threshold}."
        ),
    ]
    return PreprocessResult(
        operation="handle_outliers",
        plan=plan.to_dict(),
        evidence=evidence,
        findings=findings,
        interpretation=interpretation,
        limitations=limitations,
        recommendations=recommendations,
        methods=methods,
    )
