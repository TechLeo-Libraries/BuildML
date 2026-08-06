"""Prepare features and targets, and score results, for case-based reasoning.

The supporting layer between raw frames and case memory: resolving which columns
are features, building the numeric matrix, encoding labels, fitting the scaling
statistics, and computing the holdout metrics.

Two rules run through all of it. Anything fitted is fitted on training rows and
applied everywhere else: standardisation, ranges, label encodings: because a
statistic recomputed on holdout data would let that data shape the notion of
similarity. And nulls are refused rather than imputed or dropped. There is no
defensible distance between a missing value and a present one, and silently
dropping rows changes what a metric's denominator means without saying so.

See Also
--------
buildml.cbr.fit.fit_cbr : The main consumer of these helpers.
buildml.cbr.cases.pairwise_distances : What the prepared matrices feed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.core.validation import validate_column_names
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, frame_for_partition
from buildml.semisupervised.features import (
    matrix_from_frame as _matrix_from_frame,
)
from buildml.semisupervised.features import (
    resolve_semisupervised_columns,
)

__all__ = [
    "matrix_from_frame",
    "resolve_cbr_columns",
    "resolve_categorical_columns",
    "encode_classification_targets",
    "decode_predictions",
    "regression_targets",
    "train_partition_frame",
    "classification_accuracy",
    "regression_metrics",
    "standardize_fit",
    "standardize_apply",
    "numeric_ranges",
]


def matrix_from_frame(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Build the float matrix distances are computed over, rejecting nulls.

    Shares its implementation with the semi-supervised path and re-labels the
    error text, so a CBR user reading a failure sees CBR terminology rather than
    a message about a module they did not call.

    Parameters
    ----------
    frame:
        The source rows.
    columns:
        Feature columns, in the order the case base expects.

    Returns
    -------
    numpy.ndarray
        A float matrix, shape ``(n_rows, n_columns)``.

    Raises
    ------
    ValidationError
        If a column is missing, non-numeric, or contains nulls.

    Notes
    -----
    **Nulls are refused rather than filled.** A missing value has no distance to
    anything, and substituting a mean would place the row at the centre of the
    data: where it would retrieve neighbours it has no relationship to. Impute
    deliberately, before fitting.
    """
    try:
        return _matrix_from_frame(frame, columns)
    except ValidationError as exc:
        msg = str(exc).replace("Semi-supervised learning", "Case-based reasoning")
        raise ValidationError(msg) from exc


def resolve_cbr_columns(
    dataset: Dataset,
    frame: pd.DataFrame,
    columns: list[str] | None,
    *,
    reduce_plan: Any | None = None,
    prefer_reduce_components: bool = True,
    target_column: str,
) -> tuple[list[str], bool, list[str]]:
    """Decide which numeric columns define similarity.

    Uses an explicit list when given, otherwise infers from dtype and column
    roles, excluding the target and any protected role. When a reduce plan is
    available and preferred, its components are used instead of the raw
    features.

    That last choice is worth understanding rather than accepting. Distance
    degrades badly in high dimensions: as columns multiply, every pair of
    points drifts toward the same distance and "nearest" stops meaning much.
    Reducing first restores a useful geometry, at the cost that neighbours are
    now neighbours under the projection, and the components are not columns
    anyone can interpret.

    Parameters
    ----------
    dataset:
        Supplies column roles.
    frame:
        The rows the columns must exist in.
    columns:
        Explicit feature columns, or ``None`` to infer.
    reduce_plan:
        A fitted reduction whose components may be used instead.
    prefer_reduce_components:
        Whether to prefer those components when available.
    target_column:
        Excluded from features.

    Returns
    -------
    tuple
        ``(columns, used_reduce, disclosures)``: the resolved features, whether
        reduced components were chosen, and plain-language notes on how.

    Notes
    -----
    **Every resolved column pulls on distance equally.** Inference selects
    columns that are usable, not columns that are relevant, so an irrelevant
    numeric column will still separate otherwise-similar cases. Passing an
    explicit list is usually the better choice.
    """
    cols, used_reduce, disclosures = resolve_semisupervised_columns(
        dataset,
        frame,
        columns,
        reduce_plan=reduce_plan,
        prefer_reduce_components=prefer_reduce_components,
        target_column=target_column,
    )
    out = [note.replace("semi-supervised", "case-based reasoning") for note in disclosures]
    return cols, used_reduce, out


def resolve_categorical_columns(
    dataset: Dataset,
    frame: pd.DataFrame,
    categorical_columns: list[str] | None,
    *,
    target_column: str,
    numeric_columns: list[str],
) -> tuple[list[str], list[str]]:
    """Validate the categorical columns the mixed metric will use.

    Categoricals are never inferred, only validated. That is deliberate:
    auto-detecting string columns would sweep in free-text notes, identifiers,
    and timestamps-as-strings, each of which would contribute a near-constant
    mismatch to every distance and quietly degrade retrieval. Naming them is a
    small cost for knowing what similarity is built from.

    Parameters
    ----------
    dataset:
        Supplies column roles.
    frame:
        The rows the columns must exist in, with no nulls.
    categorical_columns:
        The columns to use, or ``None`` / empty for none at all.
    target_column:
        Excluded if named.
    numeric_columns:
        The already-resolved numeric features, checked for overlap.

    Returns
    -------
    tuple
        ``(columns, disclosures)``: the validated columns and notes on the
        resolution.

    Raises
    ------
    ValidationError
        If a column is both numeric and categorical, is absent from the frame,
        contains nulls, or if every requested column was excluded as a
        protected role.

    Notes
    -----
    **Protected roles are dropped silently, but dropping all of them raises.**
    An ID or group column excluded from features is routine; a request where
    nothing survives means the caller misunderstood, and failing is clearer than
    returning an empty list.

    **High-cardinality columns are a poor fit for the mixed metric.** It only
    asks whether two values are equal, so a column with thousands of distinct
    values contributes a mismatch almost every time and acts as a constant
    offset rather than a signal.
    """
    disclosures: list[str] = []
    if not categorical_columns:
        return [], disclosures
    protected = {
        ColumnRole.TARGET,
        ColumnRole.ID,
        ColumnRole.GROUP,
        ColumnRole.TIME,
        ColumnRole.WEIGHT,
    }
    names = validate_column_names(categorical_columns, dataset.columns)
    out: list[str] = []
    for name in names:
        if name == target_column or dataset.roles.get(name) in protected:
            continue
        if name in numeric_columns:
            raise ValidationError(
                f"Column {name!r} cannot be both numeric and categorical for CBR."
            )
        if name not in frame.columns:
            raise ValidationError(f"Categorical column {name!r} missing from frame.")
        if frame[name].isna().any():
            raise ValidationError(
                f"Categorical column {name!r} has nulls; impute or drop before CBR."
            )
        out.append(name)
    if not out:
        raise ValidationError(
            "No usable categorical columns after excluding protected roles / target."
        )
    disclosures.append(
        f"Using {len(out)} explicit categorical column(s) for mixed-metric CBR."
    )
    return out, disclosures


def encode_classification_targets(
    y: pd.Series,
    *,
    classes: Sequence[Any] | None = None,
) -> tuple[np.ndarray, Any, tuple[Any, ...]]:
    """Turn class labels into integer codes, keeping the mapping for later.

    The encoder is returned alongside the codes because predictions have to be
    decoded back to the labels the caller recognises. A prediction of ``2`` is
    not an answer.

    Parameters
    ----------
    y:
        Training targets. Must contain no nulls.
    classes:
        A fixed class ordering. Supplying it keeps codes stable across fits,
        which matters when comparing runs or loading a saved plan.

    Returns
    -------
    tuple
        ``(codes, encoder, classes)``: the integer codes, the fitted encoder,
        and the class labels in code order.

    Raises
    ------
    ValidationError
        If any target is null.

    Notes
    -----
    **Labels are compared as strings**, so an integer ``1`` and the string
    ``"1"`` become the same class. This is usually what mixed-dtype label
    columns want, and it means the decoded value may differ in type from the
    original.

    **Null targets are refused rather than dropped.** A case with no known
    outcome has nothing to contribute to memory, and dropping it silently would
    change the case count without explanation.
    """
    from sklearn.preprocessing import LabelEncoder

    if y.isna().any():
        raise ValidationError(
            "CBR classification requires non-null train targets (case solutions)."
        )
    values = y.astype(str)
    encoder = LabelEncoder()
    if classes is not None:
        encoder.fit([str(c) for c in classes])
        codes = encoder.transform(values)
    else:
        codes = encoder.fit_transform(values)
    return np.asarray(codes), encoder, tuple(encoder.classes_)


def decode_predictions(pred_codes: np.ndarray, label_encoder: Any) -> list[Any]:
    """Turn integer class codes back into labels a caller recognises.

    Inverts the encoding, then attempts to restore the original type: a label
    that reads as an integer comes back as an integer, one that reads as a float
    comes back as a float, and everything else stays a string. The encoder works
    in strings, and returning ``"2"`` where the data held ``2`` would force every
    caller to convert.

    Parameters
    ----------
    pred_codes:
        Predicted class codes.
    label_encoder:
        The encoder fitted during training.

    Returns
    -------
    list
        Predictions in their recovered types.

    Notes
    -----
    **Type recovery is heuristic and can surprise.** A class label of ``"007"``
    comes back as the integer ``7``, and a version string like ``"1.10"`` becomes
    the float ``1.1``. Where exact label identity matters, compare as strings.
    """
    codes = np.asarray(pred_codes).astype(int)
    decoded = label_encoder.inverse_transform(codes)
    out: list[Any] = []
    for value in decoded:
        text = str(value)
        if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
            out.append(int(text))
        else:
            try:
                out.append(float(text) if "." in text else text)
            except ValueError:
                out.append(text)
    return out


def regression_targets(y: pd.Series) -> np.ndarray:
    """Extract regression targets as floats, refusing anything unusable.

    The solutions stored on cases for a regression task, checked to be numeric
    and complete before they become part of memory.

    Parameters
    ----------
    y:
        Training targets.

    Returns
    -------
    numpy.ndarray
        The targets as a float array.

    Raises
    ------
    ValidationError
        If any target is null, or the column is not numeric.

    Notes
    -----
    **A non-numeric target is refused rather than coerced.** Regression reuse
    averages neighbour solutions, and averaging strings that happen to look like
    numbers is a silent path to nonsense. Convert the column deliberately.
    """
    if y.isna().any():
        raise ValidationError(
            "CBR regression requires non-null numeric targets (case solutions)."
        )
    if not pd.api.types.is_numeric_dtype(y):
        raise ValidationError(
            "CBR regression requires a numeric target column."
        )
    return y.to_numpy(dtype=float)


def train_partition_frame(
    dataset: Dataset, split_plan: SplitPlan
) -> pd.DataFrame:
    """Return the training rows, which are the only rows allowed into memory.

    A named wrapper over partition selection, so every call site that builds
    case memory reads as train-only rather than as a partition lookup that
    happens to say ``'train'``.

    Parameters
    ----------
    dataset:
        The source data.
    split_plan:
        Partition membership.

    Returns
    -------
    pandas.DataFrame
        The training rows.

    See Also
    --------
    buildml.data.splits.frame_for_partition : The underlying selection.
    """
    return frame_for_partition(dataset, split_plan, "train")


def classification_accuracy(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    """Compute the fraction of predictions that match, comparing as strings.

    String comparison rather than equality, because predictions round-trip
    through the label encoder and may come back as a different type than the
    truth column holds. Comparing ``2`` against ``"2"`` as unequal would report
    a working classifier as broken.

    Parameters
    ----------
    y_true:
        True labels.
    y_pred:
        Predicted labels, same length.

    Returns
    -------
    float
        Accuracy in ``[0, 1]``, or NaN for an empty input.

    Notes
    -----
    **Accuracy is a poor summary under class imbalance.** With ninety-five per
    cent of one class, always predicting it scores 0.95 while being useless. Look
    at the per-class behaviour when classes are skewed.

    **An empty input returns NaN, not zero.** There is no accuracy to report,
    and zero would read as total failure.
    """
    if len(y_true) == 0:
        return float("nan")
    match = sum(str(a) == str(b) for a, b in zip(y_true, y_pred, strict=True))
    return float(match) / float(len(y_true))


def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Compute the three standard regression metrics, which disagree usefully.

    Each answers a different question, and reading all three together says more
    than any one:

    ``rmse``
        Root mean squared error, in the target's units. Squaring means a few
        large misses dominate, so this is the metric to watch when big errors
        are disproportionately costly.
    ``mae``
        Mean absolute error, also in the target's units. Every error counts in
        proportion to its size, so this describes the typical miss.
    ``r2``
        Fraction of variance explained, unitless. 1.0 is perfect, 0.0 matches
        predicting the mean, and negative is worse than that.

    An RMSE much larger than the MAE is the informative case: it means the
    errors are not uniformly distributed and a minority of predictions are badly
    wrong. Those are worth finding individually.

    Parameters
    ----------
    y_true:
        True values.
    y_pred:
        Predicted values, same length.

    Returns
    -------
    dict
        ``rmse``, ``mae``, and ``r2``.

    Notes
    -----
    **R² is relative to this partition's variance**, so it is not comparable
    across datasets or across partitions with different spread. RMSE and MAE
    are absolute and travel better.

    **A negative R² is a real result**, not an error: the predictions are worse
    than always guessing the mean.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
        "r2": float(r2_score(yt, yp)),
    }


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre and scale the training matrix, keeping the statistics for reuse.

    The most consequential preprocessing step in the whole method. Distance
    treats every feature's units as commensurable, which they are not: a salary
    column ranging over hundreds of thousands and an age column ranging over
    tens contribute in that ratio, so age becomes noise regardless of how
    predictive it is. Standardising puts them on equal footing.

    Parameters
    ----------
    x:
        The training numeric matrix.

    Returns
    -------
    tuple
        ``(standardized, mean, scale)``: the transformed matrix and the
        statistics, which must be kept and applied to every later query.

    Notes
    -----
    **Fitted on train and applied everywhere.** A query standardised by its own
    statistics would sit in a different coordinate system from memory, making
    the distances meaningless.

    **A constant column gets a scale of 1.0 rather than dividing by zero.** It
    then contributes nothing to distance, which is correct: a feature that
    never varies carries no information about similarity.

    **Standardising assumes roughly symmetric features.** A heavily skewed
    column keeps its skew and its outliers still dominate; transform it first if
    that matters.

    Examples
    --------
    A wide-ranging column and a narrow one end up comparable:

    >>> import numpy as np
    >>> x = np.array([[10.0, 1.0], [20.0, 2.0], [30.0, 3.0]])
    >>> z, mean, scale = standardize_fit(x)
    >>> mean.tolist()
    [20.0, 2.0]
    >>> z.round(4).tolist()
    [[-1.2247, -1.2247], [0.0, 0.0], [1.2247, 1.2247]]

    A constant column is neutralised rather than blowing up:

    >>> _, _, scale = standardize_fit(np.array([[5.0], [5.0]]))
    >>> scale.tolist()
    [1.0]

    See Also
    --------
    standardize_apply : Applying these statistics to queries.
    """
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return (x - mean) / scale, mean, scale


def standardize_apply(
    x: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """Apply the training statistics to new rows, without recomputing them.

    The counterpart to :func:`standardize_fit`, and the reason it returns its
    parameters. Queries must land in the same coordinate system as memory, which
    means using memory's mean and scale: not their own.

    Parameters
    ----------
    x:
        The matrix to transform.
    mean:
        Per-feature means from :func:`standardize_fit`.
    scale:
        Per-feature standard deviations from :func:`standardize_fit`, already
        floored away from zero.

    Returns
    -------
    numpy.ndarray
        The standardised matrix.

    Notes
    -----
    **Transformed values are not bounded to any range.** A query far outside the
    training distribution produces large standardised values, and therefore
    large distances: which is the correct signal that it has no close analogue.

    Examples
    --------
    A query at the training mean lands at the origin; one beyond the training
    range lands outside it:

    >>> import numpy as np
    >>> _, mean, scale = standardize_fit(np.array([[10.0], [20.0], [30.0]]))
    >>> standardize_apply(np.array([[20.0], [50.0]]), mean, scale).round(4).tolist()
    [[0.0], [3.6742]]

    See Also
    --------
    standardize_fit : Producing the statistics.
    """
    return (x - mean) / scale


def numeric_ranges(x: np.ndarray) -> np.ndarray:
    """Measure each feature's training spread, for the mixed metric to divide by.

    The mixed metric normalises numeric differences by range rather than by
    standard deviation, which is what puts them on the same ``[0, 1]`` footing
    as the categorical mismatch rate. Averaging a raw numeric difference against
    a zero-or-one mismatch would let whichever numeric column has the largest
    units decide every distance.

    Parameters
    ----------
    x:
        The training numeric matrix.

    Returns
    -------
    numpy.ndarray
        Per-column range, one entry per feature.

    Notes
    -----
    **A constant column gets a range of 1.0 rather than dividing by zero.** Its
    differences are all zero anyway, so it contributes nothing.

    **Ranges are set by the extremes, so one outlier defines the scale.** A
    single anomalous training value compresses every ordinary difference in that
    column toward zero, effectively muting the feature. Check for outliers
    before relying on the mixed metric.

    **A query outside the training range produces a normalised difference above
    one**, which the metric clips. Beyond the training extremes, further away
    stops registering as further.

    Examples
    --------
    >>> import numpy as np
    >>> numeric_ranges(np.array([[1.0, 100.0], [4.0, 700.0]])).tolist()
    [3.0, 600.0]

    One outlier sets the scale for the whole column:

    >>> numeric_ranges(np.array([[1.0], [2.0], [3.0], [9999.0]])).tolist()
    [9998.0]
    """
    lo = np.min(x, axis=0)
    hi = np.max(x, axis=0)
    ranges = hi - lo
    return np.where(ranges < 1e-12, 1.0, ranges)
