"""Scale features consistently, learning the scale from training data alone.

Neural networks are unusually sensitive to feature scale. A network fed one
column in the thousands and another in the hundredths has to find a single
learning rate that suits both, and no such rate exists: it will either crawl on
one or diverge on the other. Standardising every column to roughly zero mean
and unit variance removes the problem.

The split into ``fit`` and ``apply`` is what makes it safe. Statistics are
computed once, on the training partition, and then applied unchanged to
validation, test, and anything seen later. Recomputing them per partition would
scale each by its own distribution — which changes what the model sees at
inference, and does so without any error to notice.

See Also
--------
buildml.dl.dataset : Where these are applied in the right order.
buildml.preprocess : The classical scaling surface, with more methods.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError


def fit_standardize(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Learn each feature's centre and spread from the training rows.

    Computes per-column mean and standard deviation. These are stored on the
    contract and reused everywhere else, so this must be called on training
    data only.

    Parameters
    ----------
    x_train:
        The training feature matrix.

    Returns
    -------
    numpy.ndarray
        Per-column means.
    numpy.ndarray
        Per-column standard deviations, with near-zero values replaced by 1.0.

    Raises
    ------
    ValidationError
        If the input is not 2-D.

    Notes
    -----
    **A constant column would divide by zero, so its deviation becomes 1.0.**
    Subtracting the mean already sends such a column to all zeros, and the
    substitute divisor leaves it there rather than producing ``inf`` or ``NaN``
    that would propagate through every subsequent batch.

    ``nanmean`` and ``nanstd`` are used so a stray ``NaN`` does not poison the
    statistics — though :func:`frame_to_numeric_matrix` rejects ``NaN`` upstream
    anyway.

    See Also
    --------
    apply_standardize : Applying what this learned.
    """
    if x_train.ndim != 2:
        raise ValidationError("Standardize expects a 2-D feature matrix")
    mean = np.nanmean(x_train, axis=0)
    std = np.nanstd(x_train, axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return mean.astype(np.float64), std.astype(np.float64)


def apply_standardize(
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Rescale a matrix using statistics learned earlier.

    Subtracts the training mean and divides by the training deviation. The same
    constants are used for every partition and every later prediction, which is
    what keeps a deployed model seeing the inputs it was trained on.

    Parameters
    ----------
    x:
        The matrix to rescale.
    mean:
        Per-column means from :func:`fit_standardize`.
    std:
        Per-column deviations from :func:`fit_standardize`.

    Returns
    -------
    numpy.ndarray
        The rescaled matrix, same shape as the input.

    Raises
    ------
    ValidationError
        If the matrix width does not match the statistics — normally a sign
        that columns were added, dropped, or reordered since fitting.

    Notes
    -----
    **Holdout data will not have exactly zero mean after this, and should
    not.** The statistics describe the training distribution; the extent to
    which holdout differs from it is real information, and forcing it away would
    hide the distribution shift you would want to know about.
    """
    if x.shape[1] != mean.shape[0]:
        raise ValidationError(
            f"Feature width {x.shape[1]} does not match normalize width {mean.shape[0]}"
        )
    return (x - mean) / std


def frame_to_numeric_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Extract columns as a float matrix, refusing anything ambiguous.

    The gate between pandas and tensors. Tensors have no notion of a string
    category or a missing value, so both must be resolved before this point —
    and this function refuses rather than guessing.

    Parameters
    ----------
    frame:
        The source frame.
    columns:
        The columns to take, in the order the model will expect them.

    Returns
    -------
    numpy.ndarray
        A ``(n_rows, n_columns)`` float64 matrix.

    Raises
    ------
    ValidationError
        If a column is missing, if a column is non-numeric, or if any value is
        ``NaN``.

    Notes
    -----
    **Non-numeric columns are refused rather than encoded here.** How a category
    should be represented — one-hot, ordinal, target-encoded — is a modelling
    decision with real consequences, and it belongs in an explicit preprocessing
    step where the choice is recorded, not in a silent cast.

    **``NaN`` is refused for the same reason.** Filling with zero is a decision
    that would look like no decision at all: zero is a meaningful value in a
    standardised column, so an imputed row becomes indistinguishable from an
    average one.

    See Also
    --------
    buildml.preprocess : Encoding and imputation, fitted on train.
    """
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValidationError(f"Missing columns for Torch tensors: {missing}")
    subset = frame.loc[:, columns]
    for name in columns:
        if not pd.api.types.is_numeric_dtype(subset[name]):
            raise ValidationError(
                f"Column '{name}' is not numeric; encode or drop before make_torch_loaders"
            )
    values = subset.to_numpy(dtype=np.float64, copy=True)
    if np.isnan(values).any():
        raise ValidationError(
            "Feature matrix contains NaN; impute on train before make_torch_loaders"
        )
    return values
