"""Turn a split DataFrame into the tensors a Torch loop can consume.

Four things have to happen between a pandas frame and a training loop: decide
which columns are features, decide whether this is classification or
regression, materialise each partition as numeric arrays, and record the
transformations so inference can repeat them.

The ordering constraint is what makes this careful rather than mechanical.
Normalisation statistics and the class-label vocabulary are fitted on the
training partition **before** validation and test are touched, and then applied
to those partitions. Fitting on everything would let the holdout influence the
scaling, which inflates its score by an amount nothing in the output reveals.

The result is a :class:`~buildml.dl.types.FeatureContract` alongside the arrays.
The contract is what lets the trained model be used correctly six months later,
when the column order and the scaling constants are no longer in anyone's head.

See Also
--------
buildml.dl.transforms : The standardisation primitives.
buildml.dl.labels : The class-label vocabulary.
buildml.dl.loaders : Wrapping these datasets as DataLoaders.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.dl.extras import require_torch
from buildml.dl.labels import encode_class_targets, fit_class_labels
from buildml.dl.transforms import apply_standardize, fit_standardize, frame_to_numeric_matrix
from buildml.dl.types import FeatureContract, TaskSpec


def resolve_feature_target(
    dataset: Dataset,
) -> tuple[list[str], str]:
    """Work out which columns are inputs and which is the thing to predict.

    Prefers the roles you assigned. When no column is explicitly marked as a
    feature, falls back to everything that is not the target and not marked as
    an identifier, group, time, weight, or ignored column — the columns that
    are left over are, by elimination, the ones describing each row.

    Parameters
    ----------
    dataset:
        A Dataset with a target assigned.

    Returns
    -------
    list of str
        The feature columns, in dataset order. This order becomes the model's
        input order and must be reproduced at inference.
    str
        The target column.

    Raises
    ------
    ValidationError
        If no target is set, or if nothing usable remains after the exclusions
        — which normally means every column carries a non-feature role.

    Notes
    -----
    **Identifier columns are excluded for a reason.** A row ID often correlates
    with the target through how the data was collected, and a model given one
    will happily learn from it, producing an excellent holdout score and useless
    predictions on genuinely new rows.
    """
    target = dataset.require_target()
    feature_cols = dataset.role_columns(ColumnRole.FEATURE)
    if not feature_cols:
        skip = {
            *dataset.role_columns(ColumnRole.TARGET),
            *dataset.role_columns(ColumnRole.ID),
            *dataset.role_columns(ColumnRole.IGNORE),
            *dataset.role_columns(ColumnRole.GROUP),
            *dataset.role_columns(ColumnRole.TIME),
            *dataset.role_columns(ColumnRole.WEIGHT),
        }
        feature_cols = [c for c in dataset.columns if c not in skip and c != target]
    if not feature_cols:
        raise ValidationError("No feature columns available for Torch loaders")
    return feature_cols, target


def infer_task(y: pd.Series, task: TaskSpec) -> Literal["classification", "regression"]:
    """Guess whether the target is a category or a quantity.

    The distinction decides the loss function, the output layer width, and how
    predictions are read — so getting it wrong produces a model that trains
    without complaint and means nothing.

    Parameters
    ----------
    y:
        The target column.
    task:
        ``'auto'`` to infer, or an explicit choice which is returned unchanged.

    Returns
    -------
    str
        ``'classification'`` or ``'regression'``.

    Notes
    -----
    The rule: a numeric column is regression when it has more than
    ``max(10, 20% of rows)`` distinct values, and classification otherwise.
    Non-numeric columns are always classification. The proportional term is what
    keeps the rule sensible across scales — twelve distinct values among fifty
    rows is plausibly a quantity, but among fifty thousand rows it is plainly a
    set of categories.

    **This is a heuristic, and the failure it makes is quiet.** An integer-coded
    category with many levels reads as regression, and the resulting model
    treats category 7 as being closer to category 8 than to category 2. Pass
    ``task`` explicitly whenever you know.
    """
    if task != "auto":
        return task
    if pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) > max(10, int(0.2 * len(y))):
        return "regression"
    return "classification"


def partition_arrays(
    dataset: Dataset,
    split_plan: SplitPlan,
    partition: Literal["train", "validation", "test"],
    feature_columns: list[str],
    target_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Pull one partition out of the frame as plain numeric arrays.

    Selects the partition's rows by index, converts the feature columns to a
    float matrix, and returns the target alongside.

    Parameters
    ----------
    dataset:
        The full dataset.
    split_plan:
        Which rows belong to which partition.
    partition:
        Which one to extract.
    feature_columns:
        The columns to take, in order.
    target_column:
        The column to predict.

    Returns
    -------
    numpy.ndarray
        A ``(n_rows, n_features)`` float matrix.
    numpy.ndarray
        The targets.

    Raises
    ------
    ValidationError
        If the target column is non-numeric — class labels must be encoded to
        integers first — or if it contains ``NaN``. A missing label is not a
        label, and training on one teaches the model an arbitrary answer.

    Notes
    -----
    **An empty partition returns empty arrays rather than raising.** Not every
    workflow has a validation or test split, and a missing partition should
    simply produce no loader rather than stopping the run.
    """
    indices = split_plan.indices_for(partition)
    if not indices:
        return (
            np.empty((0, len(feature_columns)), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )
    # Index membership only — avoid frame_for_partition's empty-validation raise.
    frame = dataset._ensure_pandas().iloc[list(indices)].copy()
    if frame.empty:
        return (
            np.empty((0, len(feature_columns)), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )
    x = frame_to_numeric_matrix(frame, feature_columns)
    y_series = frame[target_column]
    if pd.api.types.is_numeric_dtype(y_series):
        y = y_series.to_numpy(dtype=np.float64, copy=True)
    else:
        # Classification with non-numeric labels handled by caller via label map.
        raise ValidationError(
            f"Target '{target_column}' must be numeric for the tabular Torch slice "
            "(encode class labels to integers first)"
        )
    if np.isnan(y).any():
        raise ValidationError("Target contains NaN; clean labels before make_torch_loaders")
    return x, y


def build_feature_contract(
    dataset: Dataset,
    split_plan: SplitPlan,
    *,
    task: TaskSpec = "auto",
    normalize: bool = True,
) -> tuple[FeatureContract, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Prepare every partition, learning the transformations from train alone.

    The central function of this module, and where the leakage boundary is
    enforced. Extracts the training partition, fits normalisation statistics and
    the class-label vocabulary on it, then applies both to validation and test.
    The transformations are recorded in the returned contract so inference can
    repeat them exactly.

    Parameters
    ----------
    dataset:
        The data, with roles and a target assigned.
    split_plan:
        Which rows belong to which partition.
    task:
        ``'auto'`` to infer from the training targets, or an explicit choice.
    normalize:
        Standardise features to zero mean and unit variance using training
        statistics. Usually worth leaving on — neural networks train poorly
        when features differ by orders of magnitude, because a single learning
        rate cannot suit all of them.

    Returns
    -------
    FeatureContract
        The data shape and the fitted transformations.
    dict
        Partition name to ``(features, targets)``. Missing partitions appear
        with empty arrays.

    Raises
    ------
    ValidationError
        If no feature columns are available, if the training partition is
        empty, or if a target is non-numeric or contains ``NaN``.

    Notes
    -----
    **Statistics come from train and are applied to holdout.** Fitting them
    across all partitions would let the test set influence the scaling — a
    small effect, but one that inflates the holdout score with nothing in the
    output to reveal it.

    **The class vocabulary is also train-only.** A class appearing exclusively
    in the test set is one the model was never taught, and it should surface as
    an error at encoding time rather than be quietly folded into the vocabulary.

    See Also
    --------
    buildml.dl.transforms.fit_standardize : Where the statistics come from.
    buildml.dl.labels.fit_class_labels : Where the vocabulary comes from.
    """
    feature_cols, target = resolve_feature_target(dataset)
    x_train, y_train = partition_arrays(dataset, split_plan, "train", feature_cols, target)
    if len(x_train) == 0:
        raise ValidationError("Train partition is empty; cannot build Torch loaders")
    resolved = infer_task(pd.Series(y_train), task)

    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    if normalize:
        mean, std = fit_standardize(x_train)

    class_labels: tuple[Any, ...] = ()
    if resolved == "classification":
        class_labels = fit_class_labels(y_train)

    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in ("train", "validation", "test"):
        if name == "train":
            x, y = x_train, y_train
        else:
            x, y = partition_arrays(dataset, split_plan, name, feature_cols, target)  # type: ignore[arg-type]
        if normalize and mean is not None and std is not None and len(x):
            x = apply_standardize(x, mean, std)
        if resolved == "classification" and len(y):
            y = encode_class_targets(y, class_labels).astype(np.float64, copy=False)
        arrays[name] = (x, y)

    contract = FeatureContract(
        feature_columns=tuple(feature_cols),
        target_column=target,
        task=resolved,
        class_labels=class_labels,
        normalize_mean=None if mean is None else tuple(float(v) for v in mean),
        normalize_std=None if std is None else tuple(float(v) for v in std),
    )
    return contract, arrays


def arrays_to_tensor_dataset(x: np.ndarray, y: np.ndarray, *, task: str) -> Any:
    """Convert prepared arrays into a Torch dataset.

    The last step before batching: NumPy arrays become tensors with the dtypes
    and shapes the loss functions expect.

    Parameters
    ----------
    x:
        The feature matrix.
    y:
        The targets.
    task:
        ``'classification'`` or ``'regression'``. Determines the target's dtype
        and shape.

    Returns
    -------
    torch.utils.data.TensorDataset
        Ready to hand to a ``DataLoader``.

    Raises
    ------
    MissingExtraError
        If PyTorch is not installed.

    Notes
    -----
    **The target dtype and shape follow the task, and both matter.**
    Classification targets become 1-D ``long`` because ``CrossEntropyLoss``
    requires class indices, not floats. Regression targets become ``float32``
    reshaped to ``(n, 1)`` to match a single-output layer — without the reshape,
    broadcasting between ``(n,)`` and ``(n, 1)`` silently produces an ``(n, n)``
    loss matrix and a meaningless gradient.

    Features are always ``float32``, which is what Torch layers default to.
    """
    torch = require_torch(feature="Tabular Torch datasets")
    x_t = torch.as_tensor(x, dtype=torch.float32)
    if task == "classification":
        y_t = torch.as_tensor(y, dtype=torch.long)
    else:
        y_t = torch.as_tensor(y, dtype=torch.float32).view(-1, 1)
    return torch.utils.data.TensorDataset(x_t, y_t)
