"""Feature / task-column helpers for meta-learning (train-only meta-train)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.core.validation import validate_column_names
from buildml.data.dataset import Dataset
from buildml.semisupervised.features import (
    matrix_from_frame as _matrix_from_frame,
)
from buildml.semisupervised.features import (
    resolve_semisupervised_columns,
)

__all__ = [
    "matrix_from_frame",
    "resolve_task_column",
    "resolve_target_column",
    "resolve_metalearning_columns",
    "encode_labels",
    "decode_labels",
    "task_ids_in_frame",
    "frame_for_task",
    "sample_support_query",
    "compute_prototypes",
    "nearest_prototype_predict",
]


def matrix_from_frame(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Build a float design matrix from selected columns.

    Delegates to semi-supervised matrix building with meta-learning error
    wording when null features are detected.

    Parameters
    ----------
    frame:
        Source DataFrame.
    columns:
        Feature column names to extract.

    Returns
    -------
    numpy.ndarray
        2-D float feature matrix.

    Raises
    ------
    ValidationError
        When columns are missing or contain null values.
    """
    try:
        return _matrix_from_frame(frame, columns)
    except ValidationError as exc:
        msg = str(exc).replace("Semi-supervised learning", "Meta-learning")
        raise ValidationError(msg) from exc


def resolve_task_column(
    dataset: Dataset,
    task_column: str | None,
) -> tuple[str, list[str]]:
    """Resolve the episodic task / group column from roles or an explicit name.

    Episodic meta-learning requires a stable task identifier separate from the
    classification target.

    Parameters
    ----------
    dataset:
        BuildML dataset with column roles.
    task_column:
        Optional explicit task column; when ``None``, requires exactly one
        ``role='group'`` column.

    Returns
    -------
    tuple[str, list[str]]
        Resolved task column name and disclosure notes.

    Raises
    ------
    ValidationError
        When no task column is defined or multiple group columns exist.
    """
    disclosures: list[str] = []
    if task_column is not None:
        name = validate_column_names([task_column], dataset.columns)[0]
        disclosures.append(
            f"Meta-learning task column taken from explicit task_column={name!r}."
        )
        return name, disclosures

    groups = list(dataset.role_columns(ColumnRole.GROUP))
    if len(groups) == 1:
        disclosures.append(
            f"Meta-learning task column taken from role='group' column: {groups[0]!r}."
        )
        return groups[0], disclosures
    if len(groups) > 1:
        raise ValidationError(
            "Multiple role='group' columns found "
            f"({groups}). Pass task_column= explicitly to select the episodic "
            "task identifier."
        )
    raise ValidationError(
        "Meta-learning needs a task/group column. Assign role='group' to the "
        "task identifier column, or pass task_column=."
    )


def resolve_target_column(dataset: Dataset) -> tuple[str, list[str]]:
    """Resolve exactly one classification target column.

    Multi-target joint fitting belongs on ``fit_multitask``; meta-learning
    carves episodic tasks via a separate task/group column.

    Parameters
    ----------
    dataset:
        BuildML dataset with column roles.

    Returns
    -------
    tuple[str, list[str]]
        Target column name and disclosure notes.

    Raises
    ------
    ValidationError
        When zero or multiple ``role='target'`` columns are present.
    """
    targets = list(dataset.role_columns(ColumnRole.TARGET))
    if len(targets) != 1:
        raise ValidationError(
            "Meta-learning requires exactly one role='target' column "
            f"(found {targets!r}). Multi-target joint fitting belongs on "
            "fit_multitask; meta-learning carves episodic tasks via a "
            "task/group column."
        )
    return targets[0], [
        f"Meta-learning target taken from role='target' column: {targets[0]!r}."
    ]


def resolve_metalearning_columns(
    dataset: Dataset,
    frame: pd.DataFrame,
    columns: list[str] | None,
    *,
    reduce_plan: Any | None = None,
    prefer_reduce_components: bool = True,
    target_column: str,
    task_column: str,
) -> tuple[list[str], bool, list[str]]:
    """Resolve numeric feature columns, excluding target and task columns.

    Reuses semi-supervised column resolution then removes the episodic task
    identifier from the feature set.

    Parameters
    ----------
    dataset:
        BuildML dataset with roles and optional reduce plan.
    frame:
        Training partition frame used for column validation.
    columns:
        Optional explicit feature list; ``None`` uses role-based resolution.
    reduce_plan:
        Optional dimensionality-reduction plan from preprocess.
    prefer_reduce_components:
        When ``True``, prefer reduced component columns when available.
    target_column:
        Label column to exclude from features.
    task_column:
        Episodic task column to exclude from features.

    Returns
    -------
    tuple[list[str], bool, list[str]]
        Feature column names, whether reduce components were used, disclosures.

    Raises
    ------
    ValidationError
        When no usable feature columns remain after exclusions.
    """
    cols, used_reduce, disclosures = resolve_semisupervised_columns(
        dataset,
        frame,
        columns,
        reduce_plan=reduce_plan,
        prefer_reduce_components=prefer_reduce_components,
        target_column=target_column,
    )
    exclude = {str(target_column), str(task_column)}
    filtered = [c for c in cols if c not in exclude]
    if not filtered:
        raise ValidationError(
            "No usable feature columns after excluding the target, task column, "
            "and protected roles."
        )
    out = [
        note.replace("semi-supervised", "meta-learning") for note in disclosures
    ]
    if task_column in cols:
        out.append(
            f"Excluded task column {task_column!r} from features "
            "(task identity is not a feature)."
        )
    return filtered, used_reduce, out


def encode_labels(
    series: pd.Series,
    *,
    label_encoder: Any | None = None,
) -> tuple[np.ndarray, Any, tuple[Any, ...]]:
    """Label-encode classification targets for episodic meta-learning.

    Refuses null labels and single-class sets; reuses a fitted encoder when
    provided for consistent class codes across support/query splits.

    Parameters
    ----------
    series:
        Target labels for one support or query set.
    label_encoder:
        Optional fitted :class:`sklearn.preprocessing.LabelEncoder`.

    Returns
    -------
    tuple[numpy.ndarray, Any, tuple[Any, ...]]
        Integer codes, encoder instance, and original class tuple.

    Raises
    ------
    ValidationError
        When labels contain nulls, unseen classes, or fewer than two classes.
    """
    from sklearn.preprocessing import LabelEncoder

    if series.isna().any():
        raise ValidationError(
            "Meta-learning target contains nulls. Drop or impute labels before fit."
        )
    values = series.astype(str)
    if label_encoder is None:
        enc = LabelEncoder()
        codes = enc.fit_transform(values)
    else:
        enc = label_encoder
        known = {str(c) for c in enc.classes_}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValidationError(
                f"Meta-learning target saw unseen class label(s): {unknown}."
            )
        codes = enc.transform(values)
    if len(enc.classes_) < 2:
        raise ValidationError(
            f"Meta-learning needs >= 2 classes (found {tuple(enc.classes_)!r})."
        )
    return np.asarray(codes), enc, tuple(enc.classes_)


def decode_labels(codes: np.ndarray, label_encoder: Any) -> tuple[Any, ...]:
    """Inverse-transform integer class codes to original label values.

    Used when reporting sampled episode classes and decoding prototype keys.

    Parameters
    ----------
    codes:
        Integer prediction or prototype class codes.
    label_encoder:
        Fitted label encoder from :func:`encode_labels`.

    Returns
    -------
    tuple[Any, ...]
        Original label values in code order.
    """
    decoded = label_encoder.inverse_transform(np.asarray(codes).astype(int))
    return tuple(_coerce_label(v) for v in decoded)


def task_ids_in_frame(frame: pd.DataFrame, task_column: str) -> list[Any]:
    """List stable unique task ids present in a frame (first-seen order).

    Preserves first-seen ordering for reproducible evaluation disclosures.

    Parameters
    ----------
    frame:
        Partition or task subset DataFrame.
    task_column:
        Episodic task identifier column.

    Returns
    -------
    list[Any]
        Unique task ids in first-seen order.

    Raises
    ------
    ValidationError
        When the task column is missing or contains nulls.
    """
    if task_column not in frame.columns:
        raise ValidationError(f"Task column {task_column!r} missing from frame.")
    series = frame[task_column]
    if series.isna().any():
        raise ValidationError(
            f"Task column {task_column!r} contains nulls; drop or fill before "
            "meta-learning."
        )
    # Preserve first-seen order for reproducibility disclosures.
    seen: list[Any] = []
    for value in series.tolist():
        if value not in seen:
            seen.append(value)
    return seen


def frame_for_task(frame: pd.DataFrame, task_column: str, task_id: Any) -> pd.DataFrame:
    """Return rows belonging to one episodic task.

    Returns a copy so callers can mutate support/query subsets safely.

    Parameters
    ----------
    frame:
        Source DataFrame.
    task_column:
        Episodic task identifier column.
    task_id:
        Task value to filter on.

    Returns
    -------
    pandas.DataFrame
        Copy of rows where ``task_column == task_id``.
    """
    return frame.loc[frame[task_column] == task_id].copy()


def sample_support_query(
    frame: pd.DataFrame,
    *,
    target_column: str,
    columns: Sequence[str],
    label_encoder: Any,
    k_shot: int,
    n_query: int,
    n_way: int | None,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Any]] | None:
    """Draw a balanced support/query split for one episodic task.

    Returns ``None`` when per-class row counts are insufficient for the
    requested ``k_shot``, ``n_query``, and ``n_way`` settings.

    Parameters
    ----------
    frame:
        Rows for a single task.
    target_column:
        Classification label column.
    columns:
        Feature columns (validated for downstream matrix build).
    label_encoder:
        Fitted label encoder for consistent class codes.
    k_shot:
        Support examples per class.
    n_query:
        Query examples per class (may be relaxed when rows are scarce).
    n_way:
        Optional cap on classes sampled per episode; ``None`` uses all present.
    rng:
        NumPy random generator for reproducible splits.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, list[Any]] or None
        Support frame, query frame, and sampled class labels; ``None`` if
        infeasible.
    """
    y_codes, _, classes = encode_labels(frame[target_column], label_encoder=label_encoder)
    work = frame.copy()
    work["__y_code__"] = y_codes
    present = sorted(int(c) for c in np.unique(y_codes))
    if n_way is not None:
        if len(present) < n_way:
            return None
        chosen = list(rng.choice(present, size=n_way, replace=False))
    else:
        chosen = present
        if len(chosen) < 2:
            return None

    support_parts: list[pd.DataFrame] = []
    query_parts: list[pd.DataFrame] = []
    for code in chosen:
        subset = work.loc[work["__y_code__"] == code]
        need = int(k_shot) + max(1, int(n_query) // max(len(chosen), 1))
        if len(subset) < need:
            # Relax query size if the class has at least k_shot + 1 rows.
            if len(subset) < int(k_shot) + 1:
                return None
            need = len(subset)
        idx = rng.choice(len(subset), size=need, replace=False)
        picked = subset.iloc[idx]
        support_parts.append(picked.iloc[: int(k_shot)])
        query_parts.append(picked.iloc[int(k_shot) :])

    support = pd.concat(support_parts, axis=0).drop(columns=["__y_code__"])
    query = pd.concat(query_parts, axis=0).drop(columns=["__y_code__"])
    if len(query) < 1:
        return None
    # Ensure feature columns exist / non-null via matrix build later.
    _ = list(columns)
    class_labels = decode_labels(np.asarray(chosen), label_encoder)
    return support, query, list(class_labels)


def compute_prototypes(
    x: np.ndarray,
    y_codes: np.ndarray,
) -> dict[int, np.ndarray]:
    """Compute mean embedding (raw features) per class code for prototypical prediction.

    Prototypes are class centroids used by nearest-prototype classifiers in
    sklearn and torch episodic paths.

    Parameters
    ----------
    x:
        Feature matrix ``(n_samples, n_features)``.
    y_codes:
        Integer class codes aligned with ``x`` rows.

    Returns
    -------
    dict[int, numpy.ndarray]
        Class code to prototype vector mapping.
    """
    protos: dict[int, np.ndarray] = {}
    for code in np.unique(y_codes):
        mask = y_codes == code
        if not np.any(mask):
            continue
        protos[int(code)] = np.mean(x[mask], axis=0)
    return protos


def nearest_prototype_predict(
    x: np.ndarray,
    prototypes: dict[int, np.ndarray],
) -> np.ndarray:
    """Assign each row to the nearest class prototype using squared Euclidean distance.

    Uses an efficient expanded distance formula for batch query assignment.

    Parameters
    ----------
    x:
        Query feature matrix ``(n_samples, n_features)``.
    prototypes:
        Class code to prototype vector mapping from :func:`compute_prototypes`.

    Returns
    -------
    numpy.ndarray
        Predicted integer class codes per row.

    Raises
    ------
    ValidationError
        When ``prototypes`` is empty.
    """
    if not prototypes:
        raise ValidationError("No prototypes available for prediction.")
    codes = sorted(prototypes)
    proto_mat = np.vstack([prototypes[c] for c in codes])
    # Squared euclidean distances (n, n_classes)
    d2 = (
        np.sum(x**2, axis=1, keepdims=True)
        + np.sum(proto_mat**2, axis=1)
        - 2.0 * x @ proto_mat.T
    )
    nearest = np.argmin(d2, axis=1)
    return np.asarray([codes[i] for i in nearest], dtype=int)


def _coerce_label(value: Any) -> Any:
    text = str(value)
    if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
        return int(text)
    try:
        return float(text) if "." in text else text
    except ValueError:
        return text
