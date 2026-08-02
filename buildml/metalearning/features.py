"""Feature / task-column helpers for meta-learning (train-only meta-train)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.core.validation import validate_column_names
from buildml.data.dataset import Dataset
from buildml.semisupervised.features import (
    matrix_from_frame as _matrix_from_frame,
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
    """Build a float design matrix; refuse null features (meta-learning wording)."""
    try:
        return _matrix_from_frame(frame, columns)
    except ValidationError as exc:
        msg = str(exc).replace("Semi-supervised learning", "Meta-learning")
        raise ValidationError(msg) from exc


def resolve_task_column(
    dataset: Dataset,
    task_column: str | None,
) -> tuple[str, list[str]]:
    """Resolve the episodic task / group column from roles or an explicit name."""
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
    """Resolve exactly one classification target (classical require_target style)."""
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
    """Resolve numeric feature columns, excluding target and task columns."""
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
    """LabelEncode classification targets; refuse nulls and single-class sets."""
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
    """Inverse-transform integer codes to original labels."""
    decoded = label_encoder.inverse_transform(np.asarray(codes).astype(int))
    return tuple(_coerce_label(v) for v in decoded)


def task_ids_in_frame(frame: pd.DataFrame, task_column: str) -> list[Any]:
    """Stable unique task ids present in a frame."""
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
    """Rows belonging to one episodic task."""
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
    """Draw a balanced support/query split for one task; None if infeasible."""
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
    """Mean embedding (raw features) per class code — tabular prototypical."""
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
    """Assign each row to the nearest class prototype (euclidean)."""
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
