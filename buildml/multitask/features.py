"""Feature / multi-target helpers for multi-task learning (train-only fit)."""

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
    "resolve_target_columns",
    "resolve_multitask_columns",
    "infer_task_type",
    "encode_multitask_y",
    "decode_multitask_predictions",
]


def matrix_from_frame(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Build a float design matrix; refuse null features (multi-task wording)."""
    try:
        return _matrix_from_frame(frame, columns)
    except ValidationError as exc:
        msg = str(exc).replace("Semi-supervised learning", "Multi-task learning")
        raise ValidationError(msg) from exc


def resolve_target_columns(
    dataset: Dataset,
    targets: Sequence[str] | None,
) -> tuple[list[str], list[str]]:
    """Resolve ``>= 2`` target columns from roles or an explicit list."""
    disclosures: list[str] = []
    if targets is not None:
        names = validate_column_names(list(targets), dataset.columns)
        disclosures.append(
            f"Multi-task targets taken from explicit targets= argument: {names}."
        )
    else:
        names = list(dataset.role_columns(ColumnRole.TARGET))
        disclosures.append(
            f"Multi-task targets taken from role='target' columns: {names}."
        )
    if len(names) < 2:
        raise ValidationError(
            "Multi-task fit needs at least 2 target columns "
            f"(found {names!r}). Assign multiple role='target' columns or pass "
            "targets=[...]. Classical Session.fit still requires exactly one "
            "target via require_target()."
        )
    disclosures.append(
        "Classical Session.fit / require_target still expect a single target; "
        "this multi-task path is distinct."
    )
    return names, disclosures


def resolve_multitask_columns(
    dataset: Dataset,
    frame: pd.DataFrame,
    columns: list[str] | None,
    *,
    reduce_plan: Any | None = None,
    prefer_reduce_components: bool = True,
    target_columns: Sequence[str],
) -> tuple[list[str], bool, list[str]]:
    """Resolve numeric feature columns, excluding all multi-task targets."""
    # Reuse semi-supervised resolver with a sentinel primary target, then
    # exclude every remaining target column explicitly.
    primary = str(target_columns[0])
    cols, used_reduce, disclosures = resolve_semisupervised_columns(
        dataset,
        frame,
        columns,
        reduce_plan=reduce_plan,
        prefer_reduce_components=prefer_reduce_components,
        target_column=primary,
    )
    exclude = {str(c) for c in target_columns}
    filtered = [c for c in cols if c not in exclude]
    if not filtered:
        raise ValidationError(
            "No usable feature columns after excluding multi-task targets and "
            "protected roles."
        )
    out = [
        note.replace("semi-supervised", "multi-task") for note in disclosures
    ]
    dropped = [c for c in cols if c in exclude and c != primary]
    if dropped:
        out.append(
            f"Excluded additional target columns from features: {dropped}."
        )
    return filtered, used_reduce, out


def infer_task_type(
    frame: pd.DataFrame,
    target_columns: Sequence[str],
    task: str,
) -> tuple[str, list[str]]:
    """Resolve classification vs regression; refuse mixed-type targets.

    ``auto`` classifies a column as regression when it is numeric with many
    distinct values; otherwise classification. All targets must agree.
    """
    disclosures: list[str] = []
    if task in {"classification", "regression"}:
        _assert_targets_compatible(frame, target_columns, task)
        disclosures.append(f"Multi-task task taken from explicit task={task!r}.")
        return task, disclosures

    if task != "auto":
        raise ValidationError(
            f"Unknown multi-task task={task!r}. "
            "Supported: 'classification', 'regression', 'auto'."
        )

    kinds: list[str] = []
    for col in target_columns:
        series = frame[col]
        if series.isna().any():
            raise ValidationError(
                f"Multi-task target {col!r} contains nulls on the train "
                "partition. Impute or drop nulls before fit_multitask."
            )
        if _looks_regression(series):
            kinds.append("regression")
        else:
            kinds.append("classification")

    unique = set(kinds)
    if len(unique) > 1:
        detail = {
            col: kind for col, kind in zip(target_columns, kinds, strict=True)
        }
        raise ValidationError(
            "Mixed classification/regression multi-task targets are not "
            f"supported (inferred kinds={detail}). Use same-type targets, or "
            "pass task='classification' / task='regression' only when every "
            "target is compatible with that task."
        )
    resolved = kinds[0]
    disclosures.append(
        f"Multi-task task='auto' inferred {resolved!r} for targets "
        f"{list(target_columns)}."
    )
    return resolved, disclosures


def encode_multitask_y(
    frame: pd.DataFrame,
    target_columns: Sequence[str],
    *,
    task: str,
    label_encoders: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any], dict[str, tuple[Any, ...]]]:
    """Build a 2D target matrix; LabelEncode per classification task."""
    from sklearn.preprocessing import LabelEncoder

    if task == "regression":
        cols = []
        for col in target_columns:
            series = frame[col]
            if series.isna().any():
                raise ValidationError(
                    f"Multi-task regression target {col!r} contains nulls."
                )
            if not pd.api.types.is_numeric_dtype(series):
                raise ValidationError(
                    f"Multi-task regression requires numeric target {col!r}."
                )
            cols.append(series.to_numpy(dtype=float))
        y = np.column_stack(cols)
        return y, {}, {}

    encoders: dict[str, Any] = dict(label_encoders or {})
    classes: dict[str, tuple[Any, ...]] = {}
    cols = []
    for col in target_columns:
        series = frame[col]
        if series.isna().any():
            raise ValidationError(
                f"Multi-task classification target {col!r} contains nulls."
            )
        values = series.astype(str)
        if col not in encoders:
            enc = LabelEncoder()
            codes = enc.fit_transform(values)
            encoders[col] = enc
        else:
            enc = encoders[col]
            known = {str(c) for c in enc.classes_}
            unknown = sorted(set(values) - known)
            if unknown:
                raise ValidationError(
                    f"Multi-task target {col!r} saw unseen class label(s): "
                    f"{unknown}."
                )
            codes = enc.transform(values)
        if len(enc.classes_) < 2:
            raise ValidationError(
                f"Multi-task classification target {col!r} needs >= 2 classes "
                f"(found {tuple(enc.classes_)!r})."
            )
        classes[col] = tuple(enc.classes_)
        cols.append(np.asarray(codes))
    y = np.column_stack(cols)
    return y, encoders, classes


def decode_multitask_predictions(
    raw: np.ndarray,
    target_columns: Sequence[str],
    *,
    task: str,
    label_encoders: dict[str, Any],
) -> dict[str, tuple[Any, ...]]:
    """Decode a (n, n_tasks) prediction matrix into per-task tuples."""
    arr = np.asarray(raw)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[1] != len(target_columns):
        raise ValidationError(
            f"Prediction width {arr.shape[1]} does not match "
            f"n_tasks={len(target_columns)}."
        )
    out: dict[str, tuple[Any, ...]] = {}
    for i, col in enumerate(target_columns):
        col_pred = arr[:, i]
        if task == "classification":
            enc = label_encoders[col]
            codes = np.rint(col_pred).astype(int)
            decoded = enc.inverse_transform(codes)
            out[col] = tuple(_coerce_label(v) for v in decoded)
        else:
            out[col] = tuple(float(v) for v in col_pred)
    return out


def _looks_regression(series: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(series):
        return False
    n = int(len(series))
    nunique = int(series.nunique(dropna=True))
    # Few distinct integers → treat as classification labels.
    if nunique <= max(10, int(0.05 * max(n, 1))):
        return False
    return True


def _assert_targets_compatible(
    frame: pd.DataFrame,
    target_columns: Sequence[str],
    task: str,
) -> None:
    for col in target_columns:
        series = frame[col]
        if series.isna().any():
            raise ValidationError(
                f"Multi-task target {col!r} contains nulls on the train "
                "partition."
            )
        if task == "regression" and not pd.api.types.is_numeric_dtype(series):
            raise ValidationError(
                f"task='regression' requires numeric target {col!r}."
            )
        if task == "classification" and _looks_regression(series):
            # Allow numeric class codes; only refuse clearly continuous columns
            # when the user forced classification? Keep permissive: continuous
            # numeric columns can still be LabelEncoded. No hard refuse here.
            pass


def _coerce_label(value: Any) -> Any:
    text = str(value)
    if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
        return int(text)
    try:
        return float(text) if "." in text else text
    except ValueError:
        return text
