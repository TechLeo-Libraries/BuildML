"""Feature / train-carve helpers for probabilistic ML (train-only fit)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, frame_for_partition
from buildml.semisupervised.features import (
    matrix_from_frame as _matrix_from_frame,
    resolve_semisupervised_columns,
)

__all__ = [
    "matrix_from_frame",
    "resolve_probabilistic_columns",
    "encode_classification_targets",
    "decode_predictions",
    "regression_targets",
    "train_partition_frame",
    "split_train_for_conformal",
    "norm_ppf",
]


def matrix_from_frame(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Build a float design matrix; refuse null features."""
    try:
        return _matrix_from_frame(frame, columns)
    except ValidationError as exc:
        msg = str(exc).replace("Semi-supervised learning", "Probabilistic learning")
        raise ValidationError(msg) from exc


def resolve_probabilistic_columns(
    dataset: Dataset,
    frame: pd.DataFrame,
    columns: list[str] | None,
    *,
    reduce_plan: Any | None = None,
    prefer_reduce_components: bool = True,
    target_column: str,
) -> tuple[list[str], bool, list[str]]:
    """Resolve numeric feature columns (same contract as semi-supervised)."""
    cols, used_reduce, disclosures = resolve_semisupervised_columns(
        dataset,
        frame,
        columns,
        reduce_plan=reduce_plan,
        prefer_reduce_components=prefer_reduce_components,
        target_column=target_column,
    )
    out = [
        note.replace("semi-supervised", "probabilistic") for note in disclosures
    ]
    return cols, used_reduce, out


def encode_classification_targets(
    y: pd.Series,
    *,
    classes: Sequence[Any] | None = None,
) -> tuple[np.ndarray, Any, tuple[Any, ...]]:
    """Encode classification targets; refuse missing labels."""
    from sklearn.preprocessing import LabelEncoder

    if y.isna().any():
        raise ValidationError(
            "Probabilistic classification requires non-null train targets."
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
    """Map integer class codes back toward original label values."""
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
    """Numeric regression targets; refuse nulls."""
    if y.isna().any():
        raise ValidationError(
            "Probabilistic regression requires non-null numeric targets."
        )
    if not pd.api.types.is_numeric_dtype(y):
        raise ValidationError(
            "Probabilistic regression requires a numeric target column."
        )
    return y.to_numpy(dtype=float)


def train_partition_frame(
    dataset: Dataset, split_plan: SplitPlan
) -> pd.DataFrame:
    """Train frame helper."""
    return frame_for_partition(dataset, split_plan, "train")


def split_train_for_conformal(
    train_indices: Sequence[Any],
    *,
    calibration_fraction: float,
    random_state: int | None,
    min_fit: int = 5,
    min_calib: int = 5,
    stratify_labels: Sequence[Any] | None = None,
) -> tuple[list[Any], list[Any]]:
    """Carve a conformal calibration subset from the Session train partition.

    Never uses validation/test indices. Returns (fit_indices, calib_indices).
    When ``stratify_labels`` is provided (classification), the carve keeps
    every label in both the fit and calibration subsets when feasible.
    """
    indices = list(train_indices)
    n = len(indices)
    if n < min_fit + min_calib:
        raise ValidationError(
            "Split conformal needs enough train rows to carve a calibration "
            f"subset (need >= {min_fit + min_calib}, found {n}). "
            "Disable conformal=False or grow the train partition."
        )
    frac = float(calibration_fraction)
    if not 0.05 <= frac <= 0.5:
        raise ValidationError(
            "conformal_calibration_fraction must be in [0.05, 0.5]."
        )
    n_calib = max(min_calib, int(round(n * frac)))
    n_calib = min(n_calib, n - min_fit)
    n_fit = n - n_calib
    if n_fit < min_fit or n_calib < min_calib:
        raise ValidationError(
            "Could not carve a valid train-only conformal calibration split "
            f"(n_fit={n_fit}, n_calib={n_calib}, n_train={n})."
        )

    if stratify_labels is not None:
        if len(stratify_labels) != n:
            raise ValidationError(
                "stratify_labels length must match train_indices for conformal carve."
            )
        return _stratified_conformal_split(
            indices,
            [str(v) for v in stratify_labels],
            n_calib=n_calib,
            random_state=random_state,
            min_fit=min_fit,
            min_calib=min_calib,
        )

    rng = np.random.default_rng(random_state)
    order = rng.permutation(n)
    calib_pos = order[:n_calib]
    fit_pos = order[n_calib:]
    fit_idx = [indices[i] for i in sorted(fit_pos.tolist())]
    calib_idx = [indices[i] for i in sorted(calib_pos.tolist())]
    return fit_idx, calib_idx


def _stratified_conformal_split(
    indices: list[Any],
    labels: list[str],
    *,
    n_calib: int,
    random_state: int | None,
    min_fit: int,
    min_calib: int,
) -> tuple[list[Any], list[Any]]:
    """Per-class carve so rare labels stay in both fit and calib when possible."""
    rng = np.random.default_rng(random_state)
    by_label: dict[str, list[Any]] = {}
    for idx, label in zip(indices, labels, strict=True):
        by_label.setdefault(label, []).append(idx)

    if any(len(rows) < 2 for rows in by_label.values()):
        raise ValidationError(
            "Split conformal classification needs at least 2 train rows per "
            "class so the train-only calibration carve can keep every label "
            "in both fit and calib. Pass conformal=False or grow the train set."
        )

    calib: list[Any] = []
    fit: list[Any] = []
    # First place one row per class into each subset.
    for label, rows in by_label.items():
        order = list(rng.permutation(len(rows)))
        ordered = [rows[i] for i in order]
        calib.append(ordered[0])
        fit.append(ordered[1])
        rest = ordered[2:]
        by_label[label] = rest

    remaining = [idx for rows in by_label.values() for idx in rows]
    remaining = list(rng.permutation(remaining))
    need_calib = max(0, n_calib - len(calib))
    if need_calib > len(remaining):
        need_calib = len(remaining)
    calib.extend(remaining[:need_calib])
    fit.extend(remaining[need_calib:])

    if len(fit) < min_fit or len(calib) < min_calib:
        raise ValidationError(
            "Stratified conformal carve could not satisfy min_fit/min_calib "
            f"(n_fit={len(fit)}, n_calib={len(calib)})."
        )
    # Preserve original train order for readability.
    index_pos = {idx: pos for pos, idx in enumerate(indices)}
    fit_sorted = sorted(fit, key=lambda i: index_pos[i])
    calib_sorted = sorted(calib, key=lambda i: index_pos[i])
    return fit_sorted, calib_sorted


def norm_ppf(p: float) -> float:
    """Inverse CDF of the standard normal (via ``scipy.stats.norm``)."""
    if not 0.0 < p < 1.0:
        raise ValidationError(f"norm_ppf probability must be in (0, 1); got {p}.")
    return float(norm.ppf(p))
