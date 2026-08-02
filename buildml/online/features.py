"""Feature / chunk helpers for online learning (train-only updates)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, frame_for_partition
from buildml.semisupervised.features import (
    matrix_from_frame as _matrix_from_frame,
    resolve_semisupervised_columns,
)

__all__ = [
    "matrix_from_frame",
    "resolve_online_columns",
    "encode_classification_targets",
    "decode_predictions",
    "carve_train_chunk",
    "chunk_drift_notes",
    "align_external_frame",
]


def matrix_from_frame(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Build a float design matrix; refuse null features (online wording)."""
    try:
        return _matrix_from_frame(frame, columns)
    except ValidationError as exc:
        msg = str(exc).replace("Semi-supervised learning", "Online learning")
        raise ValidationError(msg) from exc


def resolve_online_columns(
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
    out = []
    for note in disclosures:
        out.append(note.replace("semi-supervised", "online-learning"))
    return cols, used_reduce, out


def encode_classification_targets(
    y: pd.Series,
    *,
    label_encoder: Any | None = None,
    classes: Sequence[Any] | None = None,
) -> tuple[np.ndarray, Any, tuple[Any, ...]]:
    """Encode classification targets; refuse missing labels in update chunks."""
    from sklearn.preprocessing import LabelEncoder

    if y.isna().any():
        raise ValidationError(
            "Online learning update/fit chunks require non-null targets. "
            "Blank rows are for active/semi-supervised pools, not partial_fit."
        )

    values = y.astype(str)
    if label_encoder is None:
        encoder = LabelEncoder()
        if classes is not None:
            encoder.fit([str(c) for c in classes])
            codes = encoder.transform(values)
        else:
            codes = encoder.fit_transform(values)
    else:
        encoder = label_encoder
        known = {str(c) for c in encoder.classes_}
        incoming = set(values)
        unknown = sorted(incoming - known)
        if unknown:
            raise ValidationError(
                "Online classifier saw new class label(s) not declared at init: "
                f"{unknown}. Pass classes= covering the full label vocabulary on "
                "fit_online (or ensure train targets already span all classes)."
            )
        codes = encoder.transform(values)
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


def carve_train_chunk(
    dataset: Dataset,
    split_plan: SplitPlan,
    *,
    cursor: int,
    n_rows: int,
    indices: Sequence[Any] | None = None,
) -> tuple[pd.DataFrame, list[Any], int]:
    """Carve the next train chunk (or explicit train indices) advancing cursor.

    Returns
    -------
    chunk_frame, dataset_indices, new_cursor
    """
    train_indices = list(split_plan.train_indices)
    n_train = len(train_indices)
    if indices is not None:
        requested = list(indices)
        train_set = set(train_indices)
        bad = [i for i in requested if i not in train_set]
        if bad:
            raise ValidationError(
                "Online updates refuse non-train indices (validation/test "
                f"leakage): {bad[:10]}{'...' if len(bad) > 10 else ''}."
            )
        full = dataset._ensure_pandas()
        missing = [i for i in requested if i not in full.index]
        if missing:
            raise ValidationError(
                f"Unknown dataset indices for online chunk: {missing[:10]}"
            )
        chunk = full.loc[requested]
        # Cursor advances past the highest train-order position among requested.
        positions = [train_indices.index(i) for i in requested]
        new_cursor = max(positions) + 1 if positions else cursor
        return chunk, requested, int(new_cursor)

    if n_rows < 1:
        raise ValidationError("n_rows for an online chunk must be >= 1.")
    if cursor >= n_train:
        raise ValidationError(
            "No remaining train rows for online updates "
            f"(cursor={cursor}, n_train={n_train})."
        )
    end = min(cursor + int(n_rows), n_train)
    chosen = train_indices[cursor:end]
    full = dataset._ensure_pandas()
    chunk = full.loc[chosen]
    return chunk, list(chosen), end


def align_external_frame(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    target_column: str,
) -> pd.DataFrame:
    """Validate a user-provided incremental frame against the plan contract."""
    if target_column not in frame.columns:
        raise ValidationError(
            f"External online chunk is missing target column {target_column!r}."
        )
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValidationError(
            f"External online chunk is missing feature columns: {missing}."
        )
    return frame.loc[:, list(columns) + [target_column]].copy()


def chunk_drift_notes(
    chunk_x: np.ndarray,
    init_means: Sequence[float] | None,
    *,
    columns: Sequence[str],
    enabled: bool = True,
) -> list[str]:
    """Lightweight mean-shift disclosure vs the init chunk (not a full drift product)."""
    if not enabled or init_means is None or chunk_x.size == 0:
        return []
    means = np.asarray(init_means, dtype=float)
    if means.shape[0] != chunk_x.shape[1]:
        return [
            "Drift disclosure skipped: init feature-mean length does not match "
            "the current chunk feature width."
        ]
    chunk_means = chunk_x.mean(axis=0)
    scale = np.maximum(np.abs(means), 1e-6)
    rel = np.abs(chunk_means - means) / scale
    flagged = [
        (str(columns[i]), float(rel[i]), float(chunk_means[i]), float(means[i]))
        for i in range(len(columns))
        if rel[i] >= 0.5
    ]
    notes = [
        "Optional drift disclosure compares this chunk's feature means to the "
        "init-chunk means (relative shift ≥ 0.5 flagged). This is not a full "
        "drift product — use Session.eda() train/test drift screens for richer "
        "partition diagnostics."
    ]
    if not flagged:
        notes.append("No feature mean relative-shift ≥ 0.5 vs init chunk.")
        return notes
    top = sorted(flagged, key=lambda row: row[1], reverse=True)[:5]
    detail = ", ".join(
        f"{name}: rel={rel_v:.2f} (chunk={c:.3g}, init={i:.3g})"
        for name, rel_v, c, i in top
    )
    notes.append(f"Flagged mean-shift columns (top): {detail}.")
    return notes


def regression_targets(y: pd.Series) -> np.ndarray:
    """Numeric regression targets; refuse nulls."""
    if y.isna().any():
        raise ValidationError(
            "Online regression chunks require non-null numeric targets."
        )
    if not pd.api.types.is_numeric_dtype(y):
        raise ValidationError(
            "Online regression requires a numeric target column."
        )
    return y.to_numpy(dtype=float)


def train_partition_frame(
    dataset: Dataset, split_plan: SplitPlan
) -> pd.DataFrame:
    """Train frame helper."""
    return frame_for_partition(dataset, split_plan, "train")
