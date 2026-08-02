"""Feature / treatment / outcome helpers for causal ML (train-only fit)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.causal.types import CausalAssumptions
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, frame_for_partition
from buildml.semisupervised.features import matrix_from_frame as _matrix_from_frame

__all__ = [
    "matrix_from_frame",
    "validate_columns_present",
    "encode_binary_treatment",
    "outcome_array",
    "infer_outcome_kind",
    "design_matrix",
    "train_partition_frame",
    "partition_frame",
    "propensity_clip_bounds",
]


def matrix_from_frame(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Build a float design matrix; refuse null features."""
    if not columns:
        # Intercept-only design for allow_empty_confounders.
        return np.ones((len(frame), 1), dtype=float)
    try:
        return _matrix_from_frame(frame, columns)
    except ValidationError as exc:
        msg = str(exc).replace("Semi-supervised learning", "Causal learning")
        raise ValidationError(msg) from exc


def validate_columns_present(
    frame: pd.DataFrame,
    assumptions: CausalAssumptions,
) -> None:
    """Ensure treatment, outcome, and confounders exist in the frame."""
    needed = [assumptions.treatment, assumptions.outcome, *assumptions.confounders]
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise ValidationError(
            f"CausalAssumptions reference missing columns: {missing}."
        )


def encode_binary_treatment(
    series: pd.Series,
) -> tuple[np.ndarray, tuple[Any, Any], list[str]]:
    """Encode a binary treatment column as {0, 1}.

    Returns
    -------
    codes, (control_level, treated_level), disclosures
    """
    if series.isna().any():
        raise ValidationError("Causal treatment column contains nulls.")
    levels = list(pd.unique(series))
    if len(levels) != 2:
        raise ValidationError(
            f"Causal treatment must be binary (exactly 2 levels); found "
            f"{len(levels)} unique values. Multi-valued / continuous "
            "treatments are out of scope for this surface."
        )
    # Prefer numeric 0/1 ordering when present; else sort lexicographically
    # with the first level as control.
    as_str = [str(v) for v in levels]
    if set(as_str) == {"0", "1"}:
        control, treated = (
            levels[as_str.index("0")],
            levels[as_str.index("1")],
        )
    elif set(levels) == {0, 1} or set(levels) == {0.0, 1.0}:
        control, treated = 0 if 0 in levels else 0.0, 1 if 1 in levels else 1.0
    else:
        ordered = sorted(levels, key=lambda v: str(v))
        control, treated = ordered[0], ordered[1]
    codes = np.where(series.to_numpy() == treated, 1, 0).astype(int)
    disclosures = [
        f"Binary treatment encoded as control={control!r} → 0, "
        f"treated={treated!r} → 1."
    ]
    return codes, (control, treated), disclosures


def infer_outcome_kind(series: pd.Series) -> str:
    """Infer continuous vs binary outcome for nuisance model choice."""
    if series.isna().any():
        raise ValidationError("Causal outcome column contains nulls.")
    if not pd.api.types.is_numeric_dtype(series):
        # Allow two-level categorical binary outcomes.
        levels = pd.unique(series.astype(str))
        if len(levels) == 2:
            return "binary"
        raise ValidationError(
            "Causal outcome must be numeric (continuous) or a binary label."
        )
    values = series.to_numpy()
    uniq = np.unique(values)
    if len(uniq) == 2 and set(np.asarray(uniq, dtype=float).tolist()) <= {0.0, 1.0}:
        return "binary"
    return "continuous"


def outcome_array(series: pd.Series, *, kind: str) -> np.ndarray:
    """Materialize outcome as float array."""
    if series.isna().any():
        raise ValidationError("Causal outcome column contains nulls.")
    if kind == "binary":
        if pd.api.types.is_numeric_dtype(series):
            arr = series.to_numpy(dtype=float)
            uniq = set(np.unique(arr).tolist())
            if not uniq <= {0.0, 1.0}:
                # Map two numeric levels to 0/1 by sorted order.
                levels = sorted(uniq)
                if len(levels) != 2:
                    raise ValidationError("Binary outcome must have exactly 2 levels.")
                arr = np.where(arr == levels[1], 1.0, 0.0)
            return arr.astype(float)
        levels = sorted(pd.unique(series.astype(str)).tolist())
        if len(levels) != 2:
            raise ValidationError("Binary outcome must have exactly 2 levels.")
        return np.where(series.astype(str).to_numpy() == levels[1], 1.0, 0.0)
    if not pd.api.types.is_numeric_dtype(series):
        raise ValidationError("Continuous causal outcome must be numeric.")
    return series.to_numpy(dtype=float)


def design_matrix(
    frame: pd.DataFrame,
    confounders: Sequence[str],
) -> np.ndarray:
    """Confounder design matrix (ones column when empty confounders waived)."""
    return matrix_from_frame(frame, list(confounders))


def train_partition_frame(dataset: Dataset, split_plan: SplitPlan) -> pd.DataFrame:
    return frame_for_partition(dataset, split_plan, "train")


def partition_frame(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    partition: str,
) -> pd.DataFrame:
    if partition == "all":
        return dataset._ensure_pandas()
    if split_plan is None:
        raise ValidationError("Causal partition access requires a SplitPlan.")
    return frame_for_partition(dataset, split_plan, partition)  # type: ignore[arg-type]


def propensity_clip_bounds(
    clip: tuple[float, float] | Sequence[float],
) -> tuple[float, float]:
    if len(clip) != 2:
        raise ValidationError("clip_propensity must be a (low, high) pair.")
    low, high = float(clip[0]), float(clip[1])
    if not (0.0 < low < high < 1.0):
        raise ValidationError(
            f"clip_propensity must satisfy 0 < low < high < 1; got {(low, high)}."
        )
    return low, high
