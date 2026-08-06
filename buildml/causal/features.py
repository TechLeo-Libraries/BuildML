"""Feature / treatment / outcome helpers for causal ML (train-only fit)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from buildml.causal.types import CausalAssumptions
from buildml.core.errors import ValidationError
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
    """Build a float design matrix from selected frame columns.

    Delegates to the semi-supervised feature encoder and rewrites validation
    errors so causal callers see causal-specific messaging. An empty column
    list yields an intercept-only matrix of ones when
    ``allow_empty_confounders`` is in effect.

    Parameters
    ----------
    frame:
        Partition rows containing confounder columns.
    columns:
        Confounder column names to encode as floats.

    Returns
    -------
    numpy.ndarray
        ``(n_rows, n_features)`` float matrix with no null entries.

    Raises
    ------
    ValidationError
        When any selected column contains nulls or cannot be encoded.
    """
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
    """Ensure treatment, outcome, and confounders exist in the frame.

    Called before fit, estimate, evaluate, and refute so missing columns fail
    fast with an assumption-aware error rather than a downstream pandas KeyError.

    Parameters
    ----------
    frame:
        Partition rows to score or refit on.
    assumptions:
        Declared causal column contract.

    Raises
    ------
    ValidationError
        When any declared treatment, outcome, or confounder column is absent.
    """
    needed = [assumptions.treatment, assumptions.outcome, *assumptions.confounders]
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise ValidationError(
            f"CausalAssumptions reference missing columns: {missing}."
        )


def encode_binary_treatment(
    series: pd.Series,
) -> tuple[np.ndarray, tuple[Any, Any], list[str]]:
    """Encode a binary treatment column as ``{0, 1}``.

    Maps the control arm to 0 and the treated arm to 1 using numeric 0/1 when
    present, otherwise lexicographic ordering of the two unique levels. Emits
    teaching disclosures so Session history records the encoding choice.

    Parameters
    ----------
    series:
        Treatment column from a single partition.

    Returns
    -------
    codes : numpy.ndarray
        Integer treatment indicators (0 = control, 1 = treated).
    levels : tuple[Any, Any]
        ``(control_level, treated_level)`` in original column dtype.
    disclosures : list[str]
        Human-readable notes about the encoding applied.

    Raises
    ------
    ValidationError
        When the column contains nulls or does not have exactly two levels.
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
    """Infer continuous vs binary outcome for nuisance model choice.

    Binary outcomes are detected from two-level categoricals or numeric
    columns whose unique values are a subset of ``{0, 1}``. All other numeric
    columns are treated as continuous for ridge/logistic first-stage selection.

    Parameters
    ----------
    series:
        Outcome column from the train partition.

    Returns
    -------
    str
        ``"binary"`` or ``"continuous"``.

    Raises
    ------
    ValidationError
        When the column contains nulls or is neither numeric nor two-level.
    """
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
    """Materialize outcome as a float array for sklearn/EconML paths.

    Binary outcomes are mapped to ``{0.0, 1.0}`` using sorted level order when
    the raw column is categorical or uses non-standard numeric labels.

    Parameters
    ----------
    series:
        Outcome column from the active partition.
    kind:
        ``"binary"`` or ``"continuous"`` as returned by
        :func:`infer_outcome_kind`.

    Returns
    -------
    numpy.ndarray
        One-dimensional float outcome vector aligned with ``series`` index.

    Raises
    ------
    ValidationError
        When the column contains nulls or cannot be coerced to the requested kind.
    """
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
    """Build the confounder design matrix for nuisance model fit.

    Thin wrapper around :func:`matrix_from_frame` so fit, estimate, and refute
    paths share the same empty-confounder intercept-only behaviour.

    Parameters
    ----------
    frame:
        Partition rows containing confounder columns.
    confounders:
        Declared backdoor adjustment set (may be empty when waived).

    Returns
    -------
    numpy.ndarray
        Float design matrix passed to outcome and propensity estimators.
    """
    return matrix_from_frame(frame, list(confounders))


def train_partition_frame(dataset: Dataset, split_plan: SplitPlan) -> pd.DataFrame:
    """Return the Session train partition as a pandas DataFrame.

    Convenience wrapper used by causal fit and refute paths before column
    validation and nuisance model fitting.

    Parameters
    ----------
    dataset:
        Session dataset.
    split_plan:
        Split plan with train indices.

    Returns
    -------
    pandas.DataFrame
        Rows indexed by ``split_plan.train_indices``.
    """
    return frame_for_partition(dataset, split_plan, "train")


def partition_frame(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    partition: str,
) -> pd.DataFrame:
    """Return a holdout or full-dataset frame for causal scoring.

    Wraps split-plan partition selection so estimate, evaluate, and refute
    paths share the same frame contract.

    Parameters
    ----------
    dataset:
        Session dataset.
    split_plan:
        Split plan required unless ``partition='all'``.
    partition:
        ``train``, ``validation``, ``test``, or ``all``.

    Returns
    -------
    pandas.DataFrame
        Rows for the requested partition.

    Raises
    ------
    ValidationError
        When a named partition is requested without a split plan.
    """
    if partition == "all":
        return dataset._ensure_pandas()
    if split_plan is None:
        raise ValidationError("Causal partition access requires a SplitPlan.")
    return frame_for_partition(dataset, split_plan, partition)  # type: ignore[arg-type]


def propensity_clip_bounds(
    clip: tuple[float, float] | Sequence[float],
) -> tuple[float, float]:
    """Validate and normalise IPW/AIPW propensity clipping bounds.

    Clipping stabilises inverse-propensity weights when estimated scores approach
    0 or 1; bounds must lie strictly inside the unit interval.

    Parameters
    ----------
    clip:
        ``(low, high)`` pair from :class:`~buildml.causal.types.CausalConfig`.

    Returns
    -------
    tuple[float, float]
        Validated ``(low, high)`` clipping thresholds.

    Raises
    ------
    ValidationError
        When ``clip`` is not a length-2 pair or violates ``0 < low < high < 1``.
    """
    if len(clip) != 2:
        raise ValidationError("clip_propensity must be a (low, high) pair.")
    low, high = float(clip[0]), float(clip[1])
    if not (0.0 < low < high < 1.0):
        raise ValidationError(
            f"clip_propensity must satisfy 0 < low < high < 1; got {(low, high)}."
        )
    return low, high
