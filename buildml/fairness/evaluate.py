"""Evaluate observational fairness metrics on labeled predictions."""

from __future__ import annotations

from typing import Any

import numpy as np

from buildml.core.errors import ValidationError
from buildml.fairness.metrics import (
    demographic_parity_difference,
    disparate_impact_ratio,
    equalized_odds_gaps,
    group_selection_rates,
)
from buildml.fairness.results import FairnessReport


def _unique_labels(values: np.ndarray) -> list[Any]:
    """Stable unique label list preserving first-seen order as strings for display."""
    seen: list[Any] = []
    for value in values.tolist():
        if value not in seen and value == value:  # skip NaN
            seen.append(value)
    return seen


def validate_positive_label(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    positive_label: Any,
) -> None:
    """Require ``positive_label`` to appear in truth and/or predictions.

    Hard-fails the soft-zero footgun where string labels (``\"Yes\"``/``\"No\"``)
    are compared against the default integer ``1``, producing empty matches and
    zero/NaN gaps without warning.

    Parameters
    ----------
    y_true, y_pred:
        Aligned label arrays.
    positive_label:
        Caller-declared positive class.

    Raises
    ------
    ValidationError
        When ``positive_label`` matches neither array, or when truth has no
        positives under that encoding.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    in_true = bool(np.any(yt == positive_label))
    in_pred = bool(np.any(yp == positive_label))
    if not in_true and not in_pred:
        observed = _unique_labels(yt)[:12]
        raise ValidationError(
            f"positive_label={positive_label!r} does not appear in y_true or "
            f"y_pred. Observed y_true labels (sample): {observed!r}. "
            "Pass the actual positive class value (e.g. 'approved' or 1)."
        )
    if not in_true:
        observed = _unique_labels(yt)[:12]
        raise ValidationError(
            f"positive_label={positive_label!r} does not appear in y_true "
            f"(found in predictions only). Observed y_true labels (sample): "
            f"{observed!r}. Equalized-odds metrics require positives in truth."
        )


def evaluate_fairness(
    y_true: Any,
    y_pred: Any,
    sensitive: Any,
    *,
    positive_label: Any = 1,
    partition: str = "test",
    sensitive_column: str = "sensitive",
) -> FairnessReport:
    """Compute holdout group disparity metrics for binary classification.

    Parameters
    ----------
    y_true, y_pred, sensitive:
        Aligned arrays/Series of labels, predictions, and group ids.
    positive_label:
        Label treated as the positive class. Must appear in ``y_true``
        (and typically ``y_pred``); misconfigured defaults raise
        :class:`~buildml.core.errors.ValidationError` instead of silent zeros.
    partition, sensitive_column:
        Metadata recorded on the report.

    Returns
    -------
    FairnessReport
        Per-group rates, gaps, and honesty disclosures.

    Raises
    ------
    ValidationError
        When lengths disagree, inputs are empty, or ``positive_label`` is
        misconfigured relative to observed labels.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    sens = np.asarray(sensitive)
    if len(yt) != len(yp) or len(yt) != len(sens):
        raise ValidationError("y_true, y_pred, and sensitive must have equal length.")
    if len(yt) == 0:
        raise ValidationError("Fairness evaluation requires at least one row.")

    validate_positive_label(yt, yp, positive_label=positive_label)

    rates = group_selection_rates(yp, sens, positive_label=positive_label)
    tpr, fpr, tpr_gap, fpr_gap = equalized_odds_gaps(
        yt, yp, sens, positive_label=positive_label
    )
    support = {
        g: int(sum(1 for s in sens if str(s) == g))
        for g in sorted({str(s) for s in sens})
    }
    warnings: list[str] = []
    if any(n < 30 for n in support.values()):
        warnings.append(
            "At least one group has support < 30; gap estimates are unstable."
        )
    n_groups = len(support)
    if n_groups < 2:
        warnings.append(
            "Fewer than two sensitive groups present; disparity gaps are undefined "
            "or trivial."
        )
    disclosures = (
        "Observational disparity on one partition: not a legal audit.",
        "Sensitive groups were caller-declared; BuildML did not infer them.",
        "Equalized odds gaps use TPR/FPR; undefined when a group lacks positives/negatives.",
        "positive_label is validated against observed y_true/y_pred before metrics run.",
        "Metrics are descriptive binary-classification gaps only; no mitigation applied.",
    )
    return FairnessReport(
        partition=partition,
        sensitive_column=sensitive_column,
        positive_label=positive_label,
        n_rows=int(len(yt)),
        groups=tuple(sorted(rates)),
        selection_rate_by_group=rates,
        demographic_parity_difference=demographic_parity_difference(rates),
        disparate_impact_ratio=disparate_impact_ratio(rates),
        equalized_odds_tpr_difference=tpr_gap,
        equalized_odds_fpr_difference=fpr_gap,
        tpr_by_group=tpr,
        fpr_by_group=fpr,
        support_by_group=support,
        disclosures=disclosures,
        warnings=tuple(warnings),
    )
