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
        Label treated as the positive class.
    partition, sensitive_column:
        Metadata recorded on the report.

    Returns
    -------
    FairnessReport
        Per-group rates, gaps, and honesty disclosures.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    sens = np.asarray(sensitive)
    if len(yt) != len(yp) or len(yt) != len(sens):
        raise ValidationError("y_true, y_pred, and sensitive must have equal length.")
    if len(yt) == 0:
        raise ValidationError("Fairness evaluation requires at least one row.")

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
    disclosures = (
        "Observational disparity on one partition: not a legal audit.",
        "Sensitive groups were caller-declared; BuildML did not infer them.",
        "Equalized odds gaps use TPR/FPR; undefined when a group lacks positives/negatives.",
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
