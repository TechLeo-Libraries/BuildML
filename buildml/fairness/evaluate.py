"""Evaluate observational fairness metrics on labeled predictions."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from buildml.core.errors import ValidationError
from buildml.fairness.classical_bridge import per_group_classical_metrics
from buildml.fairness.groups import (
    compose_group_keys,
    normalize_sensitive_columns,
    sensitive_column_label,
)
from buildml.fairness.metrics import (
    demographic_parity_difference,
    disparate_impact_ratio,
    equalized_odds_gaps,
    group_selection_rates,
)
from buildml.fairness.results import FairnessReport
from buildml.fairness.stability import StabilityMethod, estimate_gap_stability


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
    sensitive_column: str | Sequence[str] = "sensitive",
    y_score: Any | None = None,
    include_classical_metrics: bool = True,
    bootstrap_samples: int = 0,
    stability_method: StabilityMethod = "bootstrap",
    subsample_fraction: float = 0.8,
    confidence_level: float = 0.95,
    random_state: int | None = 0,
) -> FairnessReport:
    """Compute holdout group disparity metrics for binary classification.

    Parameters
    ----------
    y_true, y_pred:
        Aligned arrays/Series of labels and predictions.
    sensitive:
        Group ids, **or** a 2-d array / sequence of arrays when auditing
        multiple sensitive attributes. Prefer passing already-composed keys
        from :func:`~buildml.fairness.groups.compose_group_keys`, or pass a
        list/tuple of columns via ``sensitive_column`` at the Session layer.
        When ``sensitive`` is 2-d (n, k), columns are composed into
        intersectional keys automatically.
    positive_label:
        Label treated as the positive class. Must appear in ``y_true``
        (and typically ``y_pred``); misconfigured defaults raise
        :class:`~buildml.core.errors.ValidationError` instead of silent zeros.
    partition, sensitive_column:
        Metadata recorded on the report. ``sensitive_column`` may be a
        sequence of names for intersectional audits.
    y_score:
        Optional positive-class scores for per-group ROC-AUC.
    include_classical_metrics:
        When True (default), attach per-group accuracy/precision/recall/F1
        (and AUC when scores are available).
    bootstrap_samples:
        When ``> 1``, attach stability bands via resampling. ``0`` disables.
    stability_method, subsample_fraction, confidence_level, random_state:
        Stability configuration (see
        :func:`~buildml.fairness.stability.estimate_gap_stability`).

    Returns
    -------
    FairnessReport
        Per-group rates, gaps, optional classical bridge + stability, and
        honesty disclosures.

    Raises
    ------
    ValidationError
        When lengths disagree, inputs are empty, or ``positive_label`` is
        misconfigured relative to observed labels.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    sens_raw = np.asarray(sensitive)
    if sens_raw.ndim == 2:
        if sens_raw.shape[0] != len(yt):
            raise ValidationError(
                "2-d sensitive array must have one row per example."
            )
        if sens_raw.shape[1] < 1:
            raise ValidationError("2-d sensitive array must have at least one column.")
        sens = compose_group_keys(*[sens_raw[:, j] for j in range(sens_raw.shape[1])])
        if isinstance(sensitive_column, str):
            cols = tuple(f"sensitive_{j}" for j in range(sens_raw.shape[1]))
        else:
            cols = normalize_sensitive_columns(sensitive_column)
            if len(cols) != sens_raw.shape[1]:
                raise ValidationError(
                    "Length of sensitive_column names must match the number of "
                    "sensitive attribute columns."
                )
    elif sens_raw.ndim == 1:
        sens = np.asarray(
            [str(s) if s == s else "<NA>" for s in sens_raw], dtype=object
        )
        cols = normalize_sensitive_columns(sensitive_column)
    else:
        raise ValidationError(
            "sensitive must be 1-d group keys or a 2-d attribute matrix."
        )

    if len(yt) != len(yp) or len(yt) != len(sens):
        raise ValidationError("y_true, y_pred, and sensitive must have equal length.")
    if len(yt) == 0:
        raise ValidationError("Fairness evaluation requires at least one row.")

    validate_positive_label(yt, yp, positive_label=positive_label)

    scores = None if y_score is None else np.asarray(y_score, dtype=float)
    if scores is not None and len(scores) != len(yt):
        raise ValidationError("y_score must have the same length as y_true.")

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
    intersectional = len(cols) > 1
    if intersectional and any(n < 30 for n in support.values()):
        warnings.append(
            "Intersectional groups often have sparse support; interpret gaps "
            "cautiously and prefer stability bands."
        )

    classical: dict[str, dict[str, float | None]] = {}
    if include_classical_metrics:
        classical = per_group_classical_metrics(
            yt, yp, sens, positive_label=positive_label, y_score=scores
        )

    stability = None
    if bootstrap_samples and bootstrap_samples > 1:
        stability = estimate_gap_stability(
            yt,
            yp,
            sens,
            positive_label=positive_label,
            n_resamples=int(bootstrap_samples),
            confidence_level=confidence_level,
            method=stability_method,
            subsample_fraction=subsample_fraction,
            random_state=random_state,
        )

    col_label = sensitive_column_label(cols)
    scope = {
        "kind": "observational_holdout",
        "legal_audit": False,
        "causal_fairness": False,
        "mitigation_applied": False,
        "intersectional": intersectional,
        "sensitive_columns": list(cols),
        "classical_metrics_included": bool(include_classical_metrics),
        "stability_enabled": stability is not None,
        "scores_used_for_auc": scores is not None,
    }
    disclosures: tuple[str, ...] = (
        "Observational disparity on one partition: not a legal audit.",
        "Sensitive groups were caller-declared; BuildML did not infer them.",
        "Equalized odds gaps use TPR/FPR; undefined when a group lacks positives/negatives.",
        "positive_label is validated against observed y_true/y_pred before metrics run.",
        "Metrics are descriptive binary-classification gaps only; no mitigation applied "
        "by evaluate_fairness.",
        "Optional mitigation helpers (threshold equalization / reweighing) are opt-in "
        "and disclosed separately — they are not certification.",
    )
    if intersectional:
        disclosures = disclosures + (
            "Intersectional keys join attribute levels with '|'; sparse cells "
            "are expected and disclosed via support_by_group.",
        )
    if stability is not None:
        disclosures = disclosures + stability.disclosures

    return FairnessReport(
        partition=partition,
        sensitive_column=col_label,
        sensitive_columns=cols,
        intersectional=intersectional,
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
        classical_metrics_by_group=classical,
        stability=stability,
        scope=scope,
        disclosures=disclosures,
        warnings=tuple(warnings),
    )
