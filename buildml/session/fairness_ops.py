"""Session orchestration for fairness disparity reporting."""

from __future__ import annotations

from typing import Any, Sequence

from buildml.core.errors import ValidationError
from buildml.fairness.catalog import fairness_capability_matrix as _fairness_capability_matrix
from buildml.fairness.evaluate import evaluate_fairness as _evaluate_fairness
from buildml.fairness.groups import extract_sensitive_keys, normalize_sensitive_columns
from buildml.fairness.mitigation import (
    GroupThresholdSuggestion,
    ReweighingSuggestion,
    suggest_group_thresholds as _suggest_group_thresholds,
    suggest_reweighing_weights as _suggest_reweighing_weights,
)
from buildml.fairness.results import FairnessReport
from buildml.fairness.stability import StabilityMethod


def fairness_capability_matrix() -> dict[str, Any]:
    """Return the observational fairness capability / non-goal matrix.

    Prefer ``session.fairness.capability_matrix()`` on a live Session. The flat
    ``Session.fairness_capability_matrix`` alias remains until BuildML 3.0.

    Returns
    -------
    dict[str, Any]
        Backends, metrics, disclosures, and explicit non-goals for fairness.
    """
    return _fairness_capability_matrix()


def _partition_indices(session: Any, partition: str) -> list[int]:
    if session.split_plan is None:
        raise ValidationError("Fairness operations require a split plan.")
    part = str(partition).lower()
    split = session.split_plan
    if part == "test":
        idx = split.test_indices
    elif part == "validation":
        idx = split.validation_indices
    elif part == "train":
        idx = split.train_indices
    else:
        raise ValidationError(f"Unknown partition {partition!r}.")
    if idx is None or len(idx) == 0:
        raise ValidationError(f"Partition {partition!r} has no rows.")
    return list(idx)


def _resolve_last_eval_partition(session: Any, default: str = "test") -> str:
    """Find the most recent classical ``evaluate`` partition from history."""
    stored = getattr(session, "_last_evaluate_partition", None)
    if stored:
        return str(stored)
    history = getattr(session, "history", None)
    if not isinstance(history, list):
        return default
    for entry in reversed(history):
        if not isinstance(entry, dict):
            continue
        action = (
            entry.get("operation_id")
            or entry.get("action")
            or entry.get("operation")
        )
        if action != "evaluate":
            continue
        details = entry.get("parameters") or entry.get("details") or {}
        part = details.get("partition")
        if part:
            return str(part)
    return default


def _positive_scores(session: Any, partition: str, positive_label: Any) -> Any | None:
    """Best-effort positive-class scores via ``predict(..., return_proba=True)``."""
    try:
        proba = session.predict(partition=partition, return_proba=True)  # type: ignore[misc]
    except Exception:  # noqa: BLE001 — scores are optional for AUC bridge
        return None
    if proba is None:
        return None
    if hasattr(proba, "columns"):
        cols = list(proba.columns)
        if positive_label in cols:
            return proba[positive_label].to_numpy()
        # Common binary layout: columns [0, 1] or class labels
        if len(cols) == 2:
            # Prefer column matching positive_label string form, else last/col 1
            for c in cols:
                if str(c) == str(positive_label):
                    return proba[c].to_numpy()
            return proba[cols[-1]].to_numpy()
        if len(cols) == 1:
            return proba[cols[0]].to_numpy()
        return None
    import numpy as np

    arr = np.asarray(proba)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, -1]
    return None


def evaluate_fairness_op(
    session: Any,
    *,
    sensitive_column: str | Sequence[str],
    partition: str = "test",
    positive_label: Any = 1,
    include_classical_metrics: bool = True,
    bootstrap_samples: int = 0,
    stability_method: StabilityMethod = "bootstrap",
    subsample_fraction: float = 0.8,
    confidence_level: float = 0.95,
    random_state: int | None = 0,
    include_scores: bool = True,
) -> FairnessReport:
    """Score group disparity on a holdout partition using the fitted classifier.

    Prefer ``session.fairness.evaluate(...)``. This ops helper is the canonical
    implementation behind both the facade and the deprecated flat alias.

    Parameters
    ----------
    session:
        Fitted Session with roles, split, and a classical classifier.
    sensitive_column:
        Column holding caller-declared group ids, or a sequence of columns
        for intersectional (composite) group keys.
    partition:
        Holdout partition name (default ``test``).
    positive_label:
        Positive class label for rates and equalized odds.
    include_classical_metrics:
        Attach per-group accuracy/precision/recall/F1 (and AUC when scores
        are available).
    bootstrap_samples:
        When ``> 1``, attach stability bands (bootstrap or stratified subsample).
    stability_method, subsample_fraction, confidence_level, random_state:
        Stability configuration.
    include_scores:
        Attempt to fetch positive-class probabilities for per-group AUC.

    Returns
    -------
    FairnessReport
        Observational gaps and disclosures; stored on ``session._fairness_report``.

    Raises
    ------
    ValidationError
        When the Session lacks a fit/split/dataset, sensitive columns are
        missing, the partition is empty/unknown, or prediction length mismatches.
    """
    if getattr(session, "_fit_result", None) is None:
        raise ValidationError(
            "evaluate_fairness requires a fitted classifier; call Session.fit first."
        )
    dataset = session.dataset
    if dataset is None:
        raise ValidationError("Session has no dataset.")

    cols = normalize_sensitive_columns(sensitive_column)
    missing = [c for c in cols if c not in dataset.frame.columns]
    if missing:
        raise ValidationError(
            f"sensitive_column(s) not in the Session dataset: {missing!r}."
        )

    idx = _partition_indices(session, partition)
    y_hat = session.predict(partition=partition)  # type: ignore[misc]
    if hasattr(y_hat, "to_numpy"):
        y_hat_arr = y_hat.to_numpy()
    else:
        y_hat_arr = list(y_hat)

    frame = dataset.frame.iloc[idx]
    target = dataset.require_target()
    y_true = frame[target].to_numpy()
    sensitive = extract_sensitive_keys(frame, cols)
    if len(y_hat_arr) != len(y_true):
        raise ValidationError(
            "Prediction length does not match fairness partition rows."
        )

    y_score = None
    if include_scores:
        y_score = _positive_scores(session, partition, positive_label)

    report = _evaluate_fairness(
        y_true,
        y_hat_arr,
        sensitive,
        positive_label=positive_label,
        partition=partition,
        sensitive_column=cols,
        y_score=y_score,
        include_classical_metrics=include_classical_metrics,
        bootstrap_samples=bootstrap_samples,
        stability_method=stability_method,
        subsample_fraction=subsample_fraction,
        confidence_level=confidence_level,
        random_state=random_state,
    )
    session._fairness_report = report
    history = getattr(session, "history", None)
    if isinstance(history, list):
        history.append(
            {
                "operation": "evaluate_fairness",
                "partition": partition,
                "sensitive_column": list(cols),
                "intersectional": report.intersectional,
                "n_rows": report.n_rows,
                "demographic_parity_difference": report.demographic_parity_difference,
                "stability_enabled": report.stability is not None,
            }
        )
    return report


def attach_fairness_to_last_eval_op(
    session: Any,
    *,
    sensitive_column: str | Sequence[str],
    positive_label: Any = 1,
    include_classical_metrics: bool = True,
    bootstrap_samples: int = 0,
    stability_method: StabilityMethod = "bootstrap",
    subsample_fraction: float = 0.8,
    confidence_level: float = 0.95,
    random_state: int | None = 0,
    include_scores: bool = True,
    partition: str | None = None,
) -> FairnessReport:
    """Run fairness on the partition used by the latest classical ``evaluate``.

    Prefer ``session.fairness.attach_to_last_eval(...)``. Resolves the partition
    from Session history (most recent ``evaluate``) or ``partition`` override;
    defaults to ``test`` when no evaluate history exists.

    This does **not** modify classical evaluate metrics. It stores a
    :class:`~buildml.fairness.results.FairnessReport` on the Session like
    :func:`evaluate_fairness_op`.
    """
    resolved = partition or _resolve_last_eval_partition(session, default="test")
    return evaluate_fairness_op(
        session,
        sensitive_column=sensitive_column,
        partition=resolved,
        positive_label=positive_label,
        include_classical_metrics=include_classical_metrics,
        bootstrap_samples=bootstrap_samples,
        stability_method=stability_method,
        subsample_fraction=subsample_fraction,
        confidence_level=confidence_level,
        random_state=random_state,
        include_scores=include_scores,
    )


def suggest_fairness_thresholds_op(
    session: Any,
    *,
    sensitive_column: str | Sequence[str],
    partition: str = "validation",
    positive_label: Any = 1,
    target: str = "demographic_parity",
    grid_size: int = 101,
) -> GroupThresholdSuggestion:
    """Suggest per-group score thresholds (opt-in; not auto-applied).

    Prefer ``session.fairness.suggest_thresholds(...)``. Default partition is
    ``validation`` to discourage test-set threshold fishing.
    """
    if getattr(session, "_fit_result", None) is None:
        raise ValidationError(
            "suggest_thresholds requires a fitted classifier; call Session.fit first."
        )
    dataset = session.dataset
    if dataset is None:
        raise ValidationError("Session has no dataset.")
    cols = normalize_sensitive_columns(sensitive_column)
    missing = [c for c in cols if c not in dataset.frame.columns]
    if missing:
        raise ValidationError(
            f"sensitive_column(s) not in the Session dataset: {missing!r}."
        )
    idx = _partition_indices(session, partition)
    frame = dataset.frame.iloc[idx]
    target_col = dataset.require_target()
    y_true = frame[target_col].to_numpy()
    sensitive = extract_sensitive_keys(frame, cols)
    y_score = _positive_scores(session, partition, positive_label)
    if y_score is None:
        raise ValidationError(
            "suggest_thresholds requires predict_proba scores; the fitted "
            "estimator did not return usable probabilities."
        )
    suggestion = _suggest_group_thresholds(
        y_true,
        y_score,
        sensitive,
        positive_label=positive_label,
        target=target,  # type: ignore[arg-type]
        grid_size=grid_size,
    )
    session._fairness_mitigation_suggestion = suggestion
    history = getattr(session, "history", None)
    if isinstance(history, list):
        history.append(
            {
                "operation": "suggest_fairness_thresholds",
                "partition": partition,
                "sensitive_column": list(cols),
                "target": target,
                "n_rows": suggestion.n_rows,
            }
        )
    return suggestion


def suggest_fairness_reweighing_op(
    session: Any,
    *,
    sensitive_column: str | Sequence[str],
    partition: str = "train",
    positive_label: Any = 1,
) -> ReweighingSuggestion:
    """Suggest Kamiran–Calders sample weights (opt-in; not auto-applied).

    Prefer ``session.fairness.suggest_reweighing(...)``. Default partition is
    ``train``. Weights are returned for the caller to pass into a future fit.
    """
    dataset = session.dataset
    if dataset is None:
        raise ValidationError("Session has no dataset.")
    cols = normalize_sensitive_columns(sensitive_column)
    missing = [c for c in cols if c not in dataset.frame.columns]
    if missing:
        raise ValidationError(
            f"sensitive_column(s) not in the Session dataset: {missing!r}."
        )
    idx = _partition_indices(session, partition)
    frame = dataset.frame.iloc[idx]
    target_col = dataset.require_target()
    y_true = frame[target_col].to_numpy()
    sensitive = extract_sensitive_keys(frame, cols)
    suggestion = _suggest_reweighing_weights(
        y_true, sensitive, positive_label=positive_label
    )
    session._fairness_mitigation_suggestion = suggestion
    history = getattr(session, "history", None)
    if isinstance(history, list):
        history.append(
            {
                "operation": "suggest_fairness_reweighing",
                "partition": partition,
                "sensitive_column": list(cols),
                "n_rows": suggestion.n_rows,
            }
        )
    return suggestion
