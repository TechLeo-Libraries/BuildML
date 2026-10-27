"""Session orchestration for fairness disparity reporting."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from buildml.core.errors import ValidationError
from buildml.fairness.catalog import fairness_capability_matrix as _fairness_capability_matrix
from buildml.fairness.evaluate import evaluate_fairness as _evaluate_fairness
from buildml.fairness.groups import extract_sensitive_keys, normalize_sensitive_columns
from buildml.fairness.metrics import _label_vector
from buildml.fairness.mitigation import (
    GroupThresholdSuggestion,
    ReweighingSuggestion,
)
from buildml.fairness.mitigation import (
    suggest_group_thresholds as _suggest_group_thresholds,
)
from buildml.fairness.mitigation import (
    suggest_reweighing_weights as _suggest_reweighing_weights,
)
from buildml.fairness.results import FairnessReport
from buildml.fairness.stability import StabilityMethod, _validate_stability_options


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


def _classifier_positive_label(session: Any, positive_label: Any, operation: str) -> Any:
    """Resolve numeric-equivalent labels against the fitted classifier's classes."""
    fit = getattr(session, "_fit_result", None)
    if fit is None or fit.task != "classification":
        raise ValidationError(
            f"{operation} requires a fitted classifier; call Session.fit "
            "with task='classification' first."
        )
    classes = getattr(fit.estimator, "classes_", None)
    if classes is None:
        raise ValidationError(
            f"{operation} requires fitted classifier classes_ to identify positive_label."
        )
    if np.ndim(positive_label) == 0:
        for label in classes:
            try:
                if bool(label == positive_label):
                    return label
            except (TypeError, ValueError):
                continue
    raise ValidationError(
        f"positive_label={positive_label!r} is not a fitted class. "
        f"Pass an actual positive class value from {list(classes)!r}."
    )


def _positive_scores(session: Any, partition: str, positive_label: Any) -> Any | None:
    """Best-effort scores for an already resolved fitted class."""
    fit = session._fit_result
    if not callable(getattr(fit.estimator, "predict_proba", None)):
        # Classical predict falls back to labels when probabilities are absent.
        # Those labels cannot serve as probability scores for AUC or thresholds.
        return None
    try:
        proba = session.predict(partition=partition, return_proba=True)  # type: ignore[misc]
    except Exception:  # noqa: BLE001 - scores are optional for AUC bridge
        return None
    if proba is None:
        return None
    if hasattr(proba, "columns"):
        cols = list(proba.columns)
        score_column = f"proba_{positive_label}"
        if score_column in cols:
            return proba[score_column].to_numpy()
    # Classical prediction names every probability column from classes_. A
    # missing class column is unusable; another column can reverse the outcome.
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
        When the Session lacks a classification fit with fitted classes, a
        split/dataset, or sensitive columns; when the positive label is unknown,
        the partition is empty/unknown, or prediction length mismatches.
    """
    fitted_positive = _classifier_positive_label(session, positive_label, "evaluate_fairness")
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
    bootstrap_samples = _validate_stability_options(
        bootstrap_samples, confidence_level, stability_method, subsample_fraction, allow_zero=True
    )
    frame = dataset._ensure_pandas().iloc[idx]
    target = dataset.require_target()
    y_true = _label_vector(frame[target].to_numpy(), "y_true")
    sensitive = extract_sensitive_keys(frame, cols)
    y_hat = session.predict(partition=partition)  # type: ignore[misc]
    if hasattr(y_hat, "to_numpy"):
        y_hat_arr = y_hat.to_numpy()
    else:
        y_hat_arr = list(y_hat)

    if len(y_hat_arr) != len(y_true):
        raise ValidationError(
            "Prediction length does not match fairness partition rows."
        )

    y_score = None
    if include_scores:
        y_score = _positive_scores(session, partition, fitted_positive)

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
    session._record(
        "evaluate_fairness",
        {"partition": partition, "sensitive_column": list(cols), "positive_label": positive_label,
         "bootstrap_samples": bootstrap_samples, "stability_method": stability_method,
         "confidence_level": confidence_level, "subsample_fraction": subsample_fraction,
         "random_state": random_state, "include_scores": include_scores,
         "include_classical_metrics": include_classical_metrics},
        warnings=report.warnings,
        result_summary={"n_rows": report.n_rows, "n_groups": len(report.groups),
                        "intersectional": report.intersectional,
                        "demographic_parity_difference": report.demographic_parity_difference,
                        "stability_enabled": report.stability is not None},
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
    Requires a classification fit and a positive label matching its fitted
    classes. Numeric-equivalent labels select the same probability column.
    """
    fitted_positive = _classifier_positive_label(session, positive_label, "suggest_thresholds")
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
    frame = dataset._ensure_pandas().iloc[idx]
    target_col = dataset.require_target()
    y_true = frame[target_col].to_numpy()
    sensitive = extract_sensitive_keys(frame, cols)
    y_score = _positive_scores(session, partition, fitted_positive)
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
    session._record(
        "suggest_fairness_thresholds",
        {"partition": partition, "sensitive_column": list(cols), "positive_label": positive_label,
         "target": target, "grid_size": grid_size},
        warnings=suggestion.warnings,
        result_summary={"n_rows": suggestion.n_rows, "target": suggestion.target,
                        "thresholds_by_group": suggestion.thresholds_by_group},
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
    frame = dataset._ensure_pandas().iloc[idx]
    target_col = dataset.require_target()
    y_true = frame[target_col].to_numpy()
    sensitive = extract_sensitive_keys(frame, cols)
    suggestion = _suggest_reweighing_weights(
        y_true, sensitive, positive_label=positive_label
    )
    session._fairness_mitigation_suggestion = suggestion
    session._record(
        "suggest_fairness_reweighing",
        {"partition": partition, "sensitive_column": list(cols), "positive_label": positive_label},
        warnings=suggestion.warnings,
        result_summary={"n_rows": suggestion.n_rows, "weight_table": suggestion.weight_table},
    )
    return suggestion
