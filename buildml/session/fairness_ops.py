"""Session orchestration for fairness disparity reporting."""

from __future__ import annotations

from typing import Any

from buildml.core.errors import ValidationError
from buildml.fairness.catalog import fairness_capability_matrix as _fairness_capability_matrix
from buildml.fairness.evaluate import evaluate_fairness as _evaluate_fairness
from buildml.fairness.results import FairnessReport


def fairness_capability_matrix() -> dict[str, Any]:
    """Return the fairness capability matrix."""
    return _fairness_capability_matrix()


def evaluate_fairness_op(
    session: Any,
    *,
    sensitive_column: str,
    partition: str = "test",
    positive_label: Any = 1,
) -> FairnessReport:
    """Score group disparity on a holdout partition using the fitted classifier.

    Parameters
    ----------
    session:
        Fitted Session with roles, split, and a classical classifier.
    sensitive_column:
        Column holding caller-declared group ids (any role).
    partition:
        Holdout partition name (default ``test``).
    positive_label:
        Positive class label for rates and equalized odds.

    Returns
    -------
    FairnessReport
        Observational gaps and disclosures; stored on ``session._fairness_report``.
    """
    if getattr(session, "_fit_result", None) is None:
        raise ValidationError(
            "evaluate_fairness requires a fitted classifier; call Session.fit first."
        )
    dataset = session.dataset
    if dataset is None:
        raise ValidationError("Session has no dataset.")
    if sensitive_column not in dataset.frame.columns:
        raise ValidationError(
            f"sensitive_column={sensitive_column!r} is not in the Session dataset."
        )
    if session.split_plan is None:
        raise ValidationError("evaluate_fairness requires a split plan.")

    y_hat = session.predict(partition=partition)  # type: ignore[misc]
    if hasattr(y_hat, "to_numpy"):
        y_hat_arr = y_hat.to_numpy()
    else:
        y_hat_arr = list(y_hat)

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

    frame = dataset.frame.iloc[list(idx)]
    target = dataset.require_target()
    y_true = frame[target].to_numpy()
    sensitive = frame[sensitive_column].to_numpy()
    if len(y_hat_arr) != len(y_true):
        raise ValidationError(
            "Prediction length does not match fairness partition rows."
        )

    report = _evaluate_fairness(
        y_true,
        y_hat_arr,
        sensitive,
        positive_label=positive_label,
        partition=partition,
        sensitive_column=sensitive_column,
    )
    session._fairness_report = report
    history = getattr(session, "history", None)
    if isinstance(history, list):
        history.append(
            {
                "operation": "evaluate_fairness",
                "partition": partition,
                "sensitive_column": sensitive_column,
                "n_rows": report.n_rows,
                "demographic_parity_difference": report.demographic_parity_difference,
            }
        )
    return report
