"""History / catalog / walkthrough helpers for semi-supervised operations."""

from __future__ import annotations

from typing import Any


def fit_result_summary(fit_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``fit_semisupervised`` history.

Strips heavy model objects so Session history retains only fields needed for walkthrough overlays and audit replay.

Parameters
----------
fit_result:
    Optional fit summary to embed in bundle metadata or history.

Returns
-------
dict[str, Any]
    JSON-friendly mapping for history, bundles, or walkthrough overlays.
    """
    if fit_result is None:
        return {}
    payload = fit_result.to_dict() if hasattr(fit_result, "to_dict") else dict(fit_result)
    return {
        "method": payload.get("method"),
        "backend": payload.get("backend"),
        "modality": payload.get("modality"),
        "n_train_rows": payload.get("n_train_rows"),
        "n_labeled_train": payload.get("n_labeled_train"),
        "n_unlabeled_train": payload.get("n_unlabeled_train"),
        "target_column": payload.get("target_column"),
        "used_reduce_components": payload.get("used_reduce_components"),
    }


def predict_result_summary(predict_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``predict_semisupervised`` history.

Strips heavy model objects so Session history retains only fields needed for walkthrough overlays and audit replay.

Parameters
----------
predict_result:
    predict result (Any).

Returns
-------
dict[str, Any]
    JSON-friendly mapping for history, bundles, or walkthrough overlays.
    """
    if predict_result is None:
        return {}
    payload = (
        predict_result.to_dict()
        if hasattr(predict_result, "to_dict")
        else dict(predict_result)
    )
    return {
        "partition": payload.get("partition"),
        "n_rows": payload.get("n_rows"),
        "method": payload.get("method"),
        "attached": payload.get("attached"),
    }


def eval_result_summary(eval_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``evaluate_semisupervised`` history.

Strips heavy model objects so Session history retains only fields needed for walkthrough overlays and audit replay.

Parameters
----------
eval_result:
    Optional evaluation summary for bundle metadata or history.

Returns
-------
dict[str, Any]
    JSON-friendly mapping for history, bundles, or walkthrough overlays.
    """
    if eval_result is None:
        return {}
    payload = eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
    return {
        "partition": payload.get("partition"),
        "method": payload.get("method"),
        "n_rows": payload.get("n_rows"),
        "n_labeled_eval": payload.get("n_labeled_eval"),
        "n_unlabeled_eval": payload.get("n_unlabeled_eval"),
        "metrics": payload.get("metrics"),
    }


def semisupervised_status(
    plan: Any = None,
    *,
    fit_result: Any = None,
    eval_result: Any = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Factual walkthrough disclosure for semi-supervised learning.

Combines live plan fields, latest operation results, and history evidence into a teaching-oriented status dict with capability attachment.

Parameters
----------
plan:
    Fitted plan object carrying model state and feature contract.
fit_result:
    Optional fit summary to embed in bundle metadata or history.
eval_result:
    Optional evaluation summary for bundle metadata or history.
history:
    Session operation history for detecting prior activity.

Returns
-------
dict[str, Any]
    JSON-friendly mapping for history, bundles, or walkthrough overlays.
    """
    from buildml.semisupervised.catalog import semisupervised_capability_matrix

    records = list(history or [])
    saw = any(
        str(r.get("operation_id") or r.get("action"))
        in {
            "fit_semisupervised",
            "predict_semisupervised",
            "evaluate_semisupervised",
            "save_semisupervised_bundle",
            "load_semisupervised_bundle",
        }
        for r in records
    )
    enabled = plan is not None
    disclosures: list[str] = []
    if enabled:
        disclosures.extend(
            [
                f"SemiSupervisedPlan method={getattr(plan, 'method', None)}, "
                f"backend={getattr(plan, 'backend', 'sklearn')}, "
                f"modality={getattr(plan, 'modality', 'tabular')}, "
                f"n_labeled_train={getattr(plan, 'n_labeled_train', None)}, "
                f"n_unlabeled_train={getattr(plan, 'n_unlabeled_train', None)}.",
                "Unlabeled targets use NaN missingness by default (sklearn -1 internally).",
                "Session checkpoints do not embed SemiSupervisedPlan; use "
                "save_semisupervised_bundle / load_semisupervised_bundle.",
                "Distinct from anomaly novelty (normal-only) and self-supervised pretext.",
            ]
        )
        for note in getattr(plan, "disclosures", ()) or ():
            disclosures.append(str(note))
    elif saw:
        disclosures.append(
            "Semi-supervised operations appear in history, but no live "
            "SemiSupervisedPlan is attached."
        )

    eval_payload = None
    if eval_result is not None:
        eval_payload = (
            eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
        )
        disclosures.append(
            "Last semi-supervised eval: "
            f"partition={eval_payload.get('partition')}, "
            f"n_labeled_eval={eval_payload.get('n_labeled_eval')}, "
            f"metrics={eval_payload.get('metrics')}."
        )

    return {
        "enabled": enabled,
        "present": enabled or saw,
        "has_semisupervised_plan": enabled,
        "method": None if plan is None else getattr(plan, "method", None),
        "backend": None if plan is None else getattr(plan, "backend", "sklearn"),
        "modality": None if plan is None else getattr(plan, "modality", "tabular"),
        "capability_matrix": semisupervised_capability_matrix(),
        "n_labeled_train": None if plan is None else getattr(plan, "n_labeled_train", None),
        "n_unlabeled_train": (
            None if plan is None else getattr(plan, "n_unlabeled_train", None)
        ),
        "used_reduce_components": (
            None if plan is None else getattr(plan, "used_reduce_components", None)
        ),
        "has_fit_result": fit_result is not None,
        "has_eval_result": eval_result is not None,
        "eval": eval_payload,
        "disclosures": disclosures,
        "boundary": (
            "Semi-supervised learning uses scarce labeled + abundant unlabeled "
            "train rows. Holdout labels are evaluation-only; unlabeled holdout "
            "rows never invent selection labels."
        ),
    }


def semisupervised_status_for_session(session: Any) -> dict[str, Any]:
    """Build semisupervised walkthrough status from a Session instance.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
session:
    BuildML Session with optional private state attributes.

Returns
-------
dict[str, Any]
    JSON-friendly mapping for history, bundles, or walkthrough overlays.
    """
    return semisupervised_status(
        getattr(session, "_semisupervised_plan", None),
        fit_result=getattr(session, "_semisupervised_fit_result", None),
        eval_result=getattr(session, "_semisupervised_eval_result", None),
        history=list(getattr(session, "_history", ()) or ()),
    )
