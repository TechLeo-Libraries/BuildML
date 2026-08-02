"""History / catalog / walkthrough helpers for self-supervised operations."""

from __future__ import annotations

from typing import Any


def fit_result_summary(fit_result: Any) -> dict[str, Any]:
    if fit_result is None:
        return {}
    payload = fit_result.to_dict() if hasattr(fit_result, "to_dict") else dict(fit_result)
    return {
        "method": payload.get("method"),
        "n_train_rows": payload.get("n_train_rows"),
        "latent_dim": payload.get("latent_dim"),
        "reconstruction_mae": payload.get("reconstruction_mae"),
        "used_reduce_components": payload.get("used_reduce_components"),
    }


def transform_result_summary(result: Any) -> dict[str, Any]:
    if result is None:
        return {}
    payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    return {
        "partition": payload.get("partition"),
        "n_rows": payload.get("n_rows"),
        "method": payload.get("method"),
        "attached": payload.get("attached"),
        "n_representation_columns": len(payload.get("representation_columns") or []),
    }


def head_fit_result_summary(result: Any) -> dict[str, Any]:
    if result is None:
        return {}
    payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    return {
        "estimator_name": payload.get("estimator_name"),
        "n_labeled_train": payload.get("n_labeled_train"),
        "n_unlabeled_skipped": payload.get("n_unlabeled_skipped"),
        "target_column": payload.get("target_column"),
    }


def eval_result_summary(result: Any) -> dict[str, Any]:
    if result is None:
        return {}
    payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    return {
        "partition": payload.get("partition"),
        "n_rows": payload.get("n_rows"),
        "n_labeled_eval": payload.get("n_labeled_eval"),
        "n_unlabeled_eval": payload.get("n_unlabeled_eval"),
        "metrics": payload.get("metrics"),
    }


def selfsupervised_status(
    plan: Any = None,
    *,
    head_plan: Any = None,
    fit_result: Any = None,
    eval_result: Any = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Factual walkthrough disclosure for self-supervised hooks."""
    records = list(history or [])
    saw = any(
        str(r.get("operation_id") or r.get("action"))
        in {
            "fit_ssl_pretext",
            "transform_ssl",
            "finetune_ssl_head",
            "evaluate_ssl",
            "save_ssl_bundle",
            "load_ssl_bundle",
        }
        for r in records
    )
    enabled = plan is not None
    disclosures: list[str] = []
    if enabled:
        disclosures.extend(
            [
                f"SelfSupervisedPlan method={getattr(plan, 'method', None)}, "
                f"latent_dim={getattr(plan, 'latent_dim', None)}, "
                f"reconstruction_mae={getattr(plan, 'reconstruction_mae_', None)}.",
                "Story: unlabeled(+labeled) train pretext → representation export → "
                "supervised/semi-supervised head on labeled train.",
                "Session checkpoints do not embed SelfSupervisedPlan; use "
                "save_ssl_bundle / load_ssl_bundle.",
                "Not BERT-from-scratch. Vision/audio/speech transfer: "
                "load_pretrained_backbone / attach_backbone_head.",
            ]
        )
        for note in getattr(plan, "disclosures", ()) or ():
            disclosures.append(str(note))
    elif saw:
        disclosures.append(
            "Self-supervised operations appear in history, but no live "
            "SelfSupervisedPlan is attached."
        )
    if head_plan is not None:
        disclosures.append(
            f"SSLHeadPlan estimator={getattr(head_plan, 'estimator_name', None)}, "
            f"n_labeled_train={getattr(head_plan, 'n_labeled_train', None)}."
        )

    eval_payload = None
    if eval_result is not None:
        eval_payload = (
            eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
        )

    return {
        "enabled": enabled,
        "present": enabled or saw,
        "has_ssl_plan": enabled,
        "has_ssl_head": head_plan is not None,
        "method": None if plan is None else getattr(plan, "method", None),
        "latent_dim": None if plan is None else getattr(plan, "latent_dim", None),
        "reconstruction_mae": (
            None if plan is None else getattr(plan, "reconstruction_mae_", None)
        ),
        "has_fit_result": fit_result is not None,
        "has_eval_result": eval_result is not None,
        "eval": eval_payload,
        "disclosures": disclosures,
        "boundary": (
            "Self-supervised hooks learn train-only tabular representations via "
            "masked reconstruction, then attach a supervised head. Distinct from "
            "semi-supervised label propagation and from Torch zoo backbone transfer."
        ),
    }


def selfsupervised_status_for_session(session: Any) -> dict[str, Any]:
    return selfsupervised_status(
        getattr(session, "_ssl_plan", None),
        head_plan=getattr(session, "_ssl_head_plan", None),
        fit_result=getattr(session, "_ssl_fit_result", None),
        eval_result=getattr(session, "_ssl_eval_result", None),
        history=list(getattr(session, "_history", ()) or ()),
    )
