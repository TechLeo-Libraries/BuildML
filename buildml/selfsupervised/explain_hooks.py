"""History / catalog / walkthrough helpers for self-supervised operations."""

from __future__ import annotations

from typing import Any


def fit_result_summary(fit_result: Any) -> dict[str, Any]:
    """Build a compact history summary from an SSL pretext fit result.

    Strips full encoder payloads while recording method, modality, and train
    diagnostics for Session audit logs.

    Parameters
    ----------
    fit_result:
        :class:`~buildml.selfsupervised.results.SelfSupervisedFitResult` or ``None``.

    Returns
    -------
    dict[str, Any]
        Method, modality, train row count, latent width, and loss summaries.
    """
    if fit_result is None:
        return {}
    payload = fit_result.to_dict() if hasattr(fit_result, "to_dict") else dict(fit_result)
    return {
        "method": payload.get("method"),
        "modality": payload.get("modality"),
        "n_train_rows": payload.get("n_train_rows"),
        "latent_dim": payload.get("latent_dim"),
        "reconstruction_mae": payload.get("reconstruction_mae"),
        "pretext_loss": payload.get("pretext_loss"),
        "used_reduce_components": payload.get("used_reduce_components"),
    }


def transform_result_summary(result: Any) -> dict[str, Any]:
    """Build a compact history summary from an SSL transform result.

    Records partition, row count, and representation column names without
    embedding full embedding arrays in Session history.

    Parameters
    ----------
    result:
        :class:`~buildml.selfsupervised.results.SelfSupervisedTransformResult` or ``None``.

    Returns
    -------
    dict[str, Any]
        Partition, row count, method, attach flag, and column count.
    """
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
    """Build a compact history summary from an SSL head fit result.

    Records labeled/unlabeled train counts and estimator choice without
    serialising the fitted classifier.

    Parameters
    ----------
    result:
        :class:`~buildml.selfsupervised.results.SSLHeadFitResult` or ``None``.

    Returns
    -------
    dict[str, Any]
        Estimator name, labeled train count, skipped unlabeled count, and target.
    """
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
    """Build a compact history summary from an SSL evaluation result.

    Records partition metrics and labeled/unlabeled mix without listing every
    prediction in Session history.

    Parameters
    ----------
    result:
        :class:`~buildml.selfsupervised.results.SelfSupervisedEvalResult` or ``None``.

    Returns
    -------
    dict[str, Any]
        Partition, row counts, labeled eval count, and metrics dict.
    """
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
    """Build factual walkthrough disclosure for self-supervised hooks.

    Combines live plan metadata, optional head/eval summaries, history
    detection, and :func:`~buildml.selfsupervised.torch.catalog.ssl_capability_matrix`
    for teaching overlays.

    Parameters
    ----------
    plan:
        Active :class:`~buildml.selfsupervised.results.SelfSupervisedPlan`, if any.
    head_plan:
        Optional :class:`~buildml.selfsupervised.results.SSLHeadPlan`.
    fit_result:
        Last pretext fit report attached to the Session.
    eval_result:
        Last holdout evaluation report attached to the Session.
    history:
        Session operation history used to detect prior SSL calls.

    Returns
    -------
    dict[str, Any]
        Enabled flags, method metadata, eval payload, disclosures, and capability matrix.
    """
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
                f"modality={getattr(plan, 'modality', 'tabular')}, "
                f"latent_dim={getattr(plan, 'latent_dim', None)}, "
                f"pretext_loss={getattr(plan, 'pretext_loss_', None)}, "
                f"reconstruction_mae={getattr(plan, 'reconstruction_mae_', None)}.",
                "Story: unlabeled(+labeled) train pretext → representation export → "
                "supervised/semi-supervised head on labeled train.",
                "Session checkpoints do not embed SelfSupervisedPlan; use "
                "save_ssl_bundle / load_ssl_bundle (buildml.ssl_bundle.v2).",
                "Torch tabular defaults: simclr_tabular, byol_tabular, vicreg_tabular. "
                "Legacy masked_tabular (sklearn) is deprecated.",
                "Vision/audio/speech transfer also: load_pretrained_backbone / attach_backbone_head.",
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

    from buildml.explain.capability_status import attach_capability_matrix

    return attach_capability_matrix(
        {
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
            "Self-supervised hooks learn train-only representations via Torch "
            "(SimCLR/BYOL/VICReg/MAE/VAE) or deprecated sklearn masked_tabular, "
            "then attach a supervised head. Distinct from semi-supervised label "
            "propagation and from Torch zoo backbone transfer."
        ),
    },
        "ssl_capability_matrix",
    )


def selfsupervised_status_for_session(session: Any) -> dict[str, Any]:
    """Report self-supervised status for a Session walkthrough panel.

    Reads SSL plan, head, and result slots without mutating the Session.

    Parameters
    ----------
    session:
        :class:`~buildml.session.session.Session` instance.

    Returns
    -------
    dict[str, Any]
        Same payload as :func:`selfsupervised_status` for the Session's SSL state.
    """
    return selfsupervised_status(
        getattr(session, "_ssl_plan", None),
        head_plan=getattr(session, "_ssl_head_plan", None),
        fit_result=getattr(session, "_ssl_fit_result", None),
        eval_result=getattr(session, "_ssl_eval_result", None),
        history=list(getattr(session, "_history", ()) or ()),
    )
