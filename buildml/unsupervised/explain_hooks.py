"""History / catalog / walkthrough helpers for unsupervised operations."""

from __future__ import annotations

from typing import Any


def fit_result_summary(fit_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``fit_clusters`` history."""
    if fit_result is None:
        return {}
    if hasattr(fit_result, "to_dict"):
        payload = fit_result.to_dict()
    else:
        payload = dict(fit_result)
    return {
        "method": payload.get("method"),
        "n_clusters": payload.get("n_clusters"),
        "n_train_rows": payload.get("n_train_rows"),
        "assign_strategy": payload.get("assign_strategy"),
        "used_reduce_components": payload.get("used_reduce_components"),
        "inertia": payload.get("inertia"),
        "cluster_sizes": payload.get("cluster_sizes"),
    }


def assign_result_summary(assign_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``assign_clusters`` history."""
    if assign_result is None:
        return {}
    if hasattr(assign_result, "to_dict"):
        payload = assign_result.to_dict()
    else:
        payload = dict(assign_result)
    return {
        "partition": payload.get("partition"),
        "n_rows": payload.get("n_rows"),
        "method": payload.get("method"),
        "assign_strategy": payload.get("assign_strategy"),
        "attached": payload.get("attached"),
        "n_noise": payload.get("n_noise"),
        "n_unique_labels": payload.get("n_unique_labels"),
    }


def eval_result_summary(eval_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``evaluate_clusters`` history."""
    if eval_result is None:
        return {}
    if hasattr(eval_result, "to_dict"):
        payload = eval_result.to_dict()
    else:
        payload = dict(eval_result)
    return {
        "partition": payload.get("partition"),
        "method": payload.get("method"),
        "n_rows": payload.get("n_rows"),
        "n_clusters_observed": payload.get("n_clusters_observed"),
        "metrics": payload.get("metrics"),
        "external_metrics": payload.get("external_metrics"),
    }


def unsupervised_status(
    plan: Any = None,
    *,
    fit_result: Any = None,
    eval_result: Any = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Factual walkthrough disclosure for unsupervised clustering."""
    records = list(history or [])
    saw = any(
        str(r.get("operation_id") or r.get("action"))
        in {
            "fit_clusters",
            "assign_clusters",
            "evaluate_clusters",
            "save_unsupervised_bundle",
            "load_unsupervised_bundle",
        }
        for r in records
    )
    enabled = plan is not None
    disclosures: list[str] = []
    if enabled:
        disclosures.extend(
            [
                f"ClusterPlan method={getattr(plan, 'method', None)}, "
                f"n_clusters={getattr(plan, 'n_clusters', None)}, "
                f"assign_strategy={getattr(plan, 'assign_strategy', None)}.",
                "Session checkpoints do not embed ClusterPlan; use "
                "save_unsupervised_bundle / load_unsupervised_bundle.",
                "Dimensionality reduction stays on Session.reduce_dimensions (PCA); "
                "clustering may consume those components without refitting PCA.",
            ]
        )
        if getattr(plan, "used_reduce_components", False):
            disclosures.append("Clusters were fit on reduce_dimensions component columns.")
        for note in getattr(plan, "disclosures", ()) or ():
            disclosures.append(str(note))
    elif saw:
        disclosures.append(
            "Unsupervised operations appear in history, but no live ClusterPlan is attached."
        )

    eval_payload = None
    if eval_result is not None:
        eval_payload = (
            eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
        )
        disclosures.append(
            "Last cluster eval: "
            f"partition={eval_payload.get('partition')}, "
            f"metrics={eval_payload.get('metrics')}. "
            "Internal metrics are geometric validity, not ground truth."
        )

    return {
        "enabled": enabled,
        "present": enabled or saw,
        "has_cluster_plan": enabled,
        "method": None if plan is None else getattr(plan, "method", None),
        "n_clusters": None if plan is None else getattr(plan, "n_clusters", None),
        "assign_strategy": None if plan is None else getattr(plan, "assign_strategy", None),
        "used_reduce_components": (
            None if plan is None else getattr(plan, "used_reduce_components", None)
        ),
        "has_fit_result": fit_result is not None,
        "has_eval_result": eval_result is not None,
        "eval": eval_payload,
        "disclosures": disclosures,
        "boundary": (
            "Unsupervised clustering is a Session domain path distinct from EDA "
            "IsolationForest screens and from supervised Session.fit."
        ),
    }


def unsupervised_status_for_session(session: Any) -> dict[str, Any]:
    """Session-facing status helper."""
    return unsupervised_status(
        getattr(session, "_cluster_plan", None),
        fit_result=getattr(session, "_cluster_fit_result", None),
        eval_result=getattr(session, "_cluster_eval_result", None),
        history=list(getattr(session, "_history", ()) or ()),
    )
