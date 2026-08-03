"""History / catalog / walkthrough helpers for AutoML operations."""

from __future__ import annotations

from typing import Any

def fit_result_summary(result: Any) -> dict[str, Any]:
    """Build a compact history payload from an AutoML search result.

    Strips heavy estimator objects so Session history records only the fields
    needed for walkthrough overlays and audit replay.

    Parameters
    ----------
    result:
        :class:`~buildml.automl.results.AutoMLResult` or compatible mapping;
        ``None`` yields an empty dict.

    Returns
    -------
    dict[str, Any]
        Backend, method, selection, ranking metric, and best trial summary.
    """
    if result is None:
        return {}
    if hasattr(result, "to_dict"):
        payload = result.to_dict()
    else:
        payload = dict(result)
    return {
        "method": payload.get("method"),
        "backend": (payload.get("config") or {}).get("backend", "native"),
        "selection": payload.get("selection"),
        "task": payload.get("task"),
        "ranking_metric": payload.get("ranking_metric"),
        "best_family": payload.get("best_family"),
        "best_recipe_strategy": payload.get("best_recipe_strategy"),
        "best_kind": payload.get("best_kind"),
        "best_score": payload.get("best_score"),
        "n_trials": len(payload.get("trials") or []),
        "families_searched": payload.get("families_searched"),
        "outer_score_mean": payload.get("outer_score_mean"),
    }

def automl_status(
    plan: Any = None,
    *,
    result: Any = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build factual walkthrough disclosure for AutoML state.

    Combines live plan fields, latest search result payloads, and history
    evidence into a teaching-oriented status dict with capability matrix
    attachment.

    Parameters
    ----------
    plan:
        Optional :class:`~buildml.automl.results.AutoMLPlan`.
    result:
        Optional latest :class:`~buildml.automl.results.AutoMLResult` for
        ``has_result``.
    history:
        Session operation history used to detect AutoML activity when no plan
        is attached.

    Returns
    -------
    dict[str, Any]
        Enabled flags, best family/recipe summary, disclosures, boundary text,
        and nested capability matrix from
        :func:`buildml.automl.catalog.automl_capability_matrix`.
    """
    from buildml.automl.catalog import automl_capability_matrix

    records = list(history or [])
    saw = any(
        str(r.get("operation_id") or r.get("action"))
        in {
            "run_automl",
            "evaluate_automl",
            "save_automl_bundle",
            "load_automl_bundle",
        }
        for r in records
    )
    enabled = plan is not None
    disclosures: list[str] = []
    if enabled:
        disclosures.extend(
            [
                (
                    f"AutoMLPlan method={getattr(plan, 'method', None)}, "
                    f"selection={getattr(plan, 'selection', None)}, "
                    f"best={getattr(plan, 'best_family', None)}/"
                    f"{getattr(plan, 'best_recipe_strategy', None)}."
                ),
                (
                    "AutoML is finite catalog search (families + fold-local recipes), "
                    "not NAS and not causal discovery."
                ),
                (
                    "Session checkpoints do not embed AutoMLPlan; use "
                    "save_automl_bundle / load_automl_bundle (or save_pipeline)."
                ),
                "Session test never enters AutoML selection scoring.",
            ]
        )
        for note in getattr(plan, "disclosures", ()) or ():
            disclosures.append(str(note))
    elif saw:
        disclosures.append(
            "AutoML operations appear in history, but no live AutoMLPlan is attached."
        )

    return {
        "enabled": enabled,
        "present": enabled or saw,
        "has_automl_plan": enabled,
        "backend": None if plan is None else (getattr(plan, "config", {}) or {}).get("backend", "native"),
        "method": None if plan is None else getattr(plan, "method", None),
        "selection": None if plan is None else getattr(plan, "selection", None),
        "task": None if plan is None else getattr(plan, "task", None),
        "best_family": None if plan is None else getattr(plan, "best_family", None),
        "best_recipe_strategy": (
            None if plan is None else getattr(plan, "best_recipe_strategy", None)
        ),
        "best_score": None if plan is None else getattr(plan, "best_score", None),
        "has_result": result is not None,
        "disclosures": disclosures,
        "capability_matrix": automl_capability_matrix(),
        "boundary": (
            "AutoML is a Session domain path distinct from single-estimator "
            "grid_search / optuna_search on one fixed model."
        ),
    }

def automl_status_for_session(session: Any) -> dict[str, Any]:
    """Build AutoML walkthrough status from a Session instance.

    Reads private Session attributes set by AutoML operations and delegates to
    :func:`automl_status`.

    Parameters
    ----------
    session:
        BuildML Session with optional ``_automl_plan`` and ``_automl_result``
        state attributes.

    Returns
    -------
    dict[str, Any]
        Same payload as :func:`automl_status` for the session's plan and result.
    """
    return automl_status(
        getattr(session, "_automl_plan", None),
        result=getattr(session, "_automl_result", None),
        history=list(getattr(session, "_history", ()) or ()),
    )
