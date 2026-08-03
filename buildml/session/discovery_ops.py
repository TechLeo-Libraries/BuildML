"""Additive Session discoverability helpers (capability / method catalogs).

These APIs aggregate existing capability matrices, domain status ops, and the
explain operation catalog. They do **not** remove or rename any Session methods.
"""

from __future__ import annotations

import inspect
from typing import Any

from buildml.core.errors import ValidationError
from buildml.explain.capability_status import (
    CAPABILITY_MATRIX_OPERATIONS,
    DOMAIN_STATUS_CAPABILITY_OPS,
    FIT_TO_CAPABILITY_MATRIX,
    load_capability_matrix,
)

# Matrix op → short domain key for grouped discovery.
_MATRIX_TO_DOMAIN: dict[str, str] = {
    "activelearning_capability_matrix": "activelearning",
    "anomaly_capability_matrix": "anomaly",
    "automl_capability_matrix": "automl",
    "causal_capability_matrix": "causal",
    "cbr_capability_matrix": "cbr",
    "decision_capability_matrix": "optimize",
    "optimize_capability_matrix": "optimize",
    "fairness_capability_matrix": "fairness",
    "federated_capability_matrix": "federated",
    "forecast_capability_matrix": "forecasting",
    "graph_capability_matrix": "graph",
    "kg_capability_matrix": "kg",
    "metalearning_capability_matrix": "metalearning",
    "multitask_capability_matrix": "multitask",
    "nlp_capability_matrix": "nlp",
    "online_capability_matrix": "online",
    "probabilistic_capability_matrix": "probabilistic",
    "rag_capability_matrix": "rag",
    "ranking_capability_matrix": "ranking",
    "recommender_capability_matrix": "recommenders",
    "rl_capability_matrix": "rl",
    "semisupervised_capability_matrix": "semisupervised",
    "ssl_capability_matrix": "selfsupervised",
    "symbolic_capability_matrix": "symbolic",
    "synthetic_capability_matrix": "synthetic",
    "tda_capability_matrix": "tda",
    "timeseries_capability_matrix": "timeseries",
    "unsupervised_capability_matrix": "unsupervised",
    "dl_capability_matrix": "dl",
    "ensemble_capability_matrix": "ensemble",
}

_DOMAIN_MATURITY_HINTS: dict[str, str] = {
    "fairness": "observational_analysis",
    "timeseries": "analysis_only",
    "classical": "core_product",
}


def list_capabilities(
    *,
    include_matrices: bool = False,
    domain: str | None = None,
) -> dict[str, Any]:
    """List Session domain capabilities grouped for discovery.

    Parameters
    ----------
    include_matrices:
        When ``True``, eagerly load each capability matrix payload (may probe
        optional extras). Default lists names / metadata only.
    domain:
        Optional domain filter (e.g. ``fairness``, ``online``).

    Returns
    -------
    dict[str, Any]
        Grouped domain entries, matrix operation names, fit→matrix routing, and
        walkthrough status field map.
    """
    status_by_matrix = {matrix: status for status, matrix in DOMAIN_STATUS_CAPABILITY_OPS}
    domains: list[dict[str, Any]] = []
    for matrix_op in sorted(CAPABILITY_MATRIX_OPERATIONS):
        domain_key = _MATRIX_TO_DOMAIN.get(
            matrix_op, matrix_op.replace("_capability_matrix", "")
        )
        if domain is not None and domain_key != domain:
            continue
        entry: dict[str, Any] = {
            "domain": domain_key,
            "capability_matrix_operation": matrix_op,
            "session_call": f"Session.{matrix_op}()",
            "walkthrough_status_field": status_by_matrix.get(matrix_op),
            "maturity_hint": _DOMAIN_MATURITY_HINTS.get(domain_key),
        }
        if include_matrices:
            matrix = load_capability_matrix(matrix_op)
            entry["matrix"] = matrix
            backends = matrix.get("backends") if isinstance(matrix, dict) else None
            if isinstance(backends, dict):
                entry["available_backends"] = sorted(
                    name
                    for name, payload in backends.items()
                    if isinstance(payload, dict) and payload.get("available")
                )
                entry["extra_hints"] = {
                    name: payload.get("extra")
                    for name, payload in backends.items()
                    if isinstance(payload, dict) and payload.get("extra")
                }
        domains.append(entry)

    fit_routes = [
        {"fit_or_eval_operation": op, "capability_matrix_operation": matrix}
        for op, matrix in sorted(FIT_TO_CAPABILITY_MATRIX.items())
        if domain is None or _MATRIX_TO_DOMAIN.get(matrix) == domain
    ]
    return {
        "n_domains": len(domains),
        "domains": domains,
        "capability_matrix_operations": sorted(CAPABILITY_MATRIX_OPERATIONS),
        "fit_to_capability_matrix": fit_routes,
        "disclosures": (
            "Additive discovery API over existing Session.*_capability_matrix() "
            "and explain catalog — does not reduce the public surface.",
            "include_matrices=True may import optional industry stacks.",
        ),
    }


def describe_method(name: str, session_type: type | None = None) -> dict[str, Any]:
    """Describe one Session method via catalog, capability routing, or docstring.

    Parameters
    ----------
    name:
        Public Session operation name (e.g. ``evaluate_fairness``, ``fit``).
    session_type:
        Optional Session class for docstring fallback when the explain catalog
        has no entry yet.

    Returns
    -------
    dict[str, Any]
        Summary, domain tags, related capability matrix, and teaching pointers.

    Raises
    ------
    ValidationError
        When ``name`` is empty or the method cannot be resolved on Session /
        the explain catalog / capability routing tables.
    """
    if not name or not str(name).strip():
        raise ValidationError("describe_method requires a non-empty operation name.")
    op_name = str(name).strip()

    matrix_op = FIT_TO_CAPABILITY_MATRIX.get(op_name)
    if matrix_op is None and op_name in CAPABILITY_MATRIX_OPERATIONS:
        matrix_op = op_name

    catalog_payload: dict[str, Any] | None = None
    try:
        from buildml.explain.catalog import get_operation

        spec = get_operation(op_name)
        catalog_payload = {
            "source": "explain_catalog",
            "summary": str(getattr(spec, "plain_summary", None) or spec.purpose),
            "purpose": spec.purpose,
            "definition": spec.definition,
            "pipeline_role": spec.pipeline_role,
            "assumptions": list(spec.assumptions),
            "leakage_risks": list(spec.leakage_risks),
            "when_to_use": list(spec.when_to_use),
            "when_not_to_use": list(spec.when_not_to_use),
        }
    except KeyError:
        catalog_payload = None

    docstring_summary: str | None = None
    if session_type is not None and hasattr(session_type, op_name):
        method = getattr(session_type, op_name)
        docstring_summary = inspect.getdoc(method)

    if catalog_payload is None and docstring_summary is None and matrix_op is None:
        # Last resort: inspect live Session class.
        from buildml.session.session import Session as _Session

        if hasattr(_Session, op_name):
            docstring_summary = inspect.getdoc(getattr(_Session, op_name))
        else:
            raise ValidationError(
                f"Unknown Session method {op_name!r}. "
                "Use Session.list_capabilities() or session.workflow() to browse."
            )

    domain = None
    if matrix_op is not None:
        domain = _MATRIX_TO_DOMAIN.get(matrix_op)
    summary = (
        (catalog_payload or {}).get("summary")
        or (docstring_summary.splitlines()[0] if docstring_summary else None)
        or op_name
    )

    payload: dict[str, Any] = {
        "name": op_name,
        "summary": str(summary),
        "domain": domain,
        "capability_matrix_operation": matrix_op,
        "session_call": (
            f"Session.{op_name}()"
            if op_name.endswith("_capability_matrix")
            else f"session.{op_name}(...)"
        ),
        "explain": f'session.explain("{op_name}")',
        "learn": f'session.learn("{op_name}")',
        "maturity_hint": _DOMAIN_MATURITY_HINTS.get(str(domain)) if domain else None,
        "catalog": catalog_payload,
        "docstring": docstring_summary,
    }
    if matrix_op is not None:
        payload["capability_matrix_preview"] = {
            "session_call": f"Session.{matrix_op}()",
            "keys": sorted(load_capability_matrix(matrix_op).keys()),
        }
    return payload


def list_active_domains(session: Any) -> dict[str, Any]:
    """Report which domain artifacts are present on a live Session.

    Parameters
    ----------
    session:
        Active Session instance.

    Returns
    -------
    dict[str, Any]
        Active domain keys inferred from private result attributes and history.
    """
    probes: dict[str, str] = {
        "classical": "_fit_result",
        "anomaly": "_anomaly_plan",
        "online": "_online_plan",
        "activelearning": "_activelearning_plan",
        "fairness": "_fairness_report",
        "forecasting": "_forecast_plan",
        "rag": "_rag_plan",
        "ssl": "_ssl_plan",
        "causal": "_causal_plan",
        "ensemble": "_ensemble_plan",
        "automl": "_automl_result",
        "nlp": "_nlp_plan",
        "synthetic": "_synthetic_plan",
    }
    active: list[str] = []
    idle: list[str] = []
    for domain, attr in probes.items():
        if getattr(session, attr, None) is not None:
            active.append(domain)
        else:
            idle.append(domain)

    history = getattr(session, "_history", None) or []
    history_ops: list[str] = []
    if isinstance(history, list):
        for row in history:
            if isinstance(row, dict) and row.get("operation"):
                history_ops.append(str(row["operation"]))

    return {
        "active_domains": active,
        "idle_probed_domains": idle,
        "history_operations": history_ops,
        "disclosures": (
            "Presence of private plan/result attributes — not a maturity score.",
        ),
    }


__all__ = [
    "describe_method",
    "list_active_domains",
    "list_capabilities",
]
