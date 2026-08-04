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
from buildml.session.facade_registry import DOMAIN_FACADES, flat_to_facade, preferred_path
from buildml.session.facades import list_facades as list_facades

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

    Aggregates existing ``*_capability_matrix`` operations, fit→matrix routes,
    facade catalog metadata, and optional live matrix payloads so callers can
    browse the large Session surface without deleting flat methods.

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
        Grouped domain entries (with preferred facade + stability tier), matrix
        operation names, fit→matrix routing, facades catalog, and disclosures.
    """
    status_by_matrix = {matrix: status for status, matrix in DOMAIN_STATUS_CAPABILITY_OPS}
    domains: list[dict[str, Any]] = []
    for matrix_op in sorted(CAPABILITY_MATRIX_OPERATIONS):
        domain_key = _MATRIX_TO_DOMAIN.get(
            matrix_op, matrix_op.replace("_capability_matrix", "")
        )
        if domain is not None and domain_key != domain:
            continue
        facade_attr = _facade_attr_for_domain(domain_key)
        facade_spec = DOMAIN_FACADES.get(facade_attr) if facade_attr else None
        entry: dict[str, Any] = {
            "domain": domain_key,
            "capability_matrix_operation": matrix_op,
            "session_call": f"Session.{matrix_op}()",
            "preferred_facade": (
                f"session.{facade_attr}.capability_matrix"
                if facade_attr and facade_spec and "capability_matrix" in facade_spec["bindings"]
                else (f"session.{facade_attr}" if facade_attr else None)
            ),
            "stability_tier": (
                facade_spec["tier"] if facade_spec else _DOMAIN_MATURITY_HINTS.get(domain_key)
            ),
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
        "facades": list_facades(),
        "disclosures": (
            "Additive discovery API over existing Session.*_capability_matrix() "
            "and explain catalog — does not reduce the public surface.",
            "include_matrices=True may import optional industry stacks.",
            "Prefer session.<domain>.* facades; flat domain actions are deprecated "
            "until BuildML 3.0 (see docs/session-facade-migration.md).",
        ),
    }


def describe_method(name: str, session_type: type | None = None) -> dict[str, Any]:
    """Describe one Session method via catalog, capability routing, or docstring.

    Accepts flat names (``evaluate_fairness``) or facade-style names
    (``fairness.evaluate`` / ``session.fairness.evaluate``) and always reports
    the preferred facade path plus whether the flat alias is deprecated.

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
        Summary, domain tags, preferred_path, stability_tier, related capability
        matrix, and teaching pointers.

    Raises
    ------
    ValidationError
        When ``name`` is empty or the method cannot be resolved on Session /
        the explain catalog / capability routing tables.
    """
    if not name or not str(name).strip():
        raise ValidationError("describe_method requires a non-empty operation name.")
    from buildml.session.facade_registry import resolve_operation_name

    requested = str(name).strip()
    preferred_from_facade, facade_meta_early = _resolve_facade_style_name(requested)
    facade_meta: dict[str, Any] | None
    if facade_meta_early is not None:
        attr = facade_meta_early["facade_attr"]
        facade_method = facade_meta_early["facade_method"]
        op_name = DOMAIN_FACADES[attr]["bindings"][facade_method]
        preferred = preferred_from_facade
        facade_meta = facade_meta_early
    else:
        op_name = resolve_operation_name(requested)
        facade_meta = flat_to_facade().get(op_name)
        preferred = preferred_path(op_name)

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
                f"Unknown Session method {requested!r}. "
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

    stability_tier = None
    if facade_meta is not None:
        stability_tier = facade_meta["tier"]
        domain = domain or facade_meta["mixin_key"]
    elif domain is not None:
        facade_attr = _facade_attr_for_domain(domain)
        if facade_attr and facade_attr in DOMAIN_FACADES:
            stability_tier = DOMAIN_FACADES[facade_attr]["tier"]

    payload: dict[str, Any] = {
        "name": op_name,
        "requested": requested,
        "summary": str(summary),
        "domain": domain,
        "capability_matrix_operation": matrix_op,
        "session_call": (
            f"Session.{op_name}()"
            if op_name.endswith("_capability_matrix")
            else f"session.{op_name}(...)"
        ),
        "preferred_path": preferred,
        "stability_tier": stability_tier,
        "flat_deprecated": bool(
            facade_meta["warn_flat"] if facade_meta is not None else False
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

    This is a presence probe over plan/result attributes and recent history —
    not a maturity score and not a substitute for capability matrices.

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


def _facade_attr_for_domain(domain_key: str) -> str | None:
    """Map capability domain key → facade attribute name."""
    aliases = {
        "activelearning": "active_learning",
        "forecasting": "forecast",
        "optimize": "decision",
        "recommenders": "recommender",
        "selfsupervised": "ssl",
        "ssl": "ssl",
        "eda": "explore",
        "workflow": "audit",
    }
    if domain_key in aliases:
        return aliases[domain_key]
    if domain_key in DOMAIN_FACADES:
        return domain_key
    for attr, spec in DOMAIN_FACADES.items():
        if spec["mixin_key"] == domain_key:
            return attr
    return None


def _resolve_facade_style_name(
    name: str,
) -> tuple[str | None, dict[str, Any] | None]:
    """Resolve ``fairness.evaluate`` or ``session.fairness.evaluate`` style names."""
    cleaned = name.strip()
    if cleaned.startswith("session."):
        cleaned = cleaned[len("session.") :]
    if "." not in cleaned:
        return None, None
    attr, method = cleaned.split(".", 1)
    spec = DOMAIN_FACADES.get(attr)
    if spec is None or method not in spec["bindings"]:
        return None, None
    flat = spec["bindings"][method]
    meta = flat_to_facade().get(flat)
    return f"session.{attr}.{method}", meta


__all__ = [
    "describe_method",
    "list_active_domains",
    "list_capabilities",
    "list_facades",
]
