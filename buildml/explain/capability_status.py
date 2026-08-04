"""Capability-matrix payloads for walkthrough, audit, and domain status hooks.

Every domain that publishes ``*_capability_matrix()`` on Session exposes an
honest backend/method availability dict. Walkthrough and audit surfaces attach
those payloads to domain status blocks so operators can discover what will run
*before* calling a fit path.
"""

from __future__ import annotations

import copy
import importlib
from typing import Any

# Process-wide cache: capability matrices are static introspection for a given
# installed environment. Walkthrough used to re-import every domain (and on
# Windows re-spawn torch-heavy subprocess probes) on each call.
_MATRIX_CACHE: dict[str, dict[str, Any]] = {}


def clear_capability_matrix_cache() -> None:
    """Drop cached capability matrices (tests / after optional-extra installs)."""
    _MATRIX_CACHE.clear()


# Session static introspection ops (includes optimize alias of decision).
CAPABILITY_MATRIX_OPERATIONS: frozenset[str] = frozenset(
    {
        "activelearning_capability_matrix",
        "anomaly_capability_matrix",
        "automl_capability_matrix",
        "causal_capability_matrix",
        "cbr_capability_matrix",
        "decision_capability_matrix",
        "fairness_capability_matrix",
        "federated_capability_matrix",
        "forecast_capability_matrix",
        "graph_capability_matrix",
        "kg_capability_matrix",
        "metalearning_capability_matrix",
        "multitask_capability_matrix",
        "nlp_capability_matrix",
        "online_capability_matrix",
        "optimize_capability_matrix",
        "probabilistic_capability_matrix",
        "rag_capability_matrix",
        "ranking_capability_matrix",
        "recommender_capability_matrix",
        "rl_capability_matrix",
        "semisupervised_capability_matrix",
        "ssl_capability_matrix",
        "symbolic_capability_matrix",
        "synthetic_capability_matrix",
        "tda_capability_matrix",
        "timeseries_capability_matrix",
        "unsupervised_capability_matrix",
        "dl_capability_matrix",
        "ensemble_capability_matrix",
    }
)

# Walkthrough ``*_status`` field → Session capability-matrix operation.
DOMAIN_STATUS_CAPABILITY_OPS: tuple[tuple[str, str], ...] = (
    ("unsupervised_status", "unsupervised_capability_matrix"),
    ("ensemble_status", "ensemble_capability_matrix"),
    ("automl_status", "automl_capability_matrix"),
    ("forecasting_status", "forecast_capability_matrix"),
    ("timeseries_status", "timeseries_capability_matrix"),
    ("anomaly_status", "anomaly_capability_matrix"),
    ("semisupervised_status", "semisupervised_capability_matrix"),
    ("selfsupervised_status", "ssl_capability_matrix"),
    ("activelearning_status", "activelearning_capability_matrix"),
    ("online_status", "online_capability_matrix"),
    ("multitask_status", "multitask_capability_matrix"),
    ("metalearning_status", "metalearning_capability_matrix"),
    ("federated_status", "federated_capability_matrix"),
    ("probabilistic_status", "probabilistic_capability_matrix"),
    ("causal_status", "causal_capability_matrix"),
    ("graph_status", "graph_capability_matrix"),
    ("symbolic_status", "symbolic_capability_matrix"),
    ("cbr_status", "cbr_capability_matrix"),
    ("nlp_status", "nlp_capability_matrix"),
    ("imitation_status", "rl_capability_matrix"),
    ("rl_status", "rl_capability_matrix"),
    ("tda_status", "tda_capability_matrix"),
    ("recommender_status", "recommender_capability_matrix"),
    ("ranking_status", "ranking_capability_matrix"),
    ("kg_status", "kg_capability_matrix"),
    ("decision_status", "decision_capability_matrix"),
    ("synthetic_status", "synthetic_capability_matrix"),
    ("rag_status", "rag_capability_matrix"),
    ("fairness_status", "fairness_capability_matrix"),
)

_MATRIX_SOURCES: dict[str, tuple[str, str]] = {
    "activelearning_capability_matrix": (
        "buildml.activelearning.catalog",
        "activelearning_capability_matrix",
    ),
    "anomaly_capability_matrix": ("buildml.anomaly.catalog", "anomaly_capability_matrix"),
    "automl_capability_matrix": ("buildml.automl.catalog", "automl_capability_matrix"),
    "causal_capability_matrix": ("buildml.causal.catalog", "causal_capability_matrix"),
    "cbr_capability_matrix": ("buildml.cbr.catalog", "cbr_capability_matrix"),
    "decision_capability_matrix": ("buildml.optimize.catalog", "decision_capability_matrix"),
    "optimize_capability_matrix": ("buildml.optimize.catalog", "decision_capability_matrix"),
    "federated_capability_matrix": (
        "buildml.federated.catalog",
        "federated_capability_matrix",
    ),
    "forecast_capability_matrix": (
        "buildml.forecasting.catalog",
        "forecast_capability_matrix",
    ),
    "graph_capability_matrix": ("buildml.graph.catalog", "graph_capability_matrix"),
    "kg_capability_matrix": ("buildml.kg.catalog", "kg_capability_matrix"),
    "metalearning_capability_matrix": (
        "buildml.metalearning.catalog",
        "metalearning_capability_matrix",
    ),
    "multitask_capability_matrix": (
        "buildml.multitask.catalog",
        "multitask_capability_matrix",
    ),
    "nlp_capability_matrix": ("buildml.nlp.catalog", "nlp_capability_matrix"),
    "online_capability_matrix": ("buildml.online.catalog", "online_capability_matrix"),
    "probabilistic_capability_matrix": (
        "buildml.probabilistic.catalog",
        "probabilistic_capability_matrix",
    ),
    "rag_capability_matrix": ("buildml.rag.catalog", "rag_capability_matrix"),
    "ranking_capability_matrix": ("buildml.ranking.catalog", "ranking_capability_matrix"),
    "recommender_capability_matrix": (
        "buildml.recommenders.catalog",
        "recommender_capability_matrix",
    ),
    "rl_capability_matrix": ("buildml.rl.catalog", "rl_capability_matrix"),
    "semisupervised_capability_matrix": (
        "buildml.semisupervised.catalog",
        "semisupervised_capability_matrix",
    ),
    "ssl_capability_matrix": (
        "buildml.selfsupervised.torch.catalog",
        "ssl_capability_matrix",
    ),
    "symbolic_capability_matrix": ("buildml.symbolic.catalog", "symbolic_capability_matrix"),
    "synthetic_capability_matrix": ("buildml.synthetic.catalog", "synthetic_capability_matrix"),
    "tda_capability_matrix": ("buildml.tda.catalog", "tda_capability_matrix"),
    "timeseries_capability_matrix": (
        "buildml.timeseries.catalog",
        "timeseries_capability_matrix",
    ),
    "unsupervised_capability_matrix": (
        "buildml.unsupervised.catalog",
        "unsupervised_capability_matrix",
    ),
    "dl_capability_matrix": ("buildml.dl.catalog", "dl_capability_matrix"),
    "ensemble_capability_matrix": (
        "buildml.ensemble.catalog",
        "ensemble_capability_matrix",
    ),
    "fairness_capability_matrix": (
        "buildml.fairness.catalog",
        "fairness_capability_matrix",
    ),
}

# Domain fit op → capability matrix to call first (audit / workflow routing).
FIT_TO_CAPABILITY_MATRIX: dict[str, str] = {
    "fit_clusters": "unsupervised_capability_matrix",
    "fit_voting": "ensemble_capability_matrix",
    "fit_stacking": "ensemble_capability_matrix",
    "fit_blending": "ensemble_capability_matrix",
    "run_automl": "automl_capability_matrix",
    "fit_forecast": "forecast_capability_matrix",
    "fit_torch": "dl_capability_matrix",
    "analyze_timeseries": "timeseries_capability_matrix",
    "ts_decompose": "timeseries_capability_matrix",
    "ts_diagnostics": "timeseries_capability_matrix",
    "fit_anomaly": "anomaly_capability_matrix",
    "fit_semisupervised": "semisupervised_capability_matrix",
    "fit_ssl_pretext": "ssl_capability_matrix",
    "finetune_ssl_head": "ssl_capability_matrix",
    "fit_active_learner": "activelearning_capability_matrix",
    "fit_online": "online_capability_matrix",
    "fit_multitask": "multitask_capability_matrix",
    "fit_metalearning": "metalearning_capability_matrix",
    "fit_federated": "federated_capability_matrix",
    "fit_probabilistic": "probabilistic_capability_matrix",
    "fit_causal": "causal_capability_matrix",
    "fit_graph": "graph_capability_matrix",
    "fit_symbolic": "symbolic_capability_matrix",
    "fit_neuro_symbolic": "symbolic_capability_matrix",
    "fit_cbr": "cbr_capability_matrix",
    "profile_text_corpus": "nlp_capability_matrix",
    "fit_text_classifier": "nlp_capability_matrix",
    "fit_topics": "nlp_capability_matrix",
    "fit_imitation": "rl_capability_matrix",
    "fit_rl": "rl_capability_matrix",
    "fit_tda": "tda_capability_matrix",
    "fit_recommender": "recommender_capability_matrix",
    "fit_ranker": "ranking_capability_matrix",
    "fit_kg": "kg_capability_matrix",
    "fit_decision_policy": "decision_capability_matrix",
    "fit_synthesizer": "synthetic_capability_matrix",
    "rag_ingest_corpus": "rag_capability_matrix",
    "rag_embed_and_index": "rag_capability_matrix",
    "evaluate_fairness": "fairness_capability_matrix",
    "attach_fairness_to_last_eval": "fairness_capability_matrix",
    "suggest_fairness_thresholds": "fairness_capability_matrix",
    "suggest_fairness_reweighing": "fairness_capability_matrix",
}


def load_capability_matrix(operation: str) -> dict[str, Any]:
    """Load one domain capability matrix by Session operation name.

    Dispatches to the domain catalog function registered in ``_MATRIX_SOURCES``.
    Used by walkthrough status enrichment and audit routing: not a Session call
    itself.

    Parameters
    ----------
    operation:
        A Session static method name such as ``session.forecast.capability_matrix``.

    Returns
    -------
    dict
        The matrix payload from the domain catalog. Missing optional deps are
        reported as unavailable rather than raised.
    """
    cached = _MATRIX_CACHE.get(operation)
    if cached is not None:
        return copy.deepcopy(cached)

    source = _MATRIX_SOURCES.get(operation)
    if source is None:
        return {"operation": operation, "error": "unknown_capability_matrix_operation"}
    module_name, attr = source
    module = importlib.import_module(module_name)
    payload = getattr(module, attr)()
    if isinstance(payload, dict):
        result = payload
    else:
        to_dict = getattr(payload, "to_dict", None)
        if callable(to_dict):
            as_dict = to_dict()
            if isinstance(as_dict, dict):
                result = as_dict
            else:
                result = {"operation": operation, "payload": as_dict}
        else:
            result = {"operation": operation, "payload": payload}
    _MATRIX_CACHE[operation] = result
    return copy.deepcopy(result)


def capability_matrix_api_action(operation: str) -> str:
    """Return the public Session call string for one capability-matrix operation.

    Formats teaching overlays and audit suggestions as copy-pasteable
    ``Session.<operation>()`` strings.

    Parameters
    ----------
    operation:
        Capability-matrix static method name on Session.

    Returns
    -------
    str
        For example ``session.forecast.capability_matrix()``.
    """
    return f"Session.{operation}()"


def attach_capability_matrix(status: dict[str, Any], operation: str) -> dict[str, Any]:
    """Attach ``capability_matrix`` and routing metadata to a domain status dict.

    Returns a shallow copy when the matrix is not already embedded, loading the
    live catalog payload via :func:`load_capability_matrix`.

    Parameters
    ----------
    status:
        Existing walkthrough status payload (mutated copy returned).
    operation:
        Capability-matrix Session operation name.

    Returns
    -------
    dict
        Status with ``capability_matrix``, ``capability_operation``, and
        ``capability_introspection`` when not already present.
    """
    if status.get("capability_matrix") is not None:
        return status
    enriched = dict(status)
    enriched["capability_operation"] = operation
    enriched["capability_introspection"] = capability_matrix_api_action(operation)
    enriched["capability_matrix"] = load_capability_matrix(operation)
    return enriched


def capability_introspection_status(
    session: Any | None = None,
    *,
    capability_probe: str = "eager",
) -> dict[str, Any]:
    """Aggregate capability-matrix routing for walkthrough orientation tables.

    Builds one row per domain status field with the matching matrix operation,
    default backend hint, and backend count. Matrices are static introspection
    and do not read Session history today.

    Parameters
    ----------
    session:
        Optional Session (reserved for future history-aware routing). Currently
        unused; matrices are static introspection.
    capability_probe:
        ``eager`` loads every matrix (cached process-wide). ``lazy`` / ``skip``
        list operations without importing optional industry stacks.

    Returns
    -------
    dict
        Per-domain routing table, disclosures, and the full operation list.
    """
    _ = session
    probe = str(capability_probe or "eager").lower().strip()
    rows: list[dict[str, Any]] = []
    seen_ops: set[str] = set()
    for status_field, operation in DOMAIN_STATUS_CAPABILITY_OPS:
        if operation in seen_ops and status_field != "imitation_status":
            continue
        seen_ops.add(operation)
        if probe in {"lazy", "skip"}:
            rows.append(
                {
                    "domain_status_field": status_field,
                    "operation": operation,
                    "api_action": capability_matrix_api_action(operation),
                    "default_backend": None,
                    "n_backends": None,
                    "probed": False,
                }
            )
            continue
        matrix = load_capability_matrix(operation)
        default = (
            matrix.get("default_backend")
            or matrix.get("default_method")
            or matrix.get("default")
        )
        backends = matrix.get("backends")
        n_backends = len(backends) if isinstance(backends, dict) else None
        rows.append(
            {
                "domain_status_field": status_field,
                "operation": operation,
                "api_action": capability_matrix_api_action(operation),
                "default_backend": default,
                "n_backends": n_backends,
                "probed": True,
            }
        )
    return {
        "n_domains": len(rows),
        "domains": rows,
        "operations": sorted(CAPABILITY_MATRIX_OPERATIONS),
        "capability_probe": probe,
        "disclosures": [
            "Capability matrices are read-only: they report installed backends, "
            "not quality or appropriateness for your dataset.",
            "Call Session.<domain>_capability_matrix() before choosing a backend "
            "or fit method; the matching AI tool dispatches the same static method.",
            "Walkthrough defaults to capability_probe='lazy' so inactive domains "
            "do not import torch-heavy optional stacks; use 'eager' to probe all.",
            "Matrices are cached process-wide after the first load.",
        ],
    }


def suggest_capability_introspection(
    history: list[dict[str, Any]] | None,
    *,
    available_fit_ops: set[str] | frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    """Suggest capability-matrix calls when fit paths are open but unchecked.

    Compares workflow-resolver fit operations against history to recommend a
    matrix introspection call before the user commits to a backend that may be
    missing on this install.

    Parameters
    ----------
    history:
        Normalized Session history records.
    available_fit_ops:
        Fit operation names currently ``available`` in the workflow resolver.

    Returns
    -------
    list of dict
        Suggested introspection ops with ``api_action`` and ``reason``.
    """
    recorded = {
        str(record.get("operation_id") or record.get("action") or "")
        for record in list(history or [])
    }
    fit_ops = set(available_fit_ops or ())
    suggestions: list[dict[str, Any]] = []
    for fit_op, matrix_op in FIT_TO_CAPABILITY_MATRIX.items():
        if fit_op not in fit_ops:
            continue
        if matrix_op in recorded:
            continue
        suggestions.append(
            {
                "operation": matrix_op,
                "status": "available",
                "reason": (
                    f"{fit_op} is available but {matrix_op} has not been recorded; "
                    "check backend availability before fitting."
                ),
                "api_action": f"Session.explain({matrix_op!r}, moment='before')",
                "evidence": f"capability-routing-{fit_op}",
            }
        )
    return suggestions
