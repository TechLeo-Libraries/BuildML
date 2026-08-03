"""History / catalog / walkthrough helpers for symbolic operations."""

from __future__ import annotations

from typing import Any


def fit_result_summary(fit_result: Any) -> dict[str, Any]:
    """Build a compact history summary from a symbolic fit result.

    Strips full rule bodies while recording backend, source, rule counts, and
    train scores for Session audit logs.

    Parameters
    ----------
    fit_result:
        :class:`~buildml.symbolic.results.SymbolicFitResult`,
        :class:`~buildml.symbolic.results.NeuroSymbolicFitResult`, or ``None``.

    Returns
    -------
    dict[str, Any]
        Backend, method, task, rule count, and score metadata.
    """
    if fit_result is None:
        return {}
    payload = fit_result.to_dict() if hasattr(fit_result, "to_dict") else dict(fit_result)
    return {
        "backend": payload.get("backend"),
        "source": payload.get("source"),
        "method": payload.get("method"),
        "mode": payload.get("mode"),
        "base_estimator_name": payload.get("base_estimator_name"),
        "torch_method": payload.get("torch_method"),
        "task": payload.get("task"),
        "n_train_rows": payload.get("n_train_rows"),
        "n_rules": payload.get("n_rules"),
        "provenance": payload.get("provenance") or payload.get("rule_provenance"),
        "target_column": payload.get("target_column"),
        "train_accuracy": payload.get("train_accuracy"),
        "train_score": payload.get("train_score"),
    }


def eval_result_summary(eval_result: Any) -> dict[str, Any]:
    """Build a compact history summary from a symbolic evaluation result.

    Strips full metric payloads down to partition, coverage, and repair stats
    for Session audit logs after evaluate completes.

    Parameters
    ----------
    eval_result:
        :class:`~buildml.symbolic.results.SymbolicEvalResult` or ``None``.

    Returns
    -------
    dict[str, Any]
        Partition, metrics, rule coverage, and repair-rate summaries.
    """
    if eval_result is None:
        return {}
    payload = eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
    return {
        "partition": payload.get("partition"),
        "path": payload.get("path"),
        "task": payload.get("task"),
        "n_rows": payload.get("n_rows"),
        "metrics": payload.get("metrics"),
        "rule_coverage": payload.get("rule_coverage"),
        "mean_rules_fired": payload.get("mean_rules_fired"),
        "repair_rate": payload.get("repair_rate"),
    }


def predict_result_summary(predict_result: Any) -> dict[str, Any]:
    """Build a compact history summary from a symbolic predict result.

    Records prediction and trace counts without embedding full per-row traces
    in Session history entries.

    Parameters
    ----------
    predict_result:
        :class:`~buildml.symbolic.results.SymbolicPredictResult` or ``None``.

    Returns
    -------
    dict[str, Any]
        Partition, prediction counts, trace counts, and repair counts.
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
        "path": payload.get("path"),
        "task": payload.get("task"),
        "n_rows": payload.get("n_rows"),
        "n_predictions": payload.get("n_predictions"),
        "n_traces": payload.get("n_traces"),
        "n_repaired": payload.get("n_repaired"),
    }


def symbolic_status(
    symbolic_plan: Any = None,
    neuro_plan: Any = None,
    *,
    fit_result: Any = None,
    neuro_fit_result: Any = None,
    eval_result: Any = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build factual walkthrough disclosure for symbolic Session state.

    Combines live plan metadata, optional evaluation summaries, history
    detection, and :func:`symbolic_capability_matrix` for teaching overlays.

    Parameters
    ----------
    symbolic_plan:
        Active :class:`~buildml.symbolic.results.SymbolicPlan`, if any.
    neuro_plan:
        Active :class:`~buildml.symbolic.results.NeuroSymbolicPlan`, if any.
    fit_result, neuro_fit_result:
        Last fit reports attached to the Session.
    eval_result:
        Last evaluation result.
    history:
        Session operation records.

    Returns
    -------
    dict[str, Any]
        Enabled flags, backend metadata, embedded capability matrix, disclosures,
        and boundary text separating tabular rules from logic-programming products.
    """
    records = list(history or [])
    saw = any(
        str(r.get("operation_id") or r.get("action"))
        in {
            "fit_symbolic",
            "evaluate_symbolic",
            "predict_symbolic",
            "fit_neuro_symbolic",
            "evaluate_neuro_symbolic",
            "predict_neuro_symbolic",
            "save_symbolic_bundle",
            "load_symbolic_bundle",
        }
        for r in records
    )
    enabled = symbolic_plan is not None or neuro_plan is not None
    disclosures: list[str] = []
    try:
        from buildml.symbolic.catalog import symbolic_capability_matrix

        cap = symbolic_capability_matrix()
        disclosures.append(
            "Capability matrix: "
            f"symbolic default_backend={cap.get('default_symbolic_backend_when_installed')}, "
            f"neuro default_backend={cap.get('default_neuro_backend_when_installed')}, "
            f"industry={cap.get('industry_extra_present')}, z3={cap.get('z3_present')}."
        )
    except Exception:  # noqa: BLE001
        pass
    if symbolic_plan is not None:
        disclosures.extend(
            [
                f"SymbolicPlan backend={getattr(symbolic_plan, 'backend', None)}, "
                f"source={getattr(symbolic_plan, 'source', None)}, "
                f"method={getattr(symbolic_plan, 'method', None)}, "
                f"task={getattr(symbolic_plan, 'task', None)}, "
                f"n_rules={getattr(symbolic_plan, 'n_rules', None)}, "
                f"provenance={getattr(getattr(symbolic_plan, 'knowledge_base', None), 'provenance', None)}.",
                "Rule induction / compile uses Session train only. "
                "Validation/test are evaluation / scoring only.",
                "Session checkpoints do not embed SymbolicPlan; use "
                "save_symbolic_bundle / load_symbolic_bundle.",
                "Honesty: structured tabular if-then rules: not Prolog/Z3/AGI.",
            ]
        )
        for note in getattr(symbolic_plan, "disclosures", ()) or ():
            disclosures.append(str(note))
    if neuro_plan is not None:
        disclosures.extend(
            [
                f"NeuroSymbolicPlan backend={getattr(neuro_plan, 'backend', None)}, "
                f"mode={getattr(neuro_plan, 'mode', None)}, "
                f"base={getattr(neuro_plan, 'base_estimator_name', None)}, "
                f"torch_method={getattr(neuro_plan, 'torch_method', None)}, "
                f"task={getattr(neuro_plan, 'task', None)}, "
                f"n_rules={len(getattr(getattr(neuro_plan, 'knowledge_base', None), 'rules', ()) or ())}.",
                "Hybrid integrates sklearn predictions with symbolic "
                "constraints/features in one Session API (not separate ad-hoc calls).",
                "Honesty: sklearn + rule hybrid: not a deep neuro-symbolic research stack.",
            ]
        )
        for note in getattr(neuro_plan, "disclosures", ()) or ():
            disclosures.append(str(note))
    elif saw and not enabled:
        disclosures.append(
            "Symbolic operations appear in history, but no live "
            "SymbolicPlan / NeuroSymbolicPlan is attached."
        )

    eval_payload = None
    if eval_result is not None:
        eval_payload = (
            eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
        )
        disclosures.append(
            "Last symbolic eval: "
            f"path={eval_payload.get('path')}, "
            f"partition={eval_payload.get('partition')}, "
            f"metrics={eval_payload.get('metrics')}."
        )

    return {
        "enabled": enabled,
        "present": enabled or saw,
        "has_symbolic_plan": symbolic_plan is not None,
        "has_neuro_symbolic_plan": neuro_plan is not None,
        "backend": (
            None
            if symbolic_plan is None and neuro_plan is None
            else getattr(symbolic_plan or neuro_plan, "backend", None)
        ),
        "source": None if symbolic_plan is None else getattr(symbolic_plan, "source", None),
        "method": None if symbolic_plan is None else getattr(symbolic_plan, "method", None),
        "mode": None if neuro_plan is None else getattr(neuro_plan, "mode", None),
        "task": (
            None
            if symbolic_plan is None and neuro_plan is None
            else getattr(symbolic_plan or neuro_plan, "task", None)
        ),
        "n_rules": (
            None
            if symbolic_plan is None
            else getattr(symbolic_plan, "n_rules", None)
        ),
        "has_fit_result": fit_result is not None or neuro_fit_result is not None,
        "has_eval_result": eval_result is not None,
        "eval": eval_payload,
        "capability_matrix": _capability_matrix_safe(),
        "disclosures": disclosures,
        "boundary": (
            "Symbolic AI compiles/induces tabular if-then rules; neuro-symbolic "
            "hybrids combine sklearn estimators with rule overlay, features, or "
            "constraint repair. Not an AGI reasoner; not Prolog/Z3."
        ),
    }


def symbolic_status_for_session(session: Any) -> dict[str, Any]:
    """Report symbolic status for a Session walkthrough panel.

    Reads symbolic and neuro-symbolic plan slots without mutating the Session.

    Parameters
    ----------
    session:
        :class:`~buildml.session.session.Session` instance.

    Returns
    -------
    dict[str, Any]
        Same payload as :func:`symbolic_status` for the Session's symbolic state.
    """
    return symbolic_status(
        getattr(session, "_symbolic_plan", None),
        getattr(session, "_neuro_symbolic_plan", None),
        fit_result=getattr(session, "_symbolic_fit_result", None),
        neuro_fit_result=getattr(session, "_neuro_symbolic_fit_result", None),
        eval_result=getattr(session, "_symbolic_eval_result", None),
        history=list(getattr(session, "_history", ()) or ()),
    )


def _capability_matrix_safe() -> dict[str, Any]:
    try:
        from buildml.symbolic.catalog import symbolic_capability_matrix

        return symbolic_capability_matrix()
    except Exception:  # noqa: BLE001
        return {}
