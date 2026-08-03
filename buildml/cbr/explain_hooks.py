"""Condense CBR results into the small payloads history and walkthroughs record.

A prediction result carries a trace per row, each naming its neighbours and
their distances; a session that predicted a large partition would carry all of
it. History wants the shape of what happened — settings, counts, headline
metrics — not the payload.

Each summariser accepts ``None``, a result object, or a plain dict, and reads
fields through ``.get``, so a missing key becomes ``None`` rather than an
exception. Explanation should never be the thing that breaks a working session.

The status builder states absences as plainly as presences: no plan attached, no
evaluation run, and the fact that a Session checkpoint does not carry case
memory. Silence on those points reads as reassurance, which is exactly the wrong
inference.

See Also
--------
buildml.cbr.results : The result objects being summarised.
buildml.cbr.catalog.cbr_capability_matrix : What the install can do.
"""

from __future__ import annotations

from typing import Any


def fit_result_summary(fit_result: Any) -> dict[str, Any]:
    """Reduce a fit report to the settings and counts worth keeping.

    Enough to answer later "what was built, and how?" without the column lists
    or disclosure text.

    Parameters
    ----------
    fit_result:
        A :class:`~buildml.cbr.results.CbrFitResult`, an equivalent dict, or
        ``None``.

    Returns
    -------
    dict
        Task, backend, metric, reuse mode, ``k``, row and case counts, target
        column, and training score. Empty when the input is ``None``.

    Notes
    -----
    **``backend`` is what actually ran**, which may differ from what was
    requested if an optional dependency was absent.

    **``train_score`` is recorded but is not a measurement.** Every training row
    is its own nearest neighbour, so it approaches perfect regardless.
    """
    if fit_result is None:
        return {}
    payload = fit_result.to_dict() if hasattr(fit_result, "to_dict") else dict(fit_result)
    return {
        "task": payload.get("task"),
        "backend": payload.get("backend"),
        "metric": payload.get("metric"),
        "reuse": payload.get("reuse"),
        "k": payload.get("k"),
        "n_train_rows": payload.get("n_train_rows"),
        "n_cases": payload.get("n_cases"),
        "target_column": payload.get("target_column"),
        "train_score": payload.get("train_score"),
    }


def eval_result_summary(eval_result: Any) -> dict[str, Any]:
    """Reduce an evaluation to its metrics and coverage diagnostic.

    A score is only interpretable alongside what it was scored on and how far
    the neighbours were, so the summary keeps those and drops everything else.

    Parameters
    ----------
    eval_result:
        A :class:`~buildml.cbr.results.CbrEvalResult`, an equivalent dict, or
        ``None``.

    Returns
    -------
    dict
        Partition, task, row count, the metrics, and
        ``mean_neighbor_distance``. Empty when the input is ``None``.

    Notes
    -----
    **The neighbour distance is kept deliberately.** It is what separates a
    score describing interpolation from one describing extrapolation, and a
    metric recorded without it is a metric that cannot be qualified later.

    **The partition is kept for the same reason.** A number scored on train
    means something entirely different from one scored on holdout.
    """
    if eval_result is None:
        return {}
    payload = eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
    return {
        "partition": payload.get("partition"),
        "task": payload.get("task"),
        "n_rows": payload.get("n_rows"),
        "metrics": payload.get("metrics"),
        "mean_neighbor_distance": payload.get("mean_neighbor_distance"),
    }


def predict_result_summary(predict_result: Any) -> dict[str, Any]:
    """Reduce a prediction run to counts, without the predictions.

    History should record that a prediction happened and at what scale, not
    reproduce its output. The counts do that in a handful of bytes.

    Parameters
    ----------
    predict_result:
        A :class:`~buildml.cbr.results.CbrPredictResult`, an equivalent dict, or
        ``None``.

    Returns
    -------
    dict
        Partition, task, row count, prediction count, and trace count. Empty
        when the input is ``None``.

    Notes
    -----
    **Neither predictions nor traces are kept.** Both can be very large, and the
    traces contain feature-derived detail about individual rows that has no
    place in a session log.
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
        "task": payload.get("task"),
        "n_rows": payload.get("n_rows"),
        "n_predictions": payload.get("n_predictions"),
        "n_traces": payload.get("n_traces"),
    }


def retrieve_result_summary(retrieve_result: Any) -> dict[str, Any]:
    """Reduce a retrieval run to its settings and counts.

    Keeps the two things that make a later retrieval comparable to this one —
    ``k`` and the metric — plus how much was retrieved.

    Parameters
    ----------
    retrieve_result:
        A :class:`~buildml.cbr.results.CbrRetrieveResult`, an equivalent dict,
        or ``None``.

    Returns
    -------
    dict
        Partition, ``k``, metric, query count, and trace count. Empty when the
        input is ``None``.

    Notes
    -----
    **The neighbours themselves are not kept**, only how many queries ran. Read
    ``traces`` off the result while you have it.
    """
    if retrieve_result is None:
        return {}
    payload = (
        retrieve_result.to_dict()
        if hasattr(retrieve_result, "to_dict")
        else dict(retrieve_result)
    )
    return {
        "partition": payload.get("partition"),
        "k": payload.get("k"),
        "metric": payload.get("metric"),
        "n_queries": payload.get("n_queries"),
        "n_traces": payload.get("n_traces"),
    }


def retain_result_summary(retain_result: Any) -> dict[str, Any]:
    """Reduce a retention outcome to its three counts.

    How many cases were added, how large memory is now, and how many were
    refused — enough to reconstruct how the case base grew.

    Parameters
    ----------
    retain_result:
        A :class:`~buildml.cbr.results.CbrRetainResult`, an equivalent dict, or
        ``None``.

    Returns
    -------
    dict
        ``n_added``, ``n_cases_after``, and ``n_skipped``. Empty when the input
        is ``None``.

    Notes
    -----
    **These are worth keeping across a session's history.** Retention is what
    changes a case base over time, and the sequence of these counts is the
    context for any later change in behaviour.
    """
    if retain_result is None:
        return {}
    payload = (
        retain_result.to_dict()
        if hasattr(retain_result, "to_dict")
        else dict(retain_result)
    )
    return {
        "n_added": payload.get("n_added"),
        "n_cases_after": payload.get("n_cases_after"),
        "n_skipped": payload.get("n_skipped"),
    }


def cbr_status(
    cbr_plan: Any = None,
    *,
    fit_result: Any = None,
    eval_result: Any = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Describe the state of a CBR workflow, including what is missing from it.

    Assembles a factual picture for walkthroughs and status displays: whether a
    reasoner is attached, how it was configured, how large its memory is, and
    what has been evaluated.

    The absences are reported as explicitly as the presences. No plan attached
    says so. CBR operations in history with no live plan says so. And the fact
    that a Session checkpoint does not carry case memory is stated every time,
    because that is the assumption people make and it costs them the reasoner.

    Parameters
    ----------
    cbr_plan:
        The fitted reasoner, or ``None``.
    fit_result:
        The fit report, or ``None``.
    eval_result:
        The last evaluation, or ``None``.
    history:
        Session history records, scanned for past CBR operations.

    Returns
    -------
    dict
        ``enabled`` (a plan is attached), ``present`` (CBR has been used at
        all), the plan's configuration and case count, flags for which results
        exist, and ``disclosures`` in plain language.

    Notes
    -----
    **``enabled`` and ``present`` differ, and the difference matters.** A
    session restored from a checkpoint has ``present=True`` from history but
    ``enabled=False``, because case memory did not travel with the checkpoint
    and must be reloaded from a bundle or refitted.

    **Nothing here is a readiness verdict.** It reports facts; whether they
    amount to a system worth deploying is a judgement this cannot make.

    See Also
    --------
    cbr_status_for_session : The Session-facing wrapper.
    buildml.cbr.checkpoint.save_cbr_bundle : Persisting the reasoner properly.
    """
    records = list(history or [])
    saw = any(
        str(r.get("operation_id") or r.get("action"))
        in {
            "fit_cbr",
            "retrieve_cases",
            "predict_cbr",
            "evaluate_cbr",
            "retain_cbr",
            "save_cbr_bundle",
            "load_cbr_bundle",
        }
        for r in records
    )
    enabled = cbr_plan is not None
    disclosures: list[str] = []
    if cbr_plan is not None:
        n_cases = None
        n_retained = None
        cb = getattr(cbr_plan, "case_base", None)
        if cb is not None:
            n_cases = getattr(cb, "n_cases", None)
            n_retained = getattr(cb, "n_retained", None)
        disclosures.extend(
            [
                f"CbrPlan task={getattr(cbr_plan, 'task', None)}, "
                f"backend={getattr(cbr_plan, 'backend', None)}, "
                f"metric={getattr(cbr_plan, 'metric', None)}, "
                f"reuse={getattr(cbr_plan, 'reuse', None)}, "
                f"k={getattr(cbr_plan, 'k', None)}, "
                f"n_cases={n_cases}, n_retained={n_retained}.",
                "Case memory is built from Session train only. "
                "Validation/test are retrieve / predict / evaluate only "
                "(retain refuses holdout indices).",
                "Session checkpoints do not embed CbrPlan; use "
                "save_cbr_bundle / load_cbr_bundle.",
                "Honesty: tabular case→solution CBR — not RAG document "
                "retrieval for generation, not a vector DB product.",
            ]
        )
        for note in getattr(cbr_plan, "disclosures", ()) or ():
            disclosures.append(str(note))
    elif saw and not enabled:
        disclosures.append(
            "CBR operations appear in history, but no live CbrPlan is attached."
        )

    eval_payload = None
    if eval_result is not None:
        eval_payload = (
            eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
        )
        disclosures.append(
            "Last CBR eval: "
            f"partition={eval_payload.get('partition')}, "
            f"metrics={eval_payload.get('metrics')}."
        )

    return {
        "enabled": enabled,
        "present": enabled or saw,
        "has_cbr_plan": cbr_plan is not None,
        "task": None if cbr_plan is None else getattr(cbr_plan, "task", None),
        "backend": None if cbr_plan is None else getattr(cbr_plan, "backend", None),
        "metric": None if cbr_plan is None else getattr(cbr_plan, "metric", None),
        "reuse": None if cbr_plan is None else getattr(cbr_plan, "reuse", None),
        "k": None if cbr_plan is None else getattr(cbr_plan, "k", None),
        "n_cases": (
            None
            if cbr_plan is None
            else getattr(getattr(cbr_plan, "case_base", None), "n_cases", None)
        ),
        "has_fit_result": fit_result is not None,
        "has_eval_result": eval_result is not None,
        "eval": eval_payload,
        "disclosures": disclosures,
        "boundary": (
            "Case-based reasoning retrieves similar train cases and reuses/"
            "adapts their solutions for supervised-style tasks. Distinct from "
            "RAG (text corpus → generation). Not a vector DB product."
        ),
    }


def cbr_status_for_session(session: Any) -> dict[str, Any]:
    """Read a Session's CBR state and describe it.

    Pulls the plan, the fit and evaluation results, and the history off a
    Session and hands them to :func:`cbr_status`. Every read uses ``getattr``
    with a default, so this works on a Session that has never touched CBR and on
    partially-constructed or mock objects.

    Parameters
    ----------
    session:
        A :class:`~buildml.session.Session`, or anything with the same
        attributes.

    Returns
    -------
    dict
        The status payload from :func:`cbr_status`.

    See Also
    --------
    cbr_status : The underlying builder and its return shape.
    """
    return cbr_status(
        getattr(session, "_cbr_plan", None),
        fit_result=getattr(session, "_cbr_fit_result", None),
        eval_result=getattr(session, "_cbr_eval_result", None),
        history=list(getattr(session, "_history", ()) or ()),
    )
