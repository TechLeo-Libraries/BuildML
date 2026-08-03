"""Turn ensemble state into the flat records history and walkthroughs consume.

The Session records what happened, and the walkthrough explains it. Both need
ensemble facts in a plain, JSON-safe shape rather than as live objects holding
fitted estimators.

These helpers read defensively: ``getattr`` with defaults rather than attribute
access, missing keys treated as ``None``: because they run against whatever
state the Session happens to be in, including partial and legacy states. A
status helper that raises is worse than one that reports an absence.

See Also
--------
buildml.ensemble.results : The objects being summarised.
"""

from __future__ import annotations

from typing import Any


def fit_result_summary(fit_result: Any) -> dict[str, Any]:
    """Reduce an ensemble fit to the handful of fields history should keep.

    History entries are written on every operation and are meant to stay small,
    so this keeps only what identifies the ensemble: strategy, task, bases,
    meta-learner, and the strategy-specific setting that mattered (``cv`` for
    stacking, ``holdout_fraction`` for blending). The fitted estimator is
    deliberately left out; history should stay serialisable and small.

    Parameters
    ----------
    fit_result:
        An :class:`~buildml.ensemble.results.EnsembleFitResult`, a mapping, or
        ``None``.

    Returns
    -------
    dict
        The summary fields, with ``None`` for anything not applicable to the
        strategy. An empty dict when there is nothing to summarise.

    Notes
    -----
    **Keys are always present**, valued ``None`` where they do not apply, so a
    consumer can index without guarding. ``voting`` is ``None`` for a stack,
    ``cv`` is ``None`` for a blend.

    Examples
    --------
    >>> fit_result_summary(None)
    {}
    >>> summary = fit_result_summary({"strategy": "voting", "task": "classification"})
    >>> summary["strategy"], summary["cv"]
    ('voting', None)
    """
    if fit_result is None:
        return {}
    if hasattr(fit_result, "to_dict"):
        payload = fit_result.to_dict()
    else:
        payload = dict(fit_result)
    return {
        "strategy": payload.get("strategy"),
        "task": payload.get("task"),
        "estimator_names": payload.get("estimator_names"),
        "n_train_rows": payload.get("n_train_rows"),
        "final_estimator_name": payload.get("final_estimator_name"),
        "voting": payload.get("voting"),
        "cv": payload.get("cv"),
        "holdout_fraction": payload.get("holdout_fraction"),
        "blend_method": payload.get("blend_method"),
    }


def ensemble_status(
    plan: Any = None,
    *,
    fit_result: Any = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Report whether an ensemble is attached, and what it discloses.

    Distinguishes three states, and the distinction is the point. A live plan
    means an ensemble is attached now. History without a plan means one was
    fitted earlier and then lost: most often because the Session was restored
    from a checkpoint, which by design does not carry the ensemble. Neither
    means no ensemble was ever involved.

    That middle state is the one worth surfacing. Someone who fits an ensemble,
    checkpoints, restores, and finds their predictions come from a single model
    is looking at a confusing situation, and this is where it gets named.

    Parameters
    ----------
    plan:
        The attached :class:`~buildml.ensemble.results.EnsemblePlan`, or
        ``None``.
    fit_result:
        The attached fit result, if any. Only its presence is reported.
    history:
        Session history records, scanned for ensemble operations.

    Returns
    -------
    dict
        ``enabled``: a plan is attached. ``present``: a plan is attached or
        history shows one was. ``has_ensemble_plan``, ``strategy``, ``task``,
        ``estimator_names``, ``has_fit_result``: the details, ``None`` when
        absent. ``disclosures``: the plan's own notes plus the standing ones
        about train-only meta-learner fitting. ``boundary``: the reminder that
        this path differs from passing a single forest to ``fit``.

    Notes
    -----
    **Everything here is factual.** No judgement about whether the ensemble was
    a good idea; that is :mod:`buildml.model.compare`'s job.

    Examples
    --------
    >>> status = ensemble_status(None)
    >>> status["enabled"], status["present"]
    (False, False)
    >>> status = ensemble_status(None, history=[{"operation_id": "fit_voting"}])
    >>> status["enabled"], status["present"]
    (False, True)

    See Also
    --------
    ensemble_status_for_session : The same, reading a Session directly.
    """
    records = list(history or [])
    saw = any(
        str(r.get("operation_id") or r.get("action"))
        in {
            "fit_voting",
            "fit_stacking",
            "fit_blending",
            "save_ensemble_bundle",
            "load_ensemble_bundle",
        }
        for r in records
    )
    enabled = plan is not None
    disclosures: list[str] = []
    if enabled:
        disclosures.extend(
            [
                f"EnsemblePlan strategy={getattr(plan, 'strategy', None)}, "
                f"task={getattr(plan, 'task', None)}, "
                f"bases={list(getattr(plan, 'estimator_names', ()) or ())}.",
                "Session checkpoints do not embed EnsemblePlan; use "
                "save_ensemble_bundle / load_ensemble_bundle (or save_pipeline for "
                "preprocess + estimator).",
                "Stacking / blending meta-learners are fit inside train only; "
                "Session test is never used for meta features.",
            ]
        )
        for note in getattr(plan, "disclosures", ()) or ():
            disclosures.append(str(note))
    elif saw:
        disclosures.append(
            "Ensemble operations appear in history, but no live EnsemblePlan is attached."
        )

    return {
        "enabled": enabled,
        "present": enabled or saw,
        "has_ensemble_plan": enabled,
        "strategy": None if plan is None else getattr(plan, "strategy", None),
        "task": None if plan is None else getattr(plan, "task", None),
        "estimator_names": (
            None if plan is None else list(getattr(plan, "estimator_names", ()) or ())
        ),
        "has_fit_result": fit_result is not None,
        "disclosures": disclosures,
        "boundary": (
            "Native ensembles are a Session domain path distinct from passing a single "
            "RandomForest to Session.fit."
        ),
    }


def ensemble_status_for_session(session: Any) -> dict[str, Any]:
    """Read ensemble state off a Session and report it.

    The adapter the walkthrough calls. Pulls the plan, fit result, and history
    off the Session and hands them to :func:`ensemble_status`, keeping the
    knowledge of Session internals in one place instead of spread across every
    caller.

    Parameters
    ----------
    session:
        A :class:`~buildml.session.Session`. Attributes are read defensively, so
        a Session in any state: including one that never touched ensembles :
        produces a report rather than an error.

    Returns
    -------
    dict
        As :func:`ensemble_status`.

    See Also
    --------
    ensemble_status : The underlying report, for state not held by a Session.
    """
    return ensemble_status(
        getattr(session, "_ensemble_plan", None),
        fit_result=getattr(session, "_fit_result", None),
        history=list(getattr(session, "_history", ()) or ()),
    )
