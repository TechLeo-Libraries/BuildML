"""Condense policy results into the small records history and reports show.

A history entry cannot hold a full result — an acting result contains a score
per arm per row. Each summary function reduces one result to the fields that
belong in a timeline: what was done, on what, and the numbers a reader would
want at a glance.

The ``*_status`` functions do something broader: they describe where a domain
stands, for a walkthrough or an audit trail. What is attached, what the history
shows, and the caveats that apply.

Two rules run throughout. They never raise — a missing or malformed result
becomes an empty dict, because a failed history entry is worse than a thin one.
And they report facts, never advice; the teaching prose lives in
:mod:`buildml.explain`, and mixing the two would put opinions in the audit
trail.

One caveat is promoted deliberately. Bandit metrics carry an ``offline`` flag,
and it travels with the numbers everywhere in this module. A recorded metric
that has lost track of whether it was estimated or measured will eventually be
read as the stronger of the two.
"""

from __future__ import annotations

from typing import Any


def imitation_fit_summary(fit_result: Any) -> dict[str, Any]:
    """Condense a cloning fit into a history entry.

    Keeps what identifies the policy and what characterises the demonstrations
    it learned from, so two fits can be told apart without reopening either.

    Parameters
    ----------
    fit_result:
        An :class:`~buildml.rl.results.ImitationFitResult`, an equivalent dict,
        or ``None``.

    Returns
    -------
    dict
        Task, backend, estimator, method, training size, action column, and the
        in-sample score. Empty when there was nothing to summarise.

    Notes
    -----
    ``train_score`` is in-sample agreement with the demonstrator. It is recorded
    because it is part of what happened, not because it indicates quality.
    """
    if fit_result is None:
        return {}
    payload = fit_result.to_dict() if hasattr(fit_result, "to_dict") else dict(fit_result)
    return {
        "task": payload.get("task"),
        "backend": payload.get("backend"),
        "estimator": payload.get("estimator"),
        "method": payload.get("method"),
        "n_train_rows": payload.get("n_train_rows"),
        "action_column": payload.get("action_column"),
        "train_score": payload.get("train_score"),
    }


def imitation_eval_summary(eval_result: Any) -> dict[str, Any]:
    """Condense a holdout cloning evaluation into a history entry.

    Metrics are small enough to keep whole, and they are the thing a reader
    scanning a timeline is looking for.

    Parameters
    ----------
    eval_result:
        An :class:`~buildml.rl.results.ImitationEvalResult`, an equivalent
        dict, or ``None``.

    Returns
    -------
    dict
        Partition, task, row count, and the full metric mapping. Empty when
        there was nothing to summarise.
    """
    if eval_result is None:
        return {}
    payload = eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
    return {
        "partition": payload.get("partition"),
        "task": payload.get("task"),
        "n_rows": payload.get("n_rows"),
        "metrics": payload.get("metrics"),
    }


def imitation_predict_summary(predict_result: Any) -> dict[str, Any]:
    """Condense a cloning prediction run into a history entry.

    Records that actions were chosen and at what scale. The actions themselves
    are the payload, not the record.

    Parameters
    ----------
    predict_result:
        An :class:`~buildml.rl.results.ImitationPredictResult`, an equivalent
        dict, or ``None``.

    Returns
    -------
    dict
        Partition, task, row count, and how many actions were produced. Empty
        when there was nothing to summarise.
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
        "n_actions": payload.get("n_actions"),
    }


def rl_fit_summary(fit_result: Any) -> dict[str, Any]:
    """Condense an RL fit into a history entry.

    Keeps mode, backend, and algorithm together, because none of the three
    identifies a run on its own — the same algorithm name means different things
    under different modes.

    Parameters
    ----------
    fit_result:
        An :class:`~buildml.rl.results.RlFitResult`, an equivalent dict, or
        ``None``.

    Returns
    -------
    dict
        Mode, backend, algorithm, training size, arm count, environment, and
        the training metrics. Empty when there was nothing to summarise.

    Notes
    -----
    For bandits, ``train_metrics`` describes the *log* rather than the new
    policy. ``mean_logged_reward`` recorded here is the baseline a later
    evaluation should be compared against.
    """
    if fit_result is None:
        return {}
    payload = fit_result.to_dict() if hasattr(fit_result, "to_dict") else dict(fit_result)
    return {
        "mode": payload.get("mode"),
        "backend": payload.get("backend"),
        "algorithm": payload.get("algorithm"),
        "n_train_rows": payload.get("n_train_rows"),
        "n_arms": payload.get("n_arms"),
        "env_id": payload.get("env_id"),
        "train_metrics": payload.get("train_metrics"),
    }


def rl_eval_summary(eval_result: Any) -> dict[str, Any]:
    """Condense an RL evaluation into a history entry.

    Carries the ``offline`` flag alongside the metrics, deliberately. A recorded
    number that has lost track of whether it was estimated from a log or
    measured by running the policy is a number that will eventually be misread
    as the stronger of the two.

    Parameters
    ----------
    eval_result:
        An :class:`~buildml.rl.results.RlEvalResult`, an equivalent dict, or
        ``None``.

    Returns
    -------
    dict
        Partition, mode, row count, the metric mapping, and the ``offline``
        flag. Empty when there was nothing to summarise.
    """
    if eval_result is None:
        return {}
    payload = eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
    return {
        "partition": payload.get("partition"),
        "mode": payload.get("mode"),
        "n_rows": payload.get("n_rows"),
        "metrics": payload.get("metrics"),
        "offline": payload.get("offline"),
    }


def rl_act_summary(act_result: Any) -> dict[str, Any]:
    """Condense an acting run into a history entry.

    Records the scale of the run rather than its output. The score matrix alone
    is rows times arms, which has no business in a timeline.

    Parameters
    ----------
    act_result:
        An :class:`~buildml.rl.results.RlActResult`, an equivalent dict, or
        ``None``.

    Returns
    -------
    dict
        Partition, mode, row count, and how many actions were chosen. Empty
        when there was nothing to summarise.
    """
    if act_result is None:
        return {}
    payload = act_result.to_dict() if hasattr(act_result, "to_dict") else dict(act_result)
    return {
        "partition": payload.get("partition"),
        "mode": payload.get("mode"),
        "n_rows": payload.get("n_rows"),
        "n_actions": payload.get("n_actions"),
    }


def imitation_status(
    imitation_plan: Any = None,
    *,
    fit_result: Any = None,
    eval_result: Any = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Describe where the imitation side of a session currently stands.

    Answers "what cloning has happened here and what should I know about it" —
    for a walkthrough, a status display, or an audit record. It reports what is
    attached, what the history shows, and the caveats that apply, all as
    statements of fact rather than advice.

    Parameters
    ----------
    imitation_plan:
        The attached cloning policy, if any.
    fit_result:
        The most recent fit report.
    eval_result:
        The most recent holdout evaluation, surfaced in the disclosures.
    history:
        Session history records, scanned for imitation operations.

    Returns
    -------
    dict
        Whether a policy is attached, whether imitation appears in history at
        all, the task, estimator, and action column, the evaluation payload,
        the current capability matrix, the disclosures, and a statement of
        what this domain does and does not cover.

    Notes
    -----
    **A session can show imitation activity with no policy attached** — most
    often because a checkpoint was reloaded. Checkpoints do not embed policies;
    bundles do. The disclosures say so, so an absent policy is not mistaken for
    lost work.

    See Also
    --------
    imitation_status_for_session : The same report, read off a Session.
    rl_status : The reinforcement learning counterpart.
    """
    from buildml.rl.catalog import rl_capability_matrix

    records = list(history or [])
    saw = any(
        str(r.get("operation_id") or r.get("action"))
        in {
            "fit_imitation",
            "predict_imitation_action",
            "evaluate_imitation",
            "save_imitation_bundle",
            "load_imitation_bundle",
        }
        for r in records
    )
    enabled = imitation_plan is not None
    disclosures: list[str] = []
    if imitation_plan is not None:
        disclosures.extend(
            [
                f"ImitationPlan task={getattr(imitation_plan, 'task', None)}, "
                f"estimator={getattr(imitation_plan, 'estimator', None)}, "
                f"action_column={getattr(imitation_plan, 'action_column', None)}, "
                f"n_train_rows={getattr(imitation_plan, 'n_train_rows', None)}.",
                "Behavioral cloning fits on Session train demonstrations only.",
                "Session checkpoints do not embed ImitationPlan; use "
                "save_imitation_bundle / load_imitation_bundle.",
                "Honesty: BC from tables — not inverse RL / DAgger / robotics.",
            ]
        )
        for note in getattr(imitation_plan, "disclosures", ()) or ():
            disclosures.append(str(note))
    elif saw and not enabled:
        disclosures.append(
            "Imitation operations appear in history, but no live ImitationPlan is attached."
        )

    eval_payload = None
    if eval_result is not None:
        eval_payload = (
            eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
        )
        disclosures.append(
            "Last imitation eval: "
            f"partition={eval_payload.get('partition')}, "
            f"metrics={eval_payload.get('metrics')}."
        )

    return {
        "enabled": enabled,
        "present": enabled or saw,
        "has_imitation_plan": imitation_plan is not None,
        "task": None if imitation_plan is None else getattr(imitation_plan, "task", None),
        "estimator": (
            None if imitation_plan is None else getattr(imitation_plan, "estimator", None)
        ),
        "action_column": (
            None
            if imitation_plan is None
            else getattr(imitation_plan, "action_column", None)
        ),
        "has_fit_result": fit_result is not None,
        "has_eval_result": eval_result is not None,
        "eval": eval_payload,
        "capability_matrix": rl_capability_matrix(),
        "disclosures": disclosures,
        "boundary": (
            "Imitation learning here is behavioral cloning from demonstration "
            "tables (state→action) on train only. Not a robotics / MuJoCo stack."
        ),
    }


def imitation_status_for_session(session: Any) -> dict[str, Any]:
    """Report imitation status by reading the plan and history off a session.

    The convenience form of :func:`imitation_status` for callers that already
    hold a session.

    Parameters
    ----------
    session:
        A :class:`~buildml.session.Session`. Attributes are read defensively, so
        a session with no imitation work reports an empty status rather than
        failing.

    Returns
    -------
    dict
        The same structure :func:`imitation_status` returns.

    See Also
    --------
    imitation_status : The underlying report, and what each field means.
    """
    return imitation_status(
        getattr(session, "_imitation_plan", None),
        fit_result=getattr(session, "_imitation_fit_result", None),
        eval_result=getattr(session, "_imitation_eval_result", None),
        history=list(getattr(session, "_history", ()) or ()),
    )


def rl_status(
    rl_plan: Any = None,
    *,
    fit_result: Any = None,
    eval_result: Any = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Describe where the RL side of a session currently stands.

    Answers "what policy learning has happened here and what should I know
    about it" — for a walkthrough, a status display, or an audit record.

    Parameters
    ----------
    rl_plan:
        The attached policy, if any.
    fit_result:
        The most recent fit report.
    eval_result:
        The most recent evaluation, surfaced in the disclosures along with
        whether it was offline.
    history:
        Session history records, scanned for RL operations.

    Returns
    -------
    dict
        Whether a policy is attached, whether RL appears in history at all, the
        mode, algorithm, arm count and environment, the evaluation payload, the
        current capability matrix, the disclosures, and a statement of scope.

    Notes
    -----
    **The evaluation disclosure names ``offline`` explicitly.** In a walkthrough
    it is the difference between "this policy earned 14.2" and "this policy is
    estimated to have earned 14.2 had it been running", and only the second is
    true of a bandit.

    **A session can show RL activity with no policy attached**, most often after
    reloading a checkpoint — checkpoints do not embed policies, bundles do.

    See Also
    --------
    rl_status_for_session : The same report, read off a Session.
    imitation_status : The imitation counterpart.
    """
    from buildml.rl.catalog import rl_capability_matrix

    records = list(history or [])
    saw = any(
        str(r.get("operation_id") or r.get("action"))
        in {
            "fit_rl",
            "act_rl",
            "evaluate_rl",
            "save_rl_bundle",
            "load_rl_bundle",
        }
        for r in records
    )
    enabled = rl_plan is not None
    disclosures: list[str] = []
    if rl_plan is not None:
        disclosures.extend(
            [
                f"RlPlan mode={getattr(rl_plan, 'mode', None)}, "
                f"backend={getattr(rl_plan, 'backend', None)}, "
                f"algorithm={getattr(rl_plan, 'algorithm', None)}, "
                f"n_arms={getattr(rl_plan, 'n_arms', None)}, "
                f"env_id={getattr(rl_plan, 'env_id', None)}.",
                "Contextual bandits fit on train logged data only; "
                "holdout metrics are offline (DM/IPS).",
                "gym_reinforce requires buildml[rl] and hosts an env policy on the Session.",
                "tabular_q requires buildml[rl] and hosts a Q-table policy "
                "(q_learning / sarsa / expected_sarsa / double_q_learning).",
                "gym_sb3 requires buildml[rl-industry] (SB3 PPO/DQN/A2C).",
                "Session checkpoints do not embed RlPlan; use save_rl_bundle / load_rl_bundle.",
                "Honesty: Session bandit / small-env RL — not MuJoCo/robotics/multi-agent.",
            ]
        )
        for note in getattr(rl_plan, "disclosures", ()) or ():
            disclosures.append(str(note))
    elif saw and not enabled:
        disclosures.append(
            "RL operations appear in history, but no live RlPlan is attached."
        )

    eval_payload = None
    if eval_result is not None:
        eval_payload = (
            eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
        )
        disclosures.append(
            "Last RL eval: "
            f"mode={eval_payload.get('mode')}, offline={eval_payload.get('offline')}, "
            f"metrics={eval_payload.get('metrics')}."
        )

    return {
        "enabled": enabled,
        "present": enabled or saw,
        "has_rl_plan": rl_plan is not None,
        "mode": None if rl_plan is None else getattr(rl_plan, "mode", None),
        "algorithm": None if rl_plan is None else getattr(rl_plan, "algorithm", None),
        "n_arms": None if rl_plan is None else getattr(rl_plan, "n_arms", None),
        "env_id": None if rl_plan is None else getattr(rl_plan, "env_id", None),
        "has_fit_result": fit_result is not None,
        "has_eval_result": eval_result is not None,
        "eval": eval_payload,
        "capability_matrix": rl_capability_matrix(),
        "disclosures": disclosures,
        "boundary": (
            "RL here covers contextual bandits, tabular TD control "
            "(Q-learning / SARSA family), REINFORCE-lite, and optional SB3 on "
            "small Gymnasium envs. Not MuJoCo / robotics / multi-agent."
        ),
    }


def rl_status_for_session(session: Any) -> dict[str, Any]:
    """Report RL status by reading the plan and history off a session.

    The convenience form of :func:`rl_status` for callers that already hold a
    session.

    Parameters
    ----------
    session:
        A :class:`~buildml.session.Session`. Attributes are read defensively, so
        a session with no RL work reports an empty status rather than failing.

    Returns
    -------
    dict
        The same structure :func:`rl_status` returns.

    See Also
    --------
    rl_status : The underlying report, and what each field means.
    """
    return rl_status(
        getattr(session, "_rl_plan", None),
        fit_result=getattr(session, "_rl_fit_result", None),
        eval_result=getattr(session, "_rl_eval_result", None),
        history=list(getattr(session, "_history", ()) or ()),
    )
