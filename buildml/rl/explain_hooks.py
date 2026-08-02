"""History / catalog / walkthrough helpers for imitation + RL operations."""

from __future__ import annotations

from typing import Any


def imitation_fit_summary(fit_result: Any) -> dict[str, Any]:
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
    """Factual walkthrough disclosure for imitation learning."""
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
    """Factual walkthrough disclosure for reinforcement learning."""
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
            "RL here covers contextual bandits, REINFORCE-lite, and optional SB3 "
            "on small Gymnasium envs. Not MuJoCo / robotics / multi-agent."
        ),
    }


def rl_status_for_session(session: Any) -> dict[str, Any]:
    return rl_status(
        getattr(session, "_rl_plan", None),
        fit_result=getattr(session, "_rl_fit_result", None),
        eval_result=getattr(session, "_rl_eval_result", None),
        history=list(getattr(session, "_history", ()) or ()),
    )
