"""Tier B product: Vector Control Deck.

Composes imitation learning (+ optional gym RL) + decision/optimize allocation
+ classical supervised action baseline. Gymnasium path may skip.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs._lib.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.core.errors import MissingExtraError
from proofs._lib import (
    TORCH_STATUS,
    assert_no_test_in_selection,
    extra_available,
    metrics_round,
    new_proof_context,
    write_results,
)


def _control_demos(n: int = 560, seed: int = 48) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    state = rng.normal(size=(n, 4))
    action = (state[:, 2] + 0.3 * state[:, 3] > 0).astype(int)
    # Intervention cost for capacity decisions
    cost = np.where(action == 1, 2.0, 1.0)
    frame = pd.DataFrame(state, columns=["x", "x_dot", "theta", "theta_dot"])
    frame["action"] = action
    frame["cost"] = cost
    frame["traj_id"] = [f"t-{i}" for i in range(n)]
    meta = {
        "name": "vector_control_demos",
        "license": "synthetic/public-domain",
        "n_rows": n,
        "action_rate": float(action.mean()),
    }
    return frame, meta


def main() -> None:
    ctx = new_proof_context("vector-control-deck", seed=48)
    frame, data_meta = _control_demos(seed=ctx.seed)
    stages: dict = {}
    skip_notes: list[str] = []
    feats = ["x", "x_dot", "theta", "theta_dot"]

    session = (
        Session.ingest(frame.copy())
        .set_roles(
            {
                **{c: "feature" for c in feats},
                "action": "target",
                "cost": "ignore",
                "traj_id": "id",
            }
        )
        .split(test_size=0.2, validation_size=0.2, random_state=ctx.seed)
    )
    plan = session.split_plan
    assert plan is not None
    split_counts = {
        "train": len(plan.train_indices),
        "validation": len(plan.validation_indices),
        "test": len(plan.test_indices),
    }

    # --- Stage 1: imitation learning ---
    try:
        il_fit = session.fit_imitation(
            method="behavioral_cloning", random_state=ctx.seed
        )
        il_val = session.evaluate_imitation(partition="validation")
        il_test = session.evaluate_imitation(partition="test")
        stages["imitation"] = {
            "status": "ok",
            "fit": metrics_round(il_fit.to_dict() if hasattr(il_fit, "to_dict") else {}),
            "validation_metrics": metrics_round(
                dict(getattr(il_val, "metrics", {}) or {})
            ),
            "test_metrics": metrics_round(dict(getattr(il_test, "metrics", {}) or {})),
        }
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["imitation"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"imitation: {exc}")
    write_results(ctx, stages["imitation"], filename="imitation.json")

    # --- Stage 1b: optional gym RL probe ---
    rl_probe: dict = {
        "gymnasium_available": extra_available("gymnasium"),
        "ran": False,
        "skip_torch_paths": TORCH_STATUS.get("skip_torch_paths", True),
    }
    if extra_available("gymnasium") and not TORCH_STATUS.get("skip_torch_paths"):
        try:
            rf = session.fit_rl(
                mode="gym_reinforce",
                env_id="CartPole-v1",
                total_timesteps=800,
            )
            rl_probe = {
                "gymnasium_available": True,
                "ran": True,
                "fit": metrics_round(rf.to_dict() if hasattr(rf, "to_dict") else {}),
            }
        except (MissingExtraError, TypeError, ValueError) as exc:
            rl_probe["error"] = f"{type(exc).__name__}: {exc}"
            skip_notes.append(f"rl: {exc}")
    else:
        rl_probe["reason"] = "gymnasium missing and/or torch skip"
        skip_notes.append("rl: gymnasium missing and/or torch skip")
    stages["rl"] = {"status": "ok" if rl_probe.get("ran") else "skipped", **rl_probe}
    write_results(ctx, stages["rl"], filename="rl.json")

    # --- Stage 2: classical action baseline ---
    c_session = (
        Session.ingest(frame.copy())
        .set_roles(
            {
                **{c: "feature" for c in feats},
                "action": "target",
                "cost": "ignore",
                "traj_id": "id",
            }
        )
        .inject_split(
            train_indices=list(plan.train_indices),
            validation_indices=list(plan.validation_indices),
            test_indices=list(plan.test_indices),
        )
        .scale(method="standard")
    )
    c_session.fit(
        LogisticRegression(max_iter=1000, random_state=ctx.seed),
        task="classification",
    )
    c_val = c_session.evaluate(partition="validation")
    c_test = c_session.evaluate(partition="test")
    stages["supervised"] = {
        "status": "ok",
        "estimator": "LogisticRegression",
        "validation_metrics": metrics_round(dict(c_val.metrics)),
        "test_metrics": metrics_round(dict(c_test.metrics)),
    }
    write_results(ctx, stages["supervised"], filename="supervised.json")

    # --- Stage 3: decision / capacity optimize ---
    try:
        assert_no_test_in_selection(
            selection_partition="validation", evaluation_partition="test"
        )
        thr = c_session.fit_decision_policy(
            method="threshold",
            partition="validation",
            fp_cost=1.0,
            fn_cost=2.0,
        )
        thr_test = c_session.evaluate_decisions(partition="test")
        stages["decisions"] = {
            "status": "ok",
            "threshold_policy": metrics_round(
                thr.to_dict() if hasattr(thr, "to_dict") else {}
            ),
            "threshold_test": metrics_round(
                thr_test.to_dict() if hasattr(thr_test, "to_dict") else {}
            ),
        }
        try:
            knap = c_session.fit_decision_policy(
                method="knapsack",
                partition="validation",
                budget=50.0,
                cost_column="cost",
                id_column="traj_id",
                score_source="model_proba",
                knapsack_solver="dp",
            )
            applied = c_session.apply_decisions(partition="test")
            stages["decisions"]["allocation"] = {
                "status": "ok",
                "policy": metrics_round(
                    knap.to_dict() if hasattr(knap, "to_dict") else {}
                ),
                "applied": {
                    "n_selected": int(applied.n_selected),
                    "selected_value": float(applied.selected_value),
                    "selected_cost": float(applied.selected_cost),
                },
            }
        except Exception as exc:  # noqa: BLE001
            topk = c_session.fit_decision_policy(
                method="topk",
                partition="validation",
                capacity=35,
                score_source="model_proba",
            )
            applied = c_session.apply_decisions(partition="test")
            stages["decisions"]["allocation"] = {
                "status": "ok_topk_fallback",
                "error": f"{type(exc).__name__}: {exc}",
                "policy": metrics_round(
                    topk.to_dict() if hasattr(topk, "to_dict") else {}
                ),
                "applied": {"n_selected": int(applied.n_selected)},
            }
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["decisions"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"decisions: {exc}")
    write_results(ctx, stages["decisions"], filename="decisions.json")

    summary = {
        "status": "completed",
        "product": "Vector Control Deck",
        "data": data_meta,
        "split": {"kind": plan.kind, "counts": split_counts},
        "stages": {k: {"status": v.get("status")} for k, v in stages.items()},
        "stage_details": stages,
        "skip_notes": skip_notes,
        "torch": TORCH_STATUS,
        "leakage_controls": [
            "BC / classical fit on train expert rows only",
            "Optional gym RL is a separate env probe (not test-label conditioned)",
            "Decision policies selected on validation only",
            "Test imitation / supervised metrics after lock",
        ],
        "what_fails_if_leakage_ignored": [
            "BC trained on test trajectories overstates policy cloning skill",
            "Capacity policies tuned on test understate intervention cost",
            "Reporting gym returns without disclosing env/eval separation misleads",
        ],
        "limitations": [
            "Synthetic cartpole-ish demos — not a physical plant controller",
            "Gymnasium RL optional and may skip",
        ],
    }
    write_results(ctx, summary, filename="summary.json")
    write_results(ctx, summary, filename="results.json")
    print(
        "vector-control-deck OK",
        {
            "supervised_acc": stages["supervised"]["test_metrics"].get("accuracy"),
            "rl_ran": stages["rl"].get("ran"),
            "skips": skip_notes,
        },
    )


if __name__ == "__main__":
    main()
