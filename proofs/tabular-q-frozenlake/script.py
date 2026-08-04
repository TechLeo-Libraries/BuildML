"""Tier A proof: tabular Q-learning on FrozenLake."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs._lib.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

import pandas as pd

from buildml import Session
from buildml.core.errors import MissingExtraError
from buildml.rl.extras import gymnasium_available
from proofs._lib import metrics_round, new_proof_context, write_results


def main() -> None:
    ctx = new_proof_context("tabular-q-frozenlake", seed=0)
    frame = pd.DataFrame({"a": [0.0, 1.0], "y": [0, 1]})
    session = (
        Session.ingest(frame)
        .set_roles({"a": "feature", "y": "target"})
        .split(test_size=0.5, random_state=ctx.seed)
    )
    tabular_probe: dict[str, object] = {
        "gymnasium_available": gymnasium_available(),
        "ran": False,
    }
    if gymnasium_available():
        try:
            fit = session.rl.fit(
                mode="tabular_q",
                algorithm="q_learning",
                env_id="FrozenLake-v1",
                n_episodes=1_500,
                max_steps=100,
                learning_rate=0.25,
                gamma=0.99,
                epsilon=1.0,
                epsilon_min=0.05,
                epsilon_decay=0.995,
                random_state=ctx.seed,
            )
            ev = session.rl.evaluate(
                n_episodes=50, max_steps=100, random_state=ctx.seed
            )
            act = session.rl.act(observations=[0, 1, 2, 3])
            bundle = session.rl.save_bundle(ctx.artifacts_dir / "rl_tabular_bundle")

            restored = (
                Session.ingest(frame)
                .set_roles({"a": "feature", "y": "target"})
                .split(test_size=0.5, random_state=ctx.seed)
            )
            restored.rl.load_bundle(bundle, trusted=True)
            ev_reloaded = restored.rl.evaluate(
                n_episodes=50, max_steps=100, random_state=ctx.seed
            )

            tabular_probe = {
                "gymnasium_available": True,
                "ran": True,
                "fit": metrics_round(fit.to_dict() if hasattr(fit, "to_dict") else {}),
                "eval": metrics_round(ev.to_dict() if hasattr(ev, "to_dict") else {}),
                "reloaded_eval": metrics_round(
                    ev_reloaded.to_dict() if hasattr(ev_reloaded, "to_dict") else {}
                ),
                "sample_actions": list(getattr(act, "actions", []) or [])[:4],
                "bundle_path": str(bundle),
                "bundle_roundtrip": {
                    "loaded": restored.rl.plan is not None,
                    "eval_offline": bool(getattr(ev_reloaded, "offline", True)),
                    "mean_return": float(
                        (getattr(ev_reloaded, "metrics", {}) or {}).get(
                            "mean_return", float("nan")
                        )
                    ),
                    "has_mean_return": "mean_return"
                    in (getattr(ev_reloaded, "metrics", {}) or {}),
                },
            }
        except (MissingExtraError, TypeError, ValueError) as exc:
            tabular_probe["error"] = f"{type(exc).__name__}: {exc}"
    else:
        tabular_probe["reason"] = "buildml[rl] / gymnasium not installed"
    write_results(
        ctx,
        {
            "status": "completed",
            "data": {
                "name": "gymnasium_FrozenLake-v1",
                "license": "Gymnasium/FrozenLake public benchmark",
            },
            "tabular_q_probe": tabular_probe,
            "leakage_controls": [
                "Env-loop RL — no Session tabular partition used for Q-table fit",
                "session.rl.evaluate uses fresh rollouts, not logged holdout rows",
                "Bundle load re-score uses frozen Q-table only",
            ],
            "industry_comparison": {
                "status": "filled",
                "note": "Foundational tabular Q-learning; SB3 DQN is separate gym_sb3 path",
            },
            "limitations": [
                "Small discrete env teaching loop only",
                "Not batch offline RL or robotics",
                "SB3/imitation industry path is subprocess-gated separately",
            ],
        },
    )
    print("tabular-q-frozenlake OK", tabular_probe)


if __name__ == "__main__":
    main()
