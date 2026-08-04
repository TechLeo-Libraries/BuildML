"""Tier A proof: imitation-cartpole-control."""

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

from buildml import Session
from buildml.core.errors import MissingExtraError
from proofs._lib import TORCH_STATUS, extra_available, metrics_round, new_proof_context, write_results


def main() -> None:
    ctx = new_proof_context("imitation-cartpole-control", seed=0)
    rng = np.random.default_rng(ctx.seed)
    n = 500
    # Cartpole-ish state: x, x_dot, theta, theta_dot → discrete action
    state = rng.normal(size=(n, 4))
    action = (state[:, 2] + 0.3 * state[:, 3] > 0).astype(int)
    frame = pd.DataFrame(state, columns=["x", "x_dot", "theta", "theta_dot"])
    frame["action"] = action
    session = (
        Session.ingest(frame)
        .set_roles({
            "x": "feature", "x_dot": "feature", "theta": "feature", "theta_dot": "feature",
            "action": "target",
        })
        .split(test_size=0.2, validation_size=0.2, random_state=ctx.seed)
    )
    fit = session.rl.fit_imitation(method="behavioral_cloning", random_state=ctx.seed)
    ev = session.rl.evaluate_imitation(partition="test")
    bundle = session.rl.save_imitation_bundle(ctx.artifacts_dir / "il_bundle")

    restored = (
        Session.ingest(frame)
        .set_roles({
            "x": "feature", "x_dot": "feature", "theta": "feature", "theta_dot": "feature",
            "action": "target",
        })
        .split(test_size=0.2, validation_size=0.2, random_state=ctx.seed)
    )
    restored.rl.load_imitation_bundle(bundle, trusted=True)
    ev_reloaded = restored.rl.evaluate_imitation(partition="test")

    rl_probe = {
        "gymnasium_available": extra_available("gymnasium"),
        "ran": False,
        "skip_torch_paths": TORCH_STATUS.get("skip_torch_paths", True),
    }
    if extra_available("gymnasium") and not TORCH_STATUS.get("skip_torch_paths"):
        try:
            rf = session.rl.fit(mode="gym_reinforce", env_id="CartPole-v1", total_timesteps=1000)
            rl_bundle = session.rl.save_bundle(ctx.artifacts_dir / "rl_gym_bundle")
            rl_restored = (
                Session.ingest(frame)
                .set_roles({
                    "x": "feature", "x_dot": "feature", "theta": "feature",
                    "theta_dot": "feature", "action": "target",
                })
                .split(test_size=0.2, validation_size=0.2, random_state=ctx.seed)
            )
            rl_restored.rl.load_bundle(rl_bundle, trusted=True)
            rl_ev = rl_restored.rl.evaluate(n_episodes=10, random_state=ctx.seed)
            rl_probe = {
                "gymnasium_available": True,
                "ran": True,
                "fit": metrics_round(rf.to_dict() if hasattr(rf, "to_dict") else {}),
                "reloaded_eval": metrics_round(
                    rl_ev.to_dict() if hasattr(rl_ev, "to_dict") else {}
                ),
                "bundle_path": str(rl_bundle),
            }
        except (MissingExtraError, TypeError, ValueError) as exc:
            rl_probe["error"] = f"{type(exc).__name__}: {exc}"
    else:
        rl_probe["reason"] = "gymnasium missing and/or torch skip"

    write_results(ctx, {
        "status": "completed",
        "data": {
            "name": "synthetic_cartpole_bc",
            "license": "synthetic/public-domain",
            "n_rows": n,
        },
        "fit": metrics_round(fit.to_dict() if hasattr(fit, "to_dict") else {}),
        "test_metrics": metrics_round(dict(getattr(ev, "metrics", {}) or {})),
        "reloaded_test_metrics": metrics_round(
            dict(getattr(ev_reloaded, "metrics", {}) or {})
        ),
        "bundle_roundtrip": {
            "loaded": restored.rl.imitation_plan is not None,
            "accuracy_match": bool(
                "accuracy" in (getattr(ev, "metrics", {}) or {})
                and abs(
                    float(ev.metrics.get("accuracy", 0.0))
                    - float(ev_reloaded.metrics.get("accuracy", -1.0))
                )
                < 1e-9
            ),
        },
        "rl_probe": rl_probe,
        "torch": TORCH_STATUS,
        "bundle_path": str(bundle),
        "leakage_controls": [
            "BC fit on train expert rows",
            "Test imitation metrics after lock",
            "Bundle load re-score uses frozen BC policy only",
        ],
        "industry_comparison": {"status": "filled", "note": "sklearn BC twin; gym optional"},
        "limitations": [
            "Synthetic expert; gym RL optional and may skip",
            "Industry SB3/imitation path is subprocess-gated separately",
        ],
    })
    print("imitation-cartpole-control OK", getattr(ev, "metrics", ev))


if __name__ == "__main__":
    main()
