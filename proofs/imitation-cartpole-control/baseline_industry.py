"""Tier C: sklearn BC twin (+ graceful gym skip) for imitation-cartpole-control."""

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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from buildml import Session
from proofs._lib import (
    extra_available,
    load_buildml_results,
    metrics_round,
    new_proof_context,
    write_comparison,
)


def main() -> None:
    ctx = new_proof_context("imitation-cartpole-control", seed=0)
    rng = np.random.default_rng(ctx.seed)
    n = 500
    state = rng.normal(size=(n, 4))
    action = (state[:, 2] + 0.3 * state[:, 3] > 0).astype(int)
    frame = pd.DataFrame(state, columns=["x", "x_dot", "theta", "theta_dot"])
    frame["action"] = action

    session = (
        Session.ingest(frame.copy())
        .set_roles(
            {
                "x": "feature",
                "x_dot": "feature",
                "theta": "feature",
                "theta_dot": "feature",
                "action": "target",
            }
        )
        .split(test_size=0.2, validation_size=0.2, random_state=ctx.seed)
    )
    plan = session.split_plan
    assert plan is not None
    tr, te = list(plan.train_indices), list(plan.test_indices)
    cols = ["x", "x_dot", "theta", "theta_dot"]

    scaler = StandardScaler()
    x_tr = scaler.fit_transform(frame.loc[tr, cols])
    x_te = scaler.transform(frame.loc[te, cols])
    clf = LogisticRegression(max_iter=1000, random_state=ctx.seed)
    clf.fit(x_tr, frame.loc[tr, "action"])
    pred = clf.predict(x_te)
    y_te = frame.loc[te, "action"].to_numpy()
    industry_metrics = metrics_round(
        {
            "accuracy": float(accuracy_score(y_te, pred)),
            "macro_f1": float(f1_score(y_te, pred, average="macro", zero_division=0)),
        }
    )

    gym_ok = extra_available("gymnasium")
    gym_probe = {
        "gymnasium_available": gym_ok,
        "ran": False,
        "note": "Gymnasium RL twin skipped — not required for BC comparison",
    }
    if gym_ok:
        try:
            import gymnasium as gym  # noqa: F401

            gym_probe["ran"] = False
            gym_probe["note"] = (
                "gymnasium import OK; full RL twin intentionally not run "
                "(BC table comparison is the Tier C target)"
            )
        except Exception as exc:  # noqa: BLE001
            gym_probe["error"] = f"{type(exc).__name__}: {exc}"

    bml = load_buildml_results(ctx.project_dir)
    bml_metrics = metrics_round(dict(bml.get("test_metrics", {})))

    write_comparison(
        ctx,
        buildml={"backend": "buildml.imitation/behavioral_cloning", "test_metrics": bml_metrics},
        industry={
            "backend": "sklearn.LogisticRegression BC",
            "test_metrics": industry_metrics,
            "gym_probe": gym_probe,
            "leakage_controls": [
                "Same split (seed=0)",
                "BC policy fit on train expert rows only",
            ],
        },
        split_counts={
            "train": len(tr),
            "validation": len(plan.validation_indices),
            "test": len(te),
        },
        delta_keys=("accuracy", "macro_f1"),
        extra={"status": "filled", "gymnasium_available": gym_ok},
    )
    print("imitation-cartpole-control Tier C OK", industry_metrics, gym_probe)


if __name__ == "__main__":
    main()
