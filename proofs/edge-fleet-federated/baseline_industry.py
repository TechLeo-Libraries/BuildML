"""Tier C: pooled centralized SGD twin for edge-fleet-federated."""

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
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from buildml import Session
from proofs._lib import (
    load_buildml_results,
    metrics_round,
    new_proof_context,
    write_comparison,
)


def main() -> None:
    ctx = new_proof_context("edge-fleet-federated", seed=34)
    rng = np.random.default_rng(ctx.seed)
    rows = []
    for device in range(6):
        shift = device * 0.18
        for _ in range(75):
            x = rng.normal(size=5) + shift
            y = int((x[0] + 0.35 * x[2] - 0.2 * x[4] + rng.normal(scale=0.25)) > 0)
            rows.append({
                **{f"telemetry_{i}": float(x[i]) for i in range(5)},
                "fault": y,
                "device_id": f"edge{device}",
            })
    frame = pd.DataFrame(rows)
    cols = [f"telemetry_{i}" for i in range(5)]
    session = (
        Session.ingest(frame.copy())
        .set_roles({
            **{c: "feature" for c in cols},
            "fault": "target",
            "device_id": "group",
        })
        .group_split(
            test_size=0.2, validation_size=0.15,
            random_state=ctx.seed, group_column="device_id",
        )
    )
    plan = session.split_plan
    assert plan is not None
    tr, te = list(plan.train_indices), list(plan.test_indices)

    scaler = StandardScaler()
    x_tr = scaler.fit_transform(frame.loc[tr, cols])
    x_te = scaler.transform(frame.loc[te, cols])
    y_tr = frame.loc[tr, "fault"].to_numpy()
    y_te = frame.loc[te, "fault"].to_numpy()

    clf = SGDClassifier(loss="log_loss", max_iter=800, random_state=ctx.seed, tol=1e-3)
    clf.fit(x_tr, y_tr)
    pred = clf.predict(x_te)
    industry_metrics = metrics_round(
        {
            "accuracy": float(accuracy_score(y_te, pred)),
            "f1_macro": float(f1_score(y_te, pred, average="macro", zero_division=0)),
        }
    )
    bml = load_buildml_results(ctx.project_dir)
    bml_metrics = metrics_round(dict(bml.get("test_metrics", {})))

    write_comparison(
        ctx,
        buildml={"backend": "buildml.federated/fedavg", "test_metrics": bml_metrics},
        industry={
            "backend": "pooled centralized SGDClassifier (same train rows)",
            "test_metrics": industry_metrics,
            "leakage_controls": [
                "Same group_split by device_id (seed=34)",
                "Pooled fit on train-client rows only; test devices untouched",
            ],
            "disclosures": [
                "Centralized twin sees pooled train features in-process — contrast to FedAvg sim, not a privacy claim",
            ],
        },
        split_counts={
            "train": len(tr),
            "validation": len(plan.validation_indices),
            "test": len(te),
        },
        delta_keys=("accuracy", "f1_macro"),
    )
    print("edge-fleet-federated Tier C OK", industry_metrics)


if __name__ == "__main__":
    main()
