"""Tier A proof: edge-fleet-federated — FedAvg across edge device clients."""

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
from proofs._lib import metrics_round, new_proof_context, write_results


def main() -> None:
    ctx = new_proof_context("edge-fleet-federated", seed=34)
    rng = np.random.default_rng(ctx.seed)
    rows = []
    # Edge devices (not hospitals): non-IID sensor fault classification.
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
    session = (
        Session.ingest(frame)
        .set_roles({
            **{f"telemetry_{i}": "feature" for i in range(5)},
            "fault": "target",
            "device_id": "group",
        })
        .group_split(
            test_size=0.2, validation_size=0.15,
            random_state=ctx.seed, group_column="device_id",
        )
    )
    fit = session.federated.fit(
        method="fedavg", client_column="device_id", n_rounds=5, random_state=ctx.seed,
    )
    ev = session.federated.evaluate(partition="test")
    bundle = session.federated.save_bundle(ctx.artifacts_dir / "fed_bundle")
    write_results(ctx, {
        "status": "completed",
        "data": {
            "name": "synthetic_edge_fleet",
            "license": "synthetic/public-domain",
            "n_rows": int(len(frame)),
        },
        "fit": metrics_round(fit.to_dict() if hasattr(fit, "to_dict") else {}),
        "test_metrics": metrics_round(dict(getattr(ev, "metrics", {}) or {})),
        "bundle_path": str(bundle),
        "leakage_controls": [
            "Clients = edge device groups",
            "Holdout devices/rows for eval",
            "Simulated aggregation only",
        ],
        "industry_comparison": {
            "status": "filled",
            "note": (
                "Tier C baseline_industry.py: pooled SGD twin; "
                "run script then baseline_industry.py for results/comparison.json."
            ),
        },
        "disclosures": [
            "Local FedAvg simulation — raw rows stay in-process; not a deployed FL network.",
        ],
        "limitations": [
            "Simulation honesty: not production cross-device FL",
            "Distinct from federated-hospital-sim",
        ],
    })
    print("edge-fleet-federated OK", getattr(ev, "metrics", ev))


if __name__ == "__main__":
    main()
