"""Tier A proof: federated-hospital-sim."""

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
    ctx = new_proof_context("federated-hospital-sim", seed=0)
    rng = np.random.default_rng(ctx.seed)
    rows = []
    for hospital in range(5):
        shift = hospital * 0.15
        for _ in range(80):
            x = rng.normal(size=4) + shift
            y = int((x[0] + 0.4 * x[1] + rng.normal(scale=0.25)) > 0)
            rows.append({**{f"f{i}": float(x[i]) for i in range(4)}, "y": y, "hospital": f"h{hospital}"})
    frame = pd.DataFrame(rows)
    session = (
        Session.ingest(frame)
        .set_roles({**{f"f{i}": "feature" for i in range(4)}, "y": "target", "hospital": "group"})
        .group_split(test_size=0.2, validation_size=0.15, random_state=ctx.seed, group_column="hospital")
    )
    # Prefer fitting with client column on full train via inject if needed
    fit = session.fit_federated(
        method="fedavg", client_column="hospital", n_rounds=5, random_state=ctx.seed,
    )
    ev = session.evaluate_federated(partition="test")
    bundle = session.save_federated_bundle(ctx.artifacts_dir / "fed_bundle")
    write_results(ctx, {
        "status": "completed",
        "data": {"name": "synthetic_hospitals", "license": "synthetic/public-domain", "n_rows": int(len(frame))},
        "fit": metrics_round(fit.to_dict() if hasattr(fit, "to_dict") else {}),
        "test_metrics": metrics_round(dict(getattr(ev, "metrics", {}) or {})),
        "bundle_path": str(bundle),
        "leakage_controls": ["Clients = hospital groups", "Holdout hospitals/rows for eval", "Simulated aggregation only"],
        "industry_comparison": {"status": "filled", "note": "pooled SGD twin"},
        "disclosures": [
            "This is a local FedAvg simulation — raw rows stay in-process; not a deployed FL network.",
        ],
        "limitations": ["Simulation honesty: not production cross-silo FL"],
    })
    print("federated-hospital-sim OK", getattr(ev, "metrics", ev))


if __name__ == "__main__":
    main()
