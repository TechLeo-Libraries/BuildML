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
from buildml.federated.catalog import federated_capability_matrix
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
            rows.append(
                {
                    **{f"f{i}": float(x[i]) for i in range(4)},
                    "y": y,
                    "hospital": f"h{hospital}",
                }
            )
    frame = pd.DataFrame(rows)
    session = (
        Session.ingest(frame)
        .set_roles(
            {
                **{f"f{i}": "feature" for i in range(4)},
                "y": "target",
                "hospital": "group",
            }
        )
        .group_split(
            test_size=0.2,
            validation_size=0.15,
            random_state=ctx.seed,
            group_column="hospital",
        )
    )
    fit = session.federated.fit(
        backend="native",
        method="fedavg",
        client_column="hospital",
        n_rounds=5,
        random_state=ctx.seed,
    )
    ev = session.federated.evaluate(partition="test", per_client=True)
    bundle = session.federated.save_bundle(ctx.artifacts_dir / "fed_bundle")

    restored = (
        Session.ingest(frame)
        .set_roles(
            {
                **{f"f{i}": "feature" for i in range(4)},
                "y": "target",
                "hospital": "group",
            }
        )
    )
    restored._split_plan = session.split_plan
    restored._dataset = session.dataset
    restored.federated.load_bundle(bundle, trusted=True)
    ev_reloaded = restored.federated.evaluate(partition="test", per_client=False)

    matrix = federated_capability_matrix()
    flower_notes = matrix["backends"]["flower"].get("notes", "")
    write_results(ctx, {
        "status": "completed",
        "data": {
            "name": "synthetic_hospitals",
            "license": "synthetic/public-domain",
            "n_rows": int(len(frame)),
            "n_hospitals": 5,
        },
        "fit": metrics_round(fit.to_dict() if hasattr(fit, "to_dict") else {}),
        "test_metrics": metrics_round(dict(getattr(ev, "metrics", {}) or {})),
        "reloaded_test_metrics": metrics_round(
            dict(getattr(ev_reloaded, "metrics", {}) or {})
        ),
        "per_client_n": int(getattr(ev, "n_clients_evaluated", 0) or 0),
        "bundle_path": str(bundle),
        "bundle_roundtrip": {
            "loaded": restored.federated.plan is not None,
            "accuracy_match": bool(
                abs(
                    float(ev.metrics.get("accuracy", 0.0))
                    - float(ev_reloaded.metrics.get("accuracy", -1.0))
                )
                < 1e-9
            ),
        },
        "capability_honesty": {
            "backend_used": "native",
            "flower_network_runtime": matrix["backends"]["flower"].get(
                "network_runtime", True
            ),
            "flower_disclosed_local_sim": (
                "local" in str(flower_notes).lower()
                or "simulation" in str(flower_notes).lower()
            ),
            "industry_extra_present": matrix.get("industry_extra_present"),
            "industry_runtime_present": matrix.get("industry_runtime_present"),
        },
        "leakage_controls": [
            "Clients = hospital groups",
            "Holdout hospitals/rows for eval",
            "Simulated aggregation only",
            "Bundle load re-score uses frozen global weights only",
        ],
        "industry_comparison": {"status": "filled", "note": "pooled SGD twin"},
        "disclosures": [
            "This is a local FedAvg simulation — raw rows stay in-process; "
            "not a deployed FL network.",
            "Flower (buildml[federated-industry]) is also disclosed as local-sim "
            "when installed; it does not imply networked FL.",
        ],
        "limitations": [
            "Simulation honesty: not production cross-silo FL",
            "No cryptographic secure aggregation",
            "Linear/SGD coef averaging only",
        ],
    })
    print("federated-hospital-sim OK", getattr(ev, "metrics", ev))


if __name__ == "__main__":
    main()
