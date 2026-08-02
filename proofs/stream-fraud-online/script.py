"""Tier A proof: stream-fraud-online."""

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
    ctx = new_proof_context("stream-fraud-online", seed=7)
    rng = np.random.default_rng(ctx.seed)
    x0 = rng.normal([-1.0, -1.0], 0.55, size=(220, 2))
    x1 = rng.normal([1.2, 1.0], 0.55, size=(220, 2))
    frame = pd.DataFrame(np.vstack([x0, x1]), columns=["amount_z", "velocity_z"])
    frame["is_fraud"] = [0] * 220 + [1] * 220
    session = (
        Session.ingest(frame)
        .set_roles({"amount_z": "feature", "velocity_z": "feature", "is_fraud": "target"})
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=ctx.seed)
        .scale(method="standard")
    )
    fit = session.fit_online(
        estimator="sgd_classifier", chunk_size=50, n_init=50, classes=[0, 1],
    )
    updates = []
    while True:
        plan = session.online_plan
        remaining = plan.n_train_rows - plan.cursor
        if remaining <= 0:
            break
        u = session.partial_fit_online(n_rows=min(50, remaining))
        updates.append({
            "n_updates": int(u.n_updates),
            "n_chunk_rows": int(u.n_chunk_rows),
            "n_seen_rows": int(u.n_seen_rows),
        })
    val = session.evaluate_online(partition="validation")
    test = session.evaluate_online(partition="test")
    bundle = session.save_online_bundle(ctx.artifacts_dir / "online_bundle")
    write_results(ctx, {
        "status": "completed",
        "data": {"name": "synthetic_fraud_stream", "license": "synthetic/public-domain", "n_rows": int(len(frame))},
        "fit": {
            "n_init_rows": int(fit.n_init_rows),
            "n_remaining_train": int(fit.n_remaining_train),
            "classes": list(fit.classes),
        },
        "updates": updates,
        "validation_metrics": metrics_round(dict(val.metrics)),
        "test_metrics": metrics_round(dict(test.metrics)),
        "bundle_path": str(bundle),
        "leakage_controls": [
            "partial_fit consumes train cursor only",
            "Validation/test never enter online updates",
        ],
        "industry_comparison": {"status": "stub", "note": "River twin when installed"},
        "limitations": ["Batch chunks, not Kafka/Flink"],
    })
    print("stream-fraud-online OK", dict(test.metrics))


if __name__ == "__main__":
    main()
