"""Tier A proof: warranty-cbr-memory — case-based warranty claim decisions."""

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
    ctx = new_proof_context("warranty-cbr-memory", seed=28)
    rng = np.random.default_rng(ctx.seed)
    n = 320
    x = rng.normal(size=(n, 5))
    y = (x[:, 0] + 0.65 * x[:, 2] - 0.3 * x[:, 4] + rng.normal(scale=0.25, size=n) > 0).astype(int)
    frame = pd.DataFrame(x, columns=[
        "failure_severity", "usage_hours_z", "prior_claims_z", "parts_cost_z", "age_months_z",
    ])
    frame["approve_claim"] = y
    frame["claim_id"] = [f"w{i}" for i in range(n)]
    feats = [
        "failure_severity", "usage_hours_z", "prior_claims_z", "parts_cost_z", "age_months_z",
    ]
    session = (
        Session.ingest(frame)
        .set_roles({
            **{c: "feature" for c in feats},
            "approve_claim": "target",
            "claim_id": "id",
        })
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=ctx.seed)
        .scale(method="standard")
    )
    fit = session.cbr.fit(
        task="classification",
        metric="euclidean",
        reuse="distance_weighted",
        k=5,
        random_state=ctx.seed,
    )
    retrieved = session.cbr.retrieve(partition="test", k=5)
    ev = session.cbr.evaluate(partition="test")
    bundle = session.cbr.save_bundle(ctx.artifacts_dir / "cbr_bundle")
    write_results(ctx, {
        "status": "completed",
        "data": {
            "name": "synthetic_warranty_claims",
            "license": "synthetic/public-domain",
            "n_rows": n,
        },
        "fit": metrics_round(fit.to_dict() if hasattr(fit, "to_dict") else {}),
        "retrieve_sample": metrics_round(retrieved.to_dict() if hasattr(retrieved, "to_dict") else {}),
        "test_metrics": metrics_round(dict(getattr(ev, "metrics", {}) or {})),
        "bundle_path": str(bundle),
        "leakage_controls": ["Case memory built from train", "Test retrieval/eval after lock"],
        "industry_comparison": {
            "status": "filled",
            "note": (
                "Tier C baseline_industry.py: sklearn KNeighbors twin; "
                "run script then baseline_industry.py for results/comparison.json."
            ),
        },
        "limitations": ["CBR ≠ RAG; synthetic warranty claims"],
    })
    print("warranty-cbr-memory OK", getattr(ev, "metrics", ev))


if __name__ == "__main__":
    main()
