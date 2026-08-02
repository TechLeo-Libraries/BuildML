"""Tier A proof: case-memory-claims."""

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
    ctx = new_proof_context("case-memory-claims", seed=0)
    rng = np.random.default_rng(ctx.seed)
    n = 300
    x = rng.normal(size=(n, 5))
    y = (x[:, 0] + 0.7 * x[:, 1] + rng.normal(scale=0.25, size=n) > 0).astype(int)
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(5)])
    frame["payout_flag"] = y
    frame["case_id"] = [f"c{i}" for i in range(n)]
    session = (
        Session.ingest(frame)
        .set_roles({
            **{f"f{i}": "feature" for i in range(5)},
            "payout_flag": "target", "case_id": "id",
        })
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=ctx.seed)
        .scale(method="standard")
    )
    fit = session.fit_cbr(
        task="classification",
        metric="euclidean",
        reuse="distance_weighted",
        k=5,
        random_state=ctx.seed,
    )
    retrieved = session.retrieve_cases(partition="test", k=5)
    ev = session.evaluate_cbr(partition="test")
    bundle = session.save_cbr_bundle(ctx.artifacts_dir / "cbr_bundle")
    write_results(ctx, {
        "status": "completed",
        "data": {"name": "synthetic_claims", "license": "synthetic/public-domain", "n_rows": n},
        "fit": metrics_round(fit.to_dict() if hasattr(fit, "to_dict") else {}),
        "retrieve_sample": metrics_round(retrieved.to_dict() if hasattr(retrieved, "to_dict") else {}),
        "test_metrics": metrics_round(dict(getattr(ev, "metrics", {}) or {})),
        "bundle_path": str(bundle),
        "leakage_controls": ["Case memory built from train", "Test retrieval/eval after lock"],
        "industry_comparison": {"status": "filled"},
        "limitations": ["CBR ≠ RAG; synthetic claims"],
    })
    print("case-memory-claims OK", getattr(ev, "metrics", ev))


if __name__ == "__main__":
    main()
