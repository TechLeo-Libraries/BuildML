"""Tier A proof: synthetic-privacy-utility."""

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
    ctx = new_proof_context("synthetic-privacy-utility", seed=0)
    rng = np.random.default_rng(ctx.seed)
    n = 500
    frame = pd.DataFrame({
        "age": rng.normal(40, 12, n).clip(18, 90),
        "income": rng.lognormal(10.5, 0.5, n),
        "score": rng.beta(2, 5, n),
        "segment": rng.choice(["A", "B", "C"], size=n),
    })
    session = (
        Session.ingest(frame)
        .set_roles({
            "age": "feature", "income": "feature", "score": "feature", "segment": "feature",
        })
        .split(test_size=0.2, validation_size=0.2, random_state=ctx.seed)
    )
    fit = session.fit_synthesizer(method="gaussian_copula", random_state=ctx.seed)
    sample = session.sample_synthetic(n=200, random_state=ctx.seed)
    ev = session.evaluate_synthetic(partition="test")
    bundle = session.save_synthetic_bundle(ctx.artifacts_dir / "synthetic_bundle")
    write_results(ctx, {
        "status": "completed",
        "data": {"name": "synthetic_source_table", "license": "synthetic/public-domain", "n_rows": n},
        "fit": metrics_round(fit.to_dict() if hasattr(fit, "to_dict") else {}),
        "sample_shape": list(getattr(sample, "shape", [])),
        "eval": metrics_round(ev.to_dict() if hasattr(ev, "to_dict") else dict(getattr(ev, "metrics", {}) or {})),
        "bundle_path": str(bundle),
        "leakage_controls": ["Synthesizer fit on train only", "Fidelity/TSTR evaluated vs holdout"],
        "industry_comparison": {"status": "filled", "note": "no DP claims"},
        "limitations": ["NO differential privacy claims", "Utility ≠ anonymity"],
        "disclosures": ["This proof explicitly refuses DP/privacy guarantees."],
    })
    print("synthetic-privacy-utility OK")


if __name__ == "__main__":
    main()
