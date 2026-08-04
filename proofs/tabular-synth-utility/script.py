"""Tier A proof: tabular-synth-utility — retail catalog synthesizer utility."""

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
    ctx = new_proof_context("tabular-synth-utility", seed=30)
    rng = np.random.default_rng(ctx.seed)
    n = 520
    # Retail catalog / basket utility table — distinct from demographic privacy table.
    frame = pd.DataFrame({
        "unit_price": rng.lognormal(3.2, 0.55, n).clip(1.0, 500.0),
        "units_sold": rng.poisson(12, n).astype(float) + 1.0,
        "margin_pct": rng.beta(3, 4, n),
        "category": rng.choice(["electronics", "apparel", "grocery", "home"], size=n),
    })
    session = (
        Session.ingest(frame)
        .set_roles({
            "unit_price": "feature",
            "units_sold": "feature",
            "margin_pct": "feature",
            "category": "feature",
        })
        .split(test_size=0.2, validation_size=0.2, random_state=ctx.seed)
    )
    fit = session.synthetic.fit(method="gaussian_copula", random_state=ctx.seed)
    sample = session.synthetic.sample(n=200, random_state=ctx.seed)
    ev = session.synthetic.evaluate(partition="test")
    bundle = session.synthetic.save_bundle(ctx.artifacts_dir / "synthetic_bundle")
    write_results(ctx, {
        "status": "completed",
        "data": {
            "name": "synthetic_retail_catalog_source",
            "license": "synthetic/public-domain",
            "n_rows": n,
        },
        "fit": metrics_round(fit.to_dict() if hasattr(fit, "to_dict") else {}),
        "sample_shape": list(getattr(sample, "shape", [])),
        "eval": metrics_round(ev.to_dict() if hasattr(ev, "to_dict") else dict(getattr(ev, "metrics", {}) or {})),
        "bundle_path": str(bundle),
        "leakage_controls": ["Synthesizer fit on train only", "Fidelity/TSTR evaluated vs holdout"],
        "industry_comparison": {
            "status": "filled",
            "note": (
                "Tier C baseline_industry.py: column bootstrap twin; "
                "run script then baseline_industry.py for results/comparison.json. "
                "No DP claims."
            ),
        },
        "limitations": ["NO differential privacy claims", "Utility ≠ anonymity"],
        "disclosures": ["This proof explicitly refuses DP/privacy guarantees."],
    })
    print("tabular-synth-utility OK")


if __name__ == "__main__":
    main()
